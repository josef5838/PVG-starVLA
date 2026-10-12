"""Verify production stream-policy variants on a real task batch."""

import argparse
import copy
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from starVLA.dataloader.lerobot_datasets import (  # noqa: E402
    get_vla_dataset,
)
from starVLA.model.framework.VLM4A.OmegaCameraStreamDiT import (  # noqa: E402
    OmegaCameraStreamDiT,
)
from starVLA.training.trainer_utils.trainer_tools import (  # noqa: E402
    build_param_lr_groups,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--reference", type=Path, help="Require matched initialization, batch images/actions and sample IDs")
    parser.add_argument("--reference-statistics", type=Path, help="Require identical train-only normalization statistics")
    args = parser.parse_args()
    cfg = OmegaConf.load(args.config)
    kind = cfg.framework.streams.conditioning
    cfg.datasets.vla_data.statistics_cache_dir = str(args.out.parent / f"preflight_split_{kind}")
    torch.set_num_threads(4)
    torch.manual_seed(int(cfg.seed))
    torch.cuda.manual_seed(int(cfg.seed))
    dataset = get_vla_dataset(cfg.datasets.vla_data, mode="train")
    batch = [dataset[i] for i in range(int(cfg.datasets.vla_data.per_device_batch_size))]
    if args.reference_statistics:
        statistics_path = args.out.parent / "preflight_dataset_statistics.json"
        dataset.save_dataset_statistics(statistics_path)
        if json.loads(statistics_path.read_text()) != json.loads(args.reference_statistics.read_text()):
            raise ValueError("Three-call reference mismatch: train-only normalization statistics")
    assert all(row["lang"] == cfg.datasets.vla_data.instruction_override for row in batch)
    use_visual = bool(cfg.framework.get("visual_conditioning", True))
    observation_mode = cfg.framework.streams.get("observation_mode", "history")
    reference = json.loads(args.reference.read_text()) if args.reference else None
    reference_batch_size = int(reference["production_batch_size"]) if reference else len(batch)
    assert len(batch) >= reference_batch_size
    image_hash = hashlib.sha256()
    for row_index, row in enumerate(batch):
        assert len(row["image_history"]) == 3 and all(len(v) == 8 for v in row["image_history"])
        single = dataset.datasets[0]
        raw = single.get_step_data(row["sample_episode"], row["sample_frame"])
        for c, key in enumerate(single.modality_keys["video"]):
            for t, image in enumerate(row["image_history"][c]):
                assert image.size == (640, 480)
                np.testing.assert_array_equal(np.asarray(image), raw[key][7 - t])
                if row_index < reference_batch_size:
                    image_hash.update(np.asarray(image).tobytes())
        assert all(
            np.array_equal(np.asarray(view[-1]), np.asarray(now))
            for view, now in zip(row["image_history"], row["image"], strict=False)
        )
    model = OmegaCameraStreamDiT(cfg).cuda().train()
    initial = hashlib.sha256()
    for name, value in model.state_dict().items():
        if not name.startswith("visual_readout."):
            initial.update(name.encode())
            initial.update(value.detach().cpu().numpy().tobytes())
    parity = dict(
        initial_common_weights_sha256=initial.hexdigest(),
        batch_image_sha256=image_hash.hexdigest(),
        batch_action_sha256=hashlib.sha256(np.stack([x["action"] for x in batch[:reference_batch_size]]).tobytes()).hexdigest(),
        sample_keys=[dict(episode=x["sample_episode"], frame=x["sample_frame"]) for x in batch[:reference_batch_size]],
    )
    if args.reference:
        for key, value in parity.items():
            if value != reference[key]:
                raise ValueError(f"Three-call reference mismatch: {key}")
    input_calls = []
    hook = model.omega.aggregator.register_forward_pre_hook(
        lambda module, inputs: input_calls.append(list(inputs[0].shape))
    )
    optimizer = torch.optim.AdamW(build_param_lr_groups(model, cfg), lr=cfg.trainer.learning_rate.base)
    losses = []
    grads = []
    for _step in range(2):
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = model(batch)["action_loss"]
        assert torch.isfinite(loss)
        loss.backward()
        record = {
            n: float(p.grad.float().norm())
            for n, p in model.named_parameters()
            if p.grad is not None
            and n
            in [
                "visual_proj.weight",
                "time_embeddings",
                "view_embeddings",
                "token_type_embeddings",
                "memory_proj.weight",
                "visual_readout.queries",
            ]
        }
        grads.append(record)
        assert all(p.grad is None for p in model.omega.parameters())
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        losses.append(float(loss.detach()))
    hook.remove()
    joint = model.encoding == "joint24"
    assert len(input_calls) == ((2 if joint else 6) if use_visual else 0)
    assert all(s[:3] == [len(batch), 24 if joint else 8, 3] for s in input_calls)
    if use_visual:
        assert grads[-1]["visual_proj.weight"] > 0 and grads[-1]["time_embeddings"] > 0
        if kind == "fused":
            assert grads[-1]["visual_readout.queries"] > 0
    else:
        assert not grads[-1]
    assert model.last_visual_info["memory_tokens"] == (408 if kind == "direct" else 32)
    audit_visual = model.last_visual_info.copy()
    model.eval()
    original = [batch[0]]
    changed = copy.deepcopy(original)
    changed[0]["image_history"][1][0] = Image.new("RGB", (640, 480), (255, 0, 0))
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        condition, memory = model._condition(original)
        other_condition, other_memory = model._condition(changed)
        torch.testing.assert_close(condition, other_condition)
        historical_input_used = use_visual and observation_mode == "history"
        assert (not torch.allclose(memory, other_memory)) == historical_input_used
        noisy = torch.randn(1, 16, 14, device="cuda")
        time = torch.tensor([50], device="cuda")
        first = model.action_model.net(noisy, time, condition, model.action_model.net.prepare_visual(memory))
        second = model.action_model.net(
            noisy, time, other_condition, model.action_model.net.prepare_visual(other_memory)
        )
    difference = float((first.float() - second.float()).abs().mean())
    assert (difference > 0) == historical_input_used
    current = copy.deepcopy(original)
    current[0]["image_history"][2][-1] = Image.new("RGB", (640, 480), (0, 255, 0))
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        _, current_memory = model._condition(current)
    assert (not torch.allclose(memory, current_memory)) == use_visual
    sampled = model.predict_action(original, num_ddim_steps=2)["normalized_actions"]
    assert sampled.shape == (1, 16, 14) and np.isfinite(sampled).all()
    weights = model.state_dict()
    assert not any(k.startswith("omega.") for k in weights)
    model.load_pretrained_state_dict(weights)
    result = dict(
        status="passed",
        variant=kind,
        encoding=model.encoding,
        observation_mode=observation_mode,
        use_visual=use_visual,
        **parity,
        parity_batch_size=reference_batch_size,
        production_sample_keys=[dict(episode=x["sample_episode"], frame=x["sample_frame"]) for x in batch],
        production_batch_size=len(batch),
        chronological_native_image_history_verified=True,
        calls_during_two_updates=input_calls,
        losses=losses,
        gradient_norms=grads,
        visual_info=audit_visual,
        frozen_backbone_no_gradients=True,
        reference_parity_verified=bool(args.reference),
        reference_statistics_verified=bool(args.reference_statistics),
        gpu_name=torch.cuda.get_device_name(),
        torch_version=torch.__version__,
        historical_image_change_prediction_mae=difference,
        trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
        peak_allocated_gpu_gib=torch.cuda.max_memory_allocated() / 2**30,
        note="Disposable two-update preflight; production jobs initialize fresh from seed 42.",
    )
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
