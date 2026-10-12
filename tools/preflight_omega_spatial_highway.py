"""Real-batch C preflight: matched baseline, zero-gate parity and opening gradients."""

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from starVLA.dataloader.lerobot_datasets import get_vla_dataset  # noqa: E402
from starVLA.model.framework.VLM4A.OmegaCameraStreamDiT import OmegaCameraStreamDiT  # noqa: E402
from starVLA.training.trainer_utils.trainer_tools import build_param_lr_groups  # noqa: E402


def is_spatial(name):
    return any(part.startswith("spatial_") for part in name.split("."))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--reference-statistics", type=Path, required=True)
    args = parser.parse_args()
    cfg = OmegaConf.load(args.config)
    assert cfg.datasets.vla_data.per_device_batch_size == 4 and cfg.trainer.gradient_accumulation_steps == 2
    reference = json.loads(args.reference.read_text())
    torch.set_num_threads(4)
    torch.manual_seed(int(cfg.seed))
    torch.cuda.manual_seed(int(cfg.seed))
    cfg.datasets.vla_data.statistics_cache_dir = str(args.out.parent / "preflight_data_split")
    dataset = get_vla_dataset(cfg.datasets.vla_data, mode="train")
    statistics = args.out.parent / "preflight_dataset_statistics.json"
    dataset.save_dataset_statistics(statistics)
    assert json.loads(statistics.read_text()) == json.loads(args.reference_statistics.read_text())
    batch = [dataset[i] for i in range(4)]
    image_hash = hashlib.sha256()
    single = dataset.datasets[0]
    for row in batch:
        raw = single.get_step_data(row["sample_episode"], row["sample_frame"])
        for camera, key in enumerate(single.modality_keys["video"]):
            for timestamp, image in enumerate(row["image_history"][camera]):
                assert image.size == (640, 480)
                np.testing.assert_array_equal(np.asarray(image), raw[key][7-timestamp])
                image_hash.update(np.asarray(image).tobytes())
    parity = dict(batch_image_sha256=image_hash.hexdigest(),
                  batch_action_sha256=hashlib.sha256(np.stack([row["action"] for row in batch]).tobytes()).hexdigest(),
                  sample_keys=[dict(episode=row["sample_episode"], frame=row["sample_frame"]) for row in batch])
    model = OmegaCameraStreamDiT(cfg).cuda().train()
    initial = hashlib.sha256()
    for name, value in model.state_dict().items():
        if not is_spatial(name):
            initial.update(name.encode())
            initial.update(value.detach().cpu().numpy().tobytes())
    parity["initial_common_weights_sha256"] = initial.hexdigest()
    for key, value in parity.items():
        assert value == reference[key], f"Reference mismatch: {key}"

    # Probe a nonzero common output head so initial parity is not vacuous.
    head = model.action_model.net.final_layer.linear
    saved_head = head.weight.detach().clone()
    torch.nn.init.normal_(head.weight, std=0.02)
    model.eval()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        condition, memory = model._condition(batch)
        assert memory["registers"].shape == (4, 408, 1024)
        assert memory["spatial"].shape == (4, 3108, 1024)
        assert model._current_patch_tokens.shape == (4, 3, 1036, 2048)
        noise = torch.randn(4, 16, 14, device="cuda")
        timestep = torch.full((4,), 50, device="cuda", dtype=torch.long)
        baseline = model.action_model.net(noise, timestep, condition,
                                         model.action_model.net.prepare_visual(memory["registers"]))
        highway = model.action_model.net(noise, timestep, condition,
                                        model.action_model.net.prepare_visual(memory))
        torch.testing.assert_close(baseline, highway, rtol=0, atol=0)
    head.weight.data.copy_(saved_head)
    del saved_head, condition, memory, baseline, highway, noise, timestep
    model.train()
    optimizer = torch.optim.AdamW(build_param_lr_groups(model, cfg), lr=cfg.trainer.learning_rate.base,
                                 betas=tuple(cfg.trainer.optimizer.betas), eps=cfg.trainer.optimizer.eps,
                                 weight_decay=cfg.trainer.optimizer.weight_decay, fused=True)
    calls, gradients, losses, seconds = [], [], [], []
    hook = model.omega.aggregator.register_forward_pre_hook(lambda module, inputs: calls.append(list(inputs[0].shape)))
    for update in range(4):
        torch.cuda.synchronize()
        started = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        total_loss = 0.0
        for _ in range(2):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = model(batch)["action_loss"] / 2
            assert torch.isfinite(loss)
            loss.backward()
            total_loss += float(loss.detach())
            del loss
        record = {name: float(parameter.grad.float().norm()) for name, parameter in model.named_parameters()
                  if parameter.grad is not None and name in {
                      "spatial_proj.weight", "spatial_position_proj.0.weight", "spatial_output_proj.weight",
                      "action_model.net.spatial_gates", "action_model.net.spatial_attentions.0.key_value.weight",
                  }}
        assert all(np.isfinite(value) for value in record.values())
        assert all(parameter.grad is None for parameter in model.omega.parameters())
        gradients.append(record)
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.trainer.gradient_clipping)
        optimizer.step()
        torch.cuda.synchronize()
        seconds.append(time.perf_counter()-started)
        losses.append(total_loss)
    hook.remove()
    assert len(calls) == 8 and all(shape == [4,24,3,448,592] for shape in calls)
    assert any(record["action_model.net.spatial_gates"] > 0 for record in gradients)
    assert gradients[-1]["spatial_proj.weight"] > 0
    assert gradients[-1]["spatial_position_proj.0.weight"] > 0
    assert gradients[-1]["action_model.net.spatial_attentions.0.key_value.weight"] > 0
    assert model.action_model.net.spatial_gates.abs().sum() > 0
    model.eval()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        condition, memory = model._condition([batch[0]])
        model.action_model.net.spatial_gates.fill_(0.1)
        noise = torch.randn(1,16,14,device="cuda")
        timestep = torch.tensor([50],device="cuda")
        enabled = model.action_model.net(noise,timestep,condition,model.action_model.net.prepare_visual(memory))
        disabled = model.action_model.net(noise,timestep,condition,model.action_model.net.prepare_visual(memory["registers"]))
        difference = float((enabled.float()-disabled.float()).abs().mean())
        assert difference > 0
    inference_seconds = {}
    for label, active in (("baseline_registers", False), ("current_patch_highway", True)):
        model.highway_active = active
        torch.cuda.synchronize()
        started = time.perf_counter()
        sampled = model.predict_action([batch[0]], num_ddim_steps=20, diffusion_seed=4000)["normalized_actions"]
        torch.cuda.synchronize()
        inference_seconds[label] = time.perf_counter()-started
        assert sampled.shape == (1,16,14) and np.isfinite(sampled).all()
    result = dict(status="passed",**parity,reference_parity_verified=True,reference_statistics_verified=True,
                  zero_gate_baseline_output_exact=True,frozen_backbone_no_gradients=True,
                  production_batch_size=4,gradient_accumulation_steps=2,effective_batch=8,
                  losses=losses,gradient_norms=gradients,calls_during_four_updates=calls,
                  seconds_per_update=seconds,ddim20_single_probe_seconds=inference_seconds,
                  spatial_prediction_sensitivity_mae=difference,
                  trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
                  peak_allocated_gpu_gib=torch.cuda.max_memory_allocated()/2**30,
                  gpu_name=torch.cuda.get_device_name(),torch_version=torch.__version__,
                  visual_info=model.last_visual_info,
                  note="Four disposable updates allow the zero gate and spatial gradients to open; fresh production discards all probe weights.")
    args.out.write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result),flush=True)


if __name__ == "__main__":
    main()
