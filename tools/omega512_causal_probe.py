"""Fixed-noise visual and explicit-state interventions on existing Omega policies."""

import argparse
import io
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from deployment.model_server.policy_wrapper import (  # noqa: E402
    PolicyServerWrapper,
)

ORDER = [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12, 6, 13]
ACTIVE = [*list(range(6, 12)), 13]


def predict(model, processor, sample, seed, state_mode="full", visual=True, images=None):
    example = dict(
        image=sample["images"] if images is None else images,
        state=processor.apply_state(sample["state"]),
        lang="rotate_qrcode",
    )
    with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
        torch.manual_seed(seed)
        normalized = model.predict_action(
            [example], num_ddim_steps=20, state_conditioning=state_mode, visual_conditioning=visual
        )["normalized_actions"][0]
    return normalized, processor.unapply_actions(normalized)


def metrics(normalized, raw, target, normalized_target, reference):
    return dict(
        right_joint_target_rmse_rad=float(np.sqrt(np.mean((raw[:, 6:12] - target[:, 6:12]) ** 2))),
        right_active_normalized_mse=float(np.mean((normalized[:, ACTIVE] - normalized_target[:, ACTIVE]) ** 2)),
        right_gripper_binary_error=float(np.mean(raw[:, 13] != (target[:, 13] > 0.5))),
        right_joint_prediction_change_rmse_rad=float(np.sqrt(np.mean((raw[:, 6:12] - reference[:, 6:12]) ** 2))),
    )


def normalize_target(target, stats):
    lo, hi = np.asarray(stats["min"]), np.asarray(stats["max"])
    mask = hi != lo
    normalized = np.zeros_like(target)
    normalized[:, mask] = 2 * (target[:, mask] - lo[mask]) / (hi[mask] - lo[mask]) - 1
    normalized[:, 12:] = target[:, 12:] > 0.5
    return normalized


def heldout_samples(checkpoint, count):
    from omegaconf import OmegaConf

    run = checkpoint.parents[1]
    config = OmegaConf.load(run / "config.full.yaml")
    root = Path(config.datasets.vla_data.data_root_dir) / "rotate_qrcode"
    split = json.loads((run / "data_split/episode_split.json").read_text())["rotate_qrcode"]
    episodes = split["validation_episode_ids"][:count]
    samples = []
    for episode in episodes:
        table = pd.read_parquet(root / f"data/chunk-000/episode_{episode:06d}.parquet")
        for frame in (0, len(table) // 3):
            row = table.iloc[frame]
            images = [
                Image.open(io.BytesIO(row[key]["bytes"])).convert("RGB")
                for key in (
                    "observation.images.cam_high",
                    "observation.images.cam_left_wrist",
                    "observation.images.cam_right_wrist",
                )
            ]
            target = np.stack(table.action.iloc[frame : frame + 16].tolist())[:, ORDER]
            if len(target) < 16:
                target = np.concatenate([target, np.repeat(target[-1:], 16 - len(target), axis=0)])
            samples.append(
                dict(
                    episode=int(episode),
                    frame=frame,
                    images=images,
                    state=np.asarray(row["observation.state"])[ORDER],
                    target=target,
                )
            )
    return samples


def summarize(records, key="condition"):
    groups = {}
    for row in records:
        groups.setdefault(row[key], []).append(row["metrics"])
    return {
        group: {metric: float(np.mean([row[metric] for row in rows])) for metric in rows[0]}
        for group, rows in groups.items()
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--scenes", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--noise-seeds", default="4000,4001,4002")
    args = parser.parse_args()
    args.checkpoint = args.checkpoint.resolve()
    torch.set_num_threads(4)
    wrapper = PolicyServerWrapper(str(args.checkpoint), use_bf16=False)
    model, processor = wrapper._framework, wrapper.get_norm_processor()
    stats = json.loads((args.checkpoint.parents[1] / "dataset_statistics.json").read_text())["new_embodiment"]["action"]
    seeds = [int(x) for x in args.noise_seeds.split(",")]
    modes = ["full", "none"] + (["decoder_only", "readout_only"] if model.visual_readout is not None else [])
    report = dict(
        checkpoint=str(args.checkpoint),
        noise_seeds=seeds,
        heldout_records=[],
        controlled_scene_records=[],
        protocol=(
            "Task and diffusion noise fixed within each comparison. State masks remove complete decoder tokens and/or"
            " query offsets, matching training dropout semantics. All cameras rerendered consistently for simulator"
            " interventions."
        ),
        limitations=(
            "Inference masks identify functional dependence, not the training cause of failures. State-only inference is"
            " a stress test; matched state-only training is needed. Static FK/reference chunks do not measure rollout"
            " success."
        ),
    )
    samples = heldout_samples(args.checkpoint, args.episodes)
    for index, sample in enumerate(samples):
        target_normalized = normalize_target(sample["target"], stats)
        for seed in seeds:
            original_norm, original_raw = predict(model, processor, sample, seed)
            variants = [("full", original_norm, original_raw)]
            for mode in modes[1:]:
                norm, raw = predict(model, processor, sample, seed, state_mode=mode)
                variants.append((mode, norm, raw))
            for label, images, visual in [
                ("swapped_images", samples[(index + 2) % len(samples)]["images"], True),
                ("state_only", None, False),
            ]:
                norm, raw = predict(model, processor, sample, seed, images=images, visual=visual)
                variants.append((label, norm, raw))
            for label, norm, raw in variants:
                report["heldout_records"].append(
                    dict(
                        episode=sample["episode"],
                        frame=sample["frame"],
                        noise_seed=seed,
                        condition=label,
                        metrics=metrics(norm, raw, sample["target"], target_normalized, original_raw),
                    )
                )
        print(f"Held-out episode {sample['episode']} frame {sample['frame']} complete", flush=True)
    scenes = json.loads((args.scenes / "scenes.json").read_text())
    for mode in modes:
        for seed in seeds:
            reference = None
            for scene in scenes:  # Base comes first, for paired response metrics.
                data = np.load(scene["scene_file"])
                sample = dict(
                    images=[Image.fromarray(image) for image in data["images"]],
                    state=data["raw_state"][ORDER],
                    target=data["target"],
                )
                norm, raw = predict(model, processor, sample, seed, state_mode=mode)
                if reference is None:
                    reference = raw
                report["controlled_scene_records"].append(
                    dict(
                        scene=scene["name"],
                        state_conditioning=mode,
                        noise_seed=seed,
                        raw_actions=raw.tolist(),
                        metrics=metrics(
                            norm, raw, sample["target"], normalize_target(sample["target"], stats), reference
                        ),
                    )
                )
            print(f"Controlled scenes: {mode}, noise {seed} complete", flush=True)
    report["heldout_summary"] = summarize(report["heldout_records"])
    report["initial_frame_summary"] = summarize([row for row in report["heldout_records"] if row["frame"] == 0])
    report["later_frame_summary"] = summarize([row for row in report["heldout_records"] if row["frame"] != 0])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["heldout_summary"], indent=2), flush=True)


if __name__ == "__main__":
    main()
