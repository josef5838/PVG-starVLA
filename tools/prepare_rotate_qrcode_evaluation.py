"""Freeze paired post-training action/rollout evaluation using the preregistered scenes."""

import argparse
import datetime
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from launch_rotate_qrcode_streams import (  # noqa: E402
    digest,
    write,
)


def completed_runs(training):
    """Require complete training for every selected run; omit cancelled controls."""
    runs = []
    for run in training["runs"]:
        if run.get("status") == "cancelled":
            continue
        base = Path(run["output_dir"])
        status_path = base / "stream_training_status.json"
        status = json.loads(status_path.read_text()) if status_path.exists() else {}
        if (
            status.get("status") != "completed"
            or status.get("completed_updates") != training["optimizer_updates_per_run"]
        ):
            raise RuntimeError(f"Run not completed: {run['variant']}")
        runs.append(run)
    if not runs:
        raise RuntimeError("No completed runs available for evaluation")
    return runs


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--training-manifest", type=Path, required=True)
    args = p.parse_args()
    training_path = args.training_manifest.resolve()
    training = json.loads(training_path.read_text())
    selected_runs = completed_runs(training)
    registration = json.loads(Path(training["geometry_registration"]).read_text())
    assert registration["status"] == "completed"
    out = training_path.parent / ("evaluation_" + datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d_%H%M%S"))
    rename = {"validation": "heldout_validation", "test": "heldout_test", "train": "train", "unseen_eval": "unseen_eval"}
    mapping = [dict(row, split=rename[row["split"]]) for row in training["trajectory_mapping"]]
    scenes = [dict(row, split=rename[row["split"]]) for row in training["rollout_scenes"]]
    samples = []
    for row in mapping:
        count = 8 if row["split"] == "train" else 16
        for frame in np.unique(np.linspace(0, row["length"] - 1, count).round().astype(int)):
            samples.append(dict(episode=row["episode"], frame=int(frame), length=row["length"], split=row["split"]))
    runs = []
    for run in selected_runs:
        base = Path(run["output_dir"])
        checkpoints = dict(
            best=str(base / "best_model/pytorch_model.pt"),
            final=str(base / "final_model/pytorch_model.pt"),
            update10000=str(base / "checkpoints/steps_10000_pytorch_model.pt"),
        )
        runs.append(
            dict(
                variant=run["variant"],
                gpu=run["gpu"],
                output_dir=str(base),
                checkpoints=checkpoints,
                checkpoint_sha256={k: digest(v) for k, v in checkpoints.items()},
                best_validation=json.loads((base / "best_model/validation.json").read_text()),
            )
        )
    m = dict(
        task=training["task"],
        status="prepared",
        training_manifest=str(training_path),
        source=training["source"],
        dataset=training["dataset"],
        trajectory_mapping=mapping,
        rollout_scenes=scenes,
        offline_samples=samples,
        runs=runs,
        excluded_runs=[
            dict(variant=run["variant"], reason="cancelled before training")
            for run in training["runs"]
            if run.get("status") == "cancelled"
        ],
        fresh_scene_registration=training["geometry_registration"],
        initial_image_validation=training["initial_image_validation"],
        renderer_runtime="/export/home/haoliang/.cache/robotwin-nvidia-595.71.05",
        warp_runtime="/export/home/haoliang/.cache/robotwin-warp-1.9.1",
        rollout_checkpoint="best",
        execute_horizon=8,
        execute_horizon_ablation=4,
        num_ddim_steps=20,
        diffusion_seed=4000,
        history_length=8,
        history_stride=1,
        image_size=[640, 480],
        camera_order=training["camera_order"],
        heldout_note="Validation selects checkpoints; heldout_test episodes and fresh scenes do not.",
        fresh_evaluation_scope="50 stable right-arm conditional scenes; no expert/policy-success filtering.",
    )
    out.mkdir()
    write(out / "manifest.json", m)
    print(out / "manifest.json")


if __name__ == "__main__":
    main()
