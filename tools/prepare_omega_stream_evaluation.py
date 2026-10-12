"""Register paired trajectory evaluation and verify converted-to-source mappings."""

import argparse
import datetime
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SOURCE = Path("/export/home/haoliang/projects/RoboTwin/data/adjust_bottle/demo_clean")
DATA = Path("/export/home/haoliang/projects/starVLA/playground/Datasets/RoboTwin/adjust_bottle")


def digest(path):
    value = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-manifest", type=Path, required=True)
    args = parser.parse_args()
    training = json.loads(args.training_manifest.read_text())
    out = args.training_manifest.parent / (
        "evaluation_" + datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d_%H%M%S")
    )
    out.mkdir()
    seeds = list(map(int, (SOURCE / "seed.txt").read_text().split()))
    sources = {}
    for path in SOURCE.glob("data/episode*.hdf5"):
        with h5py.File(path) as file:
            sources[int(path.stem[7:])] = file["joint_action/vector"][:].astype(np.float32)
    mapping = []
    split = json.loads((Path(training["runs"][0]["output_dir"]) / "data_split/episode_split.json").read_text())[
        "adjust_bottle"
    ]
    for episode in range(50):
        df = pd.read_parquet(DATA / f"data/chunk-000/episode_{episode:06d}.parquet")
        state = np.stack(df["observation.state"]).astype(np.float32)
        action = np.stack(df["action"]).astype(np.float32)
        matches = [
            sid
            for sid, raw in sources.items()
            if raw.shape[0] == len(state) + 1
            and np.allclose(state, raw[:-1], rtol=0, atol=1e-6)
            and np.allclose(action, raw[1:], rtol=0, atol=1e-6)
        ]
        if len(matches) != 1:
            raise RuntimeError(f"Episode {episode}: ambiguous/nonexistent source {matches}")
        sid = matches[0]
        mapping.append(
            dict(
                episode=episode,
                source_episode=sid,
                seed=seeds[sid],
                length=len(state),
                split="train" if episode in split["train_episode_ids"] else "heldout_validation",
                observation_matches_source_t=True,
                action_matches_source_t_plus_1=True,
            )
        )
    selected_train = [0, 10, 22, 35, 48]
    scenes = [dict(row) for row in mapping if row["episode"] in selected_train + split["validation_episode_ids"]]
    scenes += [dict(split="unseen_eval", episode=None, source_episode=None, seed=100000 * (i + 1)) for i in range(5)]
    offline = []
    for row in mapping:
        # Equal trajectory weighting; include start, manipulation and final phases.
        count = 8 if row["split"] == "train" else 16
        for frame in sorted(set(np.linspace(0, row["length"] - 1, count).round().astype(int).tolist())):
            offline.append(dict(episode=row["episode"], frame=frame, length=row["length"], split=row["split"]))
    runs = []
    for run in training["runs"]:
        base = Path(run["output_dir"])
        best = json.loads((base / "best_model/validation.json").read_text())
        cps = {kind: str(base / f"{kind}_model/pytorch_model.pt") for kind in ("best", "final")}
        hashes = {kind: digest(path) for kind, path in cps.items()}
        metrics = [json.loads(x) for x in (base / "stream_training_metrics.jsonl").read_text().splitlines()]
        status = json.loads((base / "stream_training_status.json").read_text())
        assert status["status"] == "completed" and status["completed_updates"] == 10000
        runs.append(
            dict(
                variant=run["variant"],
                gpu=run["gpu"],
                output_dir=str(base),
                checkpoints=cps,
                checkpoint_sha256=hashes,
                best_validation=best,
                final_validation_mse=metrics[-1]["validation_action_mse"],
                last_100_logged_train_loss_mean=float(np.mean([r["action_dit_loss"] for r in metrics[-100:]])),
                training_status=status,
            )
        )
    manifest = dict(
        task="adjust_bottle",
        status="prepared",
        training_manifest=str(args.training_manifest.resolve()),
        source=str(SOURCE),
        dataset=str(DATA),
        trajectory_mapping=mapping,
        rollout_scenes=scenes,
        offline_samples=offline,
        runs=runs,
        rollout_checkpoint="best",
        diffusion_seed=4000,
        execute_horizon=8,
        num_ddim_steps=20,
        history_length=8,
        history_stride=1,
        history_sampling=(
            "Every policy control observation, before chunk reuse/replanning; first observation left-padded."
        ),
        image_size=[640, 480],
        camera_order=["head_camera", "left_camera", "right_camera"],
        heldout_note=(
            "The five heldout trajectories were used for checkpoint selection; fresh eval seeds are independent of"
            " selection."
        ),
        offline_note=(
            "Teacher-forced recorded histories/states. DDIM20 noise seed4000+batch_index; paired sample/batch order"
            " across variants and checkpoints. Report padded and valid-future-only errors."
        ),
    )
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    training["status"] = "completed"
    training["evaluation_manifest"] = str(out / "manifest.json")
    for run in training["runs"]:
        run["status"] = "completed"
        run["confirmed_completed_updates"] = 10000
    args.training_manifest.write_text(json.dumps(training, indent=2) + "\n")
    print(out / "manifest.json")


if __name__ == "__main__":
    main()
