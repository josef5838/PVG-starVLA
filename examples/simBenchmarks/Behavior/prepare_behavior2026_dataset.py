#!/usr/bin/env python3
"""Validate a BEHAVIOR-1K 2026 LeRobot v3 dataset for StarVLA training."""

import argparse
import hashlib
import json
import shutil
from pathlib import Path


REQUIRED_FEATURES = {
    "action": [23],
    "observation.state": [61],
    "observation.rgb.zed_link_camera_0": [720, 720, 3],
    "observation.rgb.left_realsense_link_camera_0": [480, 480, 3],
    "observation.rgb.right_realsense_link_camera_0": [480, 480, 3],
}
REQUIRED_STATS = ("mean", "std", "min", "max", "q01", "q99")
TRAINING_VIDEO_KEYS = {
    "observation.rgb.zed_link_camera_0",
    "observation.rgb.left_realsense_link_camera_0",
    "observation.rgb.right_realsense_link_camera_0",
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument(
        "--tasks-jsonl",
        type=Path,
        help="Official meta/tasks.jsonl containing natural-language instructions",
    )
    parser.add_argument(
        "--no-manifest",
        action="store_true",
        help="Skip the SHA-256 transfer manifest (faster for a large completed dataset)",
    )
    parser.add_argument(
        "--prune-unused-videos",
        action="store_true",
        help="Remove depth/external video directories after QA; raw HDF5 is not modified",
    )
    return parser.parse_args()


def load_json(path: Path):
    with path.open() as f:
        return json.load(f)


def validate_dataset(root: Path):
    info_path = root / "meta/info.json"
    stats_path = root / "meta/stats.json"
    tasks_path = root / "meta/tasks.parquet"
    for path in (info_path, stats_path, tasks_path, root / "data", root / "videos"):
        if not path.exists():
            raise FileNotFoundError(f"Missing required LeRobot v3 path: {path}")

    info = load_json(info_path)
    if info.get("codebase_version") != "v3.0":
        raise ValueError(f"Expected LeRobot v3.0, got {info.get('codebase_version')!r}")

    features = info.get("features", {})
    for key, expected_shape in REQUIRED_FEATURES.items():
        if key not in features:
            raise KeyError(f"Dataset is missing required feature {key!r}")
        actual_shape = features[key].get("shape")
        if actual_shape != expected_shape:
            raise ValueError(f"{key}: expected shape {expected_shape}, got {actual_shape}")

    stats = load_json(stats_path)
    for key in ("action", "observation.state"):
        if key not in stats:
            raise KeyError(f"meta/stats.json is missing {key!r}")
        missing = [name for name in REQUIRED_STATS if name not in stats[key]]
        if missing:
            raise KeyError(f"meta/stats.json {key!r} is missing statistics: {missing}")

    return info, stats


def install_starvla_metadata(root: Path, stats: dict, tasks_jsonl: Path | None):
    template = Path(__file__).parent / "train_files/modality_2026.json"
    shutil.copyfile(template, root / "meta/modality.json")

    cache = {
        "__format_version": 2,
        "__cache_config": {"mode": "abs"},
        "statistics": {
            key: {name: stats[key][name] for name in REQUIRED_STATS}
            for key in ("action", "observation.state")
        },
    }
    with (root / "meta/stats_gr00t.json").open("w") as f:
        json.dump(cache, f, indent=2)

    destination = root / "meta/tasks.jsonl"
    if tasks_jsonl is not None:
        source = tasks_jsonl.resolve()
        if source != destination.resolve():
            shutil.copyfile(source, destination)
    if not destination.exists():
        raise FileNotFoundError(
            "Natural-language tasks metadata is required. Download the official "
            "meta/tasks.jsonl and pass it with --tasks-jsonl."
        )

    task_count = 0
    with destination.open() as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            if not {"task_index", "task_name", "task"}.issubset(row):
                raise ValueError(f"Invalid task metadata row: {row}")
            task_count += 1
    if task_count == 0:
        raise ValueError(f"No task descriptions found in {destination}")
    return task_count


def sha256(path: Path):
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_manifest(root: Path):
    manifest = root / "transfer.sha256"
    files = sorted(path for path in root.rglob("*") if path.is_file() and path != manifest)
    with manifest.open("w") as f:
        for index, path in enumerate(files, start=1):
            relative = path.relative_to(root)
            f.write(f"{sha256(path)}  {relative.as_posix()}\n")
            print(f"hashed {index}/{len(files)}: {relative}", end="\r", flush=True)
    print(f"\nWrote {manifest}")


def prune_unused_videos(root: Path):
    videos_root = root / "videos"
    removed = []
    for path in videos_root.iterdir():
        if path.is_dir() and path.name not in TRAINING_VIDEO_KEYS:
            shutil.rmtree(path)
            removed.append(path.name)
    if removed:
        print("Removed unused video modalities: " + ", ".join(sorted(removed)))


def main():
    args = parse_args()
    root = args.dataset_root.resolve()
    info, stats = validate_dataset(root)
    task_count = install_starvla_metadata(root, stats, args.tasks_jsonl)
    if args.prune_unused_videos:
        prune_unused_videos(root)
    if not args.no_manifest:
        write_manifest(root)
    print(
        f"Ready for StarVLA: {root} "
        f"({info.get('total_episodes', 'unknown')} episodes, {task_count} task descriptions)"
    )


if __name__ == "__main__":
    main()
