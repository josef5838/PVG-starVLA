#!/usr/bin/env python3
"""Build a task-specific BEHAVIOR-1K 2026 LeRobot v3 dataset."""

import argparse
import json
import os
import shutil
from pathlib import Path

import pandas as pd


RGB_VIDEO_KEYS = (
    "observation.rgb.zed_link_camera_0",
    "observation.rgb.left_realsense_link_camera_0",
    "observation.rgb.right_realsense_link_camera_0",
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--task", required=True, help="Task name or integer task index")
    parser.add_argument(
        "--file-mode",
        choices=("hardlink", "symlink", "copy"),
        default="hardlink",
        help="How to materialize large parquet and video shards",
    )
    return parser.parse_args()


def load_json(path: Path):
    with path.open() as f:
        return json.load(f)


def resolve_task(tasks_path: Path, task: str):
    rows = []
    with tasks_path.open() as f:
        rows = [json.loads(line) for line in f if line.strip()]

    try:
        requested_index = int(task)
    except ValueError:
        requested_index = None

    normalized = task.strip().lower().replace(" ", "_")
    matches = [
        row
        for row in rows
        if (requested_index is not None and row["task_index"] == requested_index)
        or str(row.get("task_name", "")).strip().lower().replace(" ", "_") == normalized
    ]
    if len(matches) != 1:
        raise ValueError(f"Expected one task matching {task!r}, found {len(matches)}")
    return matches[0]


def load_task_episodes(source_root: Path, task_index: int):
    episode_files = sorted((source_root / "meta/episodes").glob("**/*.parquet"))
    if not episode_files:
        raise FileNotFoundError(f"No episode metadata under {source_root / 'meta/episodes'}")

    selected = []
    for path in episode_files:
        frame = pd.read_parquet(path)
        task_rows = frame.loc[frame["task_index"] == task_index]
        if not task_rows.empty:
            selected.append(task_rows)
    if not selected:
        raise ValueError(f"No episodes found for task index {task_index}")
    return pd.concat(selected, ignore_index=True).sort_values("episode_index")


def referenced_files(source_root: Path, episodes: pd.DataFrame):
    info = load_json(source_root / "meta/info.json")
    paths = set()
    for _, episode in episodes.iterrows():
        paths.add(
            Path(
                info["data_path"].format(
                    chunk_index=int(episode["data/chunk_index"]),
                    file_index=int(episode["data/file_index"]),
                )
            )
        )
        for video_key in RGB_VIDEO_KEYS:
            paths.add(
                Path(
                    info["video_path"].format(
                        video_key=video_key,
                        chunk_index=int(episode[f"videos/{video_key}/chunk_index"]),
                        file_index=int(episode[f"videos/{video_key}/file_index"]),
                    )
                )
            )
    missing = sorted(path for path in paths if not (source_root / path).is_file())
    if missing:
        formatted = "\n".join(f"  {path}" for path in missing)
        raise FileNotFoundError(f"Source dataset is missing referenced files:\n{formatted}")
    return sorted(paths)


def materialize(source: Path, destination: Path, mode: str):
    destination.parent.mkdir(parents=True, exist_ok=True)
    if mode == "hardlink":
        os.link(source, destination)
    elif mode == "symlink":
        destination.symlink_to(os.path.relpath(source, destination.parent))
    else:
        shutil.copy2(source, destination)


def write_metadata(source_root: Path, output_root: Path, task: dict, episodes: pd.DataFrame):
    meta = output_root / "meta"
    (meta / "episodes/chunk-000").mkdir(parents=True)

    info = load_json(source_root / "meta/info.json")
    info.update(
        total_episodes=len(episodes),
        total_frames=int(episodes["length"].sum()),
        total_tasks=1,
        splits={"train": f"0:{len(episodes)}"},
    )
    with (meta / "info.json").open("w") as f:
        json.dump(info, f, indent=2)
        f.write("\n")

    shutil.copy2(source_root / "meta/stats.json", meta / "stats.json")
    episodes.to_parquet(meta / "episodes/chunk-000/file-000.parquet", index=False)

    with (meta / "tasks.jsonl").open("w") as f:
        f.write(json.dumps(task) + "\n")

    tasks = pd.read_parquet(source_root / "meta/tasks.parquet")
    tasks.loc[tasks["task_index"] == task["task_index"]].to_parquet(
        meta / "tasks.parquet", index=True
    )
    with (meta / "starvla_subset.json").open("w") as f:
        json.dump(
            {
                "source": str(source_root),
                "task_index": task["task_index"],
                "task_name": task["task_name"],
                "episode_indices": episodes["episode_index"].astype(int).tolist(),
                "statistics": "Official full-dataset statistics from meta/stats.json",
            },
            f,
            indent=2,
        )
        f.write("\n")


def main():
    args = parse_args()
    source_root = args.source_root.resolve()
    output_root = args.output_root.resolve()
    if output_root.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {output_root}")

    task = resolve_task(source_root / "meta/tasks.jsonl", args.task)
    episodes = load_task_episodes(source_root, int(task["task_index"]))
    paths = referenced_files(source_root, episodes)

    output_root.mkdir(parents=True)
    write_metadata(source_root, output_root, task, episodes)
    for path in paths:
        materialize(source_root / path, output_root / path, args.file_mode)

    print(
        f"Created {output_root}: {len(episodes)} episodes, "
        f"{int(episodes['length'].sum())} frames, {len(paths)} data/video files"
    )


if __name__ == "__main__":
    main()
