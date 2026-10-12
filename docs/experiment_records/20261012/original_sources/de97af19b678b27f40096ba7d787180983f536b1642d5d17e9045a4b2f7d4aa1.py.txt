#!/usr/bin/env python3
"""Download one BEHAVIOR-1K 2026 task without fetching the 3 TB dataset."""

import argparse
import json
from pathlib import Path

import pandas as pd
from huggingface_hub import HfApi, hf_hub_download


REPO_ID = "behavior-1k/2026-challenge-demos"
RGB_VIDEO_KEYS = (
    "observation.rgb.zed_link_camera_0",
    "observation.rgb.left_realsense_link_camera_0",
    "observation.rgb.right_realsense_link_camera_0",
)
METADATA_FILES = (
    "meta/info.json",
    "meta/stats.json",
    "meta/tasks.jsonl",
    "meta/tasks.parquet",
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--task", default="turning_on_radio", help="Task name or integer task index")
    parser.add_argument("--repo-id", default=REPO_ID)
    parser.add_argument(
        "--revision",
        default="v3.0",
        help="Dataset revision; v3.0 currently resolves to the corrected v3.9.2-compatible dataset",
    )
    parser.add_argument("--dry-run", action="store_true", help="Fetch metadata and list the task files only")
    return parser.parse_args()


def download(repo_id: str, revision: str, filename: str, output_root: Path):
    return Path(
        hf_hub_download(
            repo_id=repo_id,
            repo_type="dataset",
            revision=revision,
            filename=filename,
            local_dir=output_root,
        )
    )


def resolve_task(tasks_jsonl: Path, requested: str):
    rows = [json.loads(line) for line in tasks_jsonl.read_text().splitlines() if line.strip()]
    try:
        task_index = int(requested)
    except ValueError:
        task_index = None
    normalized = requested.strip().lower().replace(" ", "_")
    matches = [
        row
        for row in rows
        if (task_index is not None and row["task_index"] == task_index)
        or str(row.get("task_name", "")).strip().lower().replace(" ", "_") == normalized
    ]
    if len(matches) != 1:
        raise ValueError(f"Expected one task matching {requested!r}, found {len(matches)}")
    return matches[0]


def load_task_episodes(root: Path, task_index: int):
    files = sorted((root / "meta/episodes").glob("**/*.parquet"))
    if not files:
        raise FileNotFoundError("No episode metadata was downloaded")
    selected = []
    for path in files:
        task_rows = pd.read_parquet(path).query("task_index == @task_index")
        if not task_rows.empty:
            selected.append(task_rows)
    if not selected:
        raise ValueError(f"No episodes found for task index {task_index}")
    return pd.concat(selected, ignore_index=True).sort_values("episode_index")


def referenced_files(info: dict, episodes: pd.DataFrame):
    paths = set()
    for _, episode in episodes.iterrows():
        paths.add(
            info["data_path"].format(
                chunk_index=int(episode["data/chunk_index"]),
                file_index=int(episode["data/file_index"]),
            )
        )
        for video_key in RGB_VIDEO_KEYS:
            paths.add(
                info["video_path"].format(
                    video_key=video_key,
                    chunk_index=int(episode[f"videos/{video_key}/chunk_index"]),
                    file_index=int(episode[f"videos/{video_key}/file_index"]),
                )
            )
    return sorted(paths)


def main():
    args = parse_args()
    root = args.output_root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    api = HfApi()

    for filename in METADATA_FILES:
        download(args.repo_id, args.revision, filename, root)
    episode_files = sorted(
        path
        for path in api.list_repo_files(args.repo_id, repo_type="dataset", revision=args.revision)
        if path.startswith("meta/episodes/") and path.endswith(".parquet")
    )
    if not episode_files:
        raise RuntimeError("The repository did not expose any episode metadata files")
    for index, filename in enumerate(episode_files, start=1):
        print(f"metadata {index}/{len(episode_files)}: {filename}")
        download(args.repo_id, args.revision, filename, root)

    task = resolve_task(root / "meta/tasks.jsonl", args.task)
    episodes = load_task_episodes(root, int(task["task_index"]))
    files = referenced_files(json.loads((root / "meta/info.json").read_text()), episodes)
    parquet_files = [path for path in files if path.endswith(".parquet")]
    video_files = [path for path in files if path.endswith(".mp4")]
    print(
        f"Task {task['task_index']} ({task['task_name']}): {len(episodes)} episodes, "
        f"{len(parquet_files)} parquet files, and {len(video_files)} RGB videos."
    )
    if args.dry_run:
        return

    for index, filename in enumerate(files, start=1):
        print(f"file {index}/{len(files)}: {filename}")
        download(args.repo_id, args.revision, filename, root)

    (root / "meta/starvla_sparse_download.json").write_text(
        json.dumps(
            {
                "repo_id": args.repo_id,
                "revision": args.revision,
                "task_index": task["task_index"],
                "task_name": task["task_name"],
                "episodes": episodes["episode_index"].astype(int).tolist(),
                "video_keys": RGB_VIDEO_KEYS,
            },
            indent=2,
        )
        + "\n"
    )
    print(f"Downloaded only {task['task_name']} into {root}")


if __name__ == "__main__":
    main()
