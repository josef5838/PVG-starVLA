#!/usr/bin/env python3
"""Validate the BEHAVIOR-1K data contract without loading a GPU model."""

import argparse
import json
import sys
from pathlib import Path


REQUIRED_FEATURES = {
    "action": [23],
    "observation.state": [61],
    "observation.rgb.zed_link_camera_0": [720, 720, 3],
    "observation.rgb.left_realsense_link_camera_0": [480, 480, 3],
    "observation.rgb.right_realsense_link_camera_0": [480, 480, 3],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--task", required=True)
    parser.add_argument(
        "--protocol",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "protocol.json",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict:
    with path.open() as stream:
        return json.load(stream)


def main() -> int:
    args = parse_args()
    root = args.dataset.resolve()
    required_paths = [root / "data", root / "videos", root / "meta" / "info.json", root / "meta" / "tasks.jsonl"]
    missing = [str(path) for path in required_paths if not path.exists()]
    if missing:
        print("Dataset is incomplete:", *missing, sep="\n  ", file=sys.stderr)
        return 2

    protocol = load_json(args.protocol)
    if protocol["behavior_version"] != "v3.9.2":
        print("Protocol must be pinned to BEHAVIOR-1K v3.9.2", file=sys.stderr)
        return 2

    info = load_json(root / "meta" / "info.json")
    if info.get("codebase_version") != "v3.0":
        print(f"Expected LeRobot v3.0, found {info.get('codebase_version')!r}", file=sys.stderr)
        return 2
    features = info.get("features", {})
    for key, expected in REQUIRED_FEATURES.items():
        actual = features.get(key, {}).get("shape")
        if actual != expected:
            print(f"{key}: expected {expected}, found {actual}", file=sys.stderr)
            return 2

    with (root / "meta" / "tasks.jsonl").open() as stream:
        tasks = [json.loads(line) for line in stream if line.strip()]
    matched = [row for row in tasks if row.get("task_name") == args.task]
    if len(matched) != 1 or not matched[0].get("task"):
        print(f"Task {args.task!r} is absent or has no language instruction", file=sys.stderr)
        return 2

    manifest = root / "transfer.sha256"
    print(f"Dataset validated: {root}")
    print(f"Task: {matched[0]['task']}")
    print(f"Transfer manifest: {'present' if manifest.is_file() else 'absent (create before machine transfer)'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
