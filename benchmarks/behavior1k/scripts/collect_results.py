#!/usr/bin/env python3
"""Collect BEHAVIOR rollout JSON files into a portable per-rollout CSV."""

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any


METRIC_NAMES = ("q_score", "time", "normalized_agent_distance", "agent_distance", "inference_ms")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", required=True, type=Path)
    parser.add_argument("--method", required=True)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def find_metric(value: Any, name: str) -> Any:
    if isinstance(value, dict):
        if name in value and isinstance(value[name], (int, float)):
            return value[name]
        for child in value.values():
            found = find_metric(child, name)
            if found is not None:
                return found
    elif isinstance(value, list):
        for child in value:
            found = find_metric(child, name)
            if found is not None:
                return found
    return None


def main() -> None:
    args = parse_args()
    source = args.input_root.resolve()
    rows = []
    for path in sorted(source.rglob("*.json")):
        if path.name == "provenance.json":
            continue
        try:
            with path.open() as stream:
                payload = json.load(stream)
        except json.JSONDecodeError:
            continue
        row = {"method": args.method, "rollout_json": str(path.relative_to(source))}
        row.update({name: find_metric(payload, name) for name in METRIC_NAMES})
        rows.append(row)

    if not rows:
        raise SystemExit(f"No rollout JSON files found under {source}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=("method", "rollout_json", *METRIC_NAMES))
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} rollouts to {args.output}")
    for metric in METRIC_NAMES:
        values = [row[metric] for row in rows if isinstance(row[metric], (int, float)) and math.isfinite(row[metric])]
        if values:
            print(f"mean {metric}: {sum(values) / len(values):.6f} ({len(values)} rollouts)")


if __name__ == "__main__":
    main()
