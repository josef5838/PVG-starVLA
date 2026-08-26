#!/usr/bin/env python3
"""Write a minimal immutable provenance record before a benchmark run."""

import argparse
import hashlib
import json
import subprocess
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--method", required=True)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--runtime", required=True, help="e.g. H100 training or RTX 4090 training")
    parser.add_argument("--external-commit", action="append", default=[], metavar="NAME=SHA")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_head(root: Path) -> str:
    return subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()


def main() -> None:
    args = parse_args()
    dataset = args.dataset.resolve()
    manifest = dataset / "transfer.sha256"
    if not manifest.is_file():
        raise SystemExit(f"Dataset manifest is required: {manifest}")
    repo_root = Path(__file__).resolve().parents[3]
    external = {}
    for item in args.external_commit:
        name, separator, commit = item.partition("=")
        if not separator or not name or not commit:
            raise SystemExit(f"Invalid --external-commit {item!r}; use NAME=SHA")
        external[name] = commit
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format_version": 1,
        "method": args.method,
        "seed": args.seed,
        "runtime": args.runtime,
        "starvla_commit": git_head(repo_root),
        "dataset_root": str(dataset),
        "dataset_manifest": str(manifest),
        "dataset_manifest_sha256": sha256(manifest),
        "external_commits": external,
    }
    with output.open("w") as stream:
        json.dump(payload, stream, indent=2)
        stream.write("\n")
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
