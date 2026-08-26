#!/usr/bin/env python3
"""Write a deterministic SHA-256 manifest for a prepared LeRobot dataset."""

import argparse
import hashlib
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    root = args.dataset.resolve()
    output = root / "transfer.sha256"
    if output.exists() and not args.overwrite:
        raise SystemExit(f"Refusing to replace existing manifest: {output}; pass --overwrite to regenerate it")
    files = sorted(path for path in root.rglob("*") if path.is_file() and path != output)
    if not files:
        raise SystemExit(f"No files found under {root}")
    with output.open("w") as stream:
        for index, path in enumerate(files, start=1):
            stream.write(f"{sha256(path)}  {path.relative_to(root).as_posix()}\n")
            print(f"hashed {index}/{len(files)}: {path.relative_to(root)}", end="\r", flush=True)
    print(f"\nWrote {output}")


if __name__ == "__main__":
    main()
