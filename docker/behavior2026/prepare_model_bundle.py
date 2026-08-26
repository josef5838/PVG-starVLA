#!/usr/bin/env python3
"""Create a self-contained, read-only model bundle for Docker inference."""

import argparse
import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path

from huggingface_hub import snapshot_download
from omegaconf import OmegaConf


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument(
        "--tasks-jsonl",
        default=Path("examples/simBenchmarks/Behavior/tasks.jsonl"),
        type=Path,
    )
    parser.add_argument("--clip-source", type=Path)
    parser.add_argument(
        "--vggt-source",
        default=Path("../vggt-omega-action-pred"),
        type=Path,
    )
    parser.add_argument("--file-mode", choices=("hardlink", "copy"), default="hardlink")
    parser.add_argument(
        "--dataset-revision",
        default="4f50b44796641a4d526a19d9aeadc8aa51e2f2c2",
    )
    return parser.parse_args()


def git_state(path: Path):
    def run(*args):
        return subprocess.run(
            ["git", "-C", str(path), *args],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()

    return {
        "path": str(path),
        "commit": run("rev-parse", "HEAD"),
        "dirty": bool(run("status", "--porcelain")),
    }


def materialize(source: Path, destination: Path, mode: str):
    destination.parent.mkdir(parents=True, exist_ok=True)
    if mode == "hardlink":
        os.link(source.resolve(), destination)
    else:
        shutil.copy2(source, destination, follow_symlinks=True)


def materialize_tree(source: Path, destination: Path, mode: str):
    for path in sorted(source.rglob("*")):
        if path.is_file():
            materialize(path, destination / path.relative_to(source), mode)


def sha256(path: Path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_manifest(root: Path):
    manifest = root / "SHA256SUMS"
    files = sorted(path for path in root.rglob("*") if path.is_file() and path != manifest)
    with manifest.open("w") as stream:
        for path in files:
            stream.write(f"{sha256(path)}  {path.relative_to(root).as_posix()}\n")


def main():
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[2]
    checkpoint = args.checkpoint.resolve()
    output_root = args.output_root.resolve()
    tasks_jsonl = args.tasks_jsonl.resolve()
    vggt_source = args.vggt_source.resolve()

    if output_root.exists():
        raise FileExistsError(f"Refusing to overwrite existing bundle: {output_root}")
    if checkpoint.suffix not in {".pt", ".safetensors"}:
        raise ValueError(f"Unsupported checkpoint format: {checkpoint}")
    run_dir = checkpoint.parents[1]
    config_path = run_dir / "config.yaml"
    statistics_path = run_dir / "dataset_statistics.json"
    for path in (checkpoint, config_path, statistics_path, tasks_jsonl):
        if not path.is_file():
            raise FileNotFoundError(path)

    clip_source = args.clip_source
    if clip_source is None:
        clip_source = Path(
            snapshot_download("openai/clip-vit-base-patch32", local_files_only=True)
        )
    clip_source = clip_source.resolve()

    cfg = OmegaConf.load(config_path)
    cfg.framework.vggt_omega.checkpoint_path = None
    cfg.framework.clip_text.model_name = "${oc.env:MODEL_ROOT,/models}/clip"
    cfg.framework.clip_text.local_files_only = True
    if "datasets" in cfg and "vla_data" in cfg.datasets:
        cfg.datasets.vla_data.data_root_dir = "/not-used-for-inference"

    output_root.mkdir(parents=True)
    bundle_run = output_root / "run"
    bundle_run.mkdir()
    OmegaConf.save(cfg, bundle_run / "config.yaml", resolve=False)
    shutil.copy2(statistics_path, bundle_run / "dataset_statistics.json")
    materialize(
        checkpoint,
        bundle_run / "final_model" / checkpoint.name,
        args.file_mode,
    )
    shutil.copy2(tasks_jsonl, output_root / "tasks.jsonl")
    materialize_tree(clip_source, output_root / "clip", args.file_mode)

    provenance = {
        "format_version": 1,
        "checkpoint": checkpoint.name,
        "checkpoint_source": str(checkpoint),
        "starvla": git_state(repo_root),
        "vggt_omega": git_state(vggt_source),
        "dataset": "behavior-1k/2026-challenge-demos",
        "dataset_revision": args.dataset_revision,
        "runtime_model_root": "/models",
        "file_mode": args.file_mode,
    }
    with (output_root / "provenance.json").open("w") as stream:
        json.dump(provenance, stream, indent=2)
        stream.write("\n")

    write_manifest(output_root)
    print(f"Created model bundle: {output_root}")
    print(f"Checkpoint: run/final_model/{checkpoint.name}")
    print("Verify with: (cd MODEL_BUNDLE && sha256sum -c SHA256SUMS)")


if __name__ == "__main__":
    main()
