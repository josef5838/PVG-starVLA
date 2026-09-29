#!/usr/bin/env python3
"""Build a portable SurfloDiT evaluation bundle from a training checkpoint."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import torch
from omegaconf import OmegaConf

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from starVLA.model.framework.surflo_bundle import sha256_file, surflo_source_digest


def _git_revision(root: Path) -> str | None:
    result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True)
    return result.stdout.strip() if result.returncode == 0 else None


def _copy_code_snapshot(destination: Path) -> None:
    ignored = shutil.ignore_patterns("__pycache__", "*.pyc", ".pytest_cache")
    for directory in ("starVLA", "deployment"):
        shutil.copytree(REPO_ROOT / directory, destination / directory, ignore=ignored)
    server = Path("examples/simBenchmarks/Behavior/serve_vggt_omega_2026.py")
    target = destination / server
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(REPO_ROOT / server, target)
    verifier = Path("tools/verify_surflo_bundle.py")
    target = destination / verifier
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(REPO_ROOT / verifier, target)


def package_checkpoint(checkpoint: Path, output: Path) -> Path:
    checkpoint = checkpoint.expanduser().resolve()
    output = output.expanduser().resolve()
    run_dir = checkpoint.parents[1]
    if not checkpoint.is_file() or checkpoint.parent.parent != run_dir:
        raise FileNotFoundError(checkpoint)
    if output.exists():
        raise FileExistsError(f"Bundle already exists: {output}")
    config_path = run_dir / "config.yaml"
    stats_path = run_dir / "dataset_statistics.json"
    cfg = OmegaConf.load(config_path)
    if cfg.framework.name != "SurfloDiT":
        raise ValueError("Only SurfloDiT checkpoints can be packaged")
    if not stats_path.is_file():
        raise FileNotFoundError(stats_path)
    surflo = cfg.framework.surflo
    root = Path(surflo.source_root).expanduser().resolve()
    model_config = Path(surflo.model_config_path).expanduser().resolve()
    mode = "reference" if bool(surflo.freeze_backbone) else "embedded"
    weights = torch.load(checkpoint, map_location="cpu", mmap=True, weights_only=True)
    if not isinstance(weights, dict) or not any(key.startswith("surflo_backbone.") for key in weights):
        raise ValueError("Checkpoint does not contain Surflo backbone weights")

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    try:
        (temporary / "checkpoints").mkdir()
        _copy_code_snapshot(temporary / "code")
        shutil.copy2(stats_path, temporary / "dataset_statistics.json")
        portable_cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
        portable_cfg.framework.surflo.weights_mode = mode
        portable_cfg.framework.surflo.source_root = "SET_VIA_BUNDLE_LOADER"
        portable_cfg.framework.surflo.model_config_path = "SET_VIA_BUNDLE_LOADER"
        portable_cfg.framework.surflo.checkpoint_path = "SET_VIA_BUNDLE_LOADER" if mode == "reference" else None
        OmegaConf.save(portable_cfg, temporary / "config.yaml")

        backbone = {
            "mode": mode,
            "source_sha256": surflo_source_digest(root),
            "source_git_revision": _git_revision(root),
            "model_config_sha256": sha256_file(model_config),
        }
        if mode == "reference":
            release = Path(surflo.checkpoint_path).expanduser().resolve()
            backbone["model_config_relative_path"] = model_config.relative_to(root).as_posix()
            backbone["release_checkpoint_relative_path"] = release.relative_to(root).as_posix()
            backbone["release_checkpoint_sha256"] = sha256_file(release)
            policy_weights = {key: value for key, value in weights.items()
                              if not key.startswith("surflo_backbone.")}
        else:
            (temporary / "backbone").mkdir()
            shutil.copy2(model_config, temporary / "backbone/surflo.yaml")
            backbone["model_config_path"] = "backbone/surflo.yaml"
            policy_weights = weights
        if not policy_weights:
            raise ValueError("No policy weights remain after packaging")
        torch.save(policy_weights, temporary / "checkpoints/model.pt")
        del policy_weights, weights

        file_hashes = {
            path.relative_to(temporary).as_posix(): sha256_file(path)
            for path in sorted(temporary.rglob("*")) if path.is_file()
        }
        manifest = {
            "format_version": 1,
            "framework": "SurfloDiT",
            "run_id": str(cfg.run_id),
            "checkpoint_path": "checkpoints/model.pt",
            "architecture": {
                "state_conditioning": str(getattr(cfg.framework, "state_conditioning", "add")),
                "num_condition_tokens": int(cfg.framework.action_model.num_condition_tokens),
                "per_view_camera_tokens": bool(getattr(surflo, "append_per_view_camera_tokens", False)),
                "starvla_git_revision": _git_revision(REPO_ROOT),
            },
            "backbone": backbone,
            "files": file_hashes,
        }
        (temporary / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        temporary.rename(output)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    print(package_checkpoint(args.checkpoint, args.output))


if __name__ == "__main__":
    main()
