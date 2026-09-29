"""Validation and local Surflo path resolution for portable policy bundles."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def surflo_source_digest(root: Path) -> str:
    source = root / "surflo"
    files = sorted(source.rglob("*.py"))
    if not files:
        raise FileNotFoundError(f"No Surflo Python sources under {source}")
    digest = hashlib.sha256()
    for path in files:
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(bytes.fromhex(sha256_file(path)))
    return digest.hexdigest()


def _bundle_file(bundle: Path, relative: str) -> Path:
    path = Path(relative)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"Invalid bundle-relative path: {relative}")
    return bundle / path


def resolve_surflo_bundle(bundle_dir: str | Path, surflo_root: str | Path):
    bundle = Path(bundle_dir).expanduser().resolve()
    root = Path(surflo_root).expanduser().resolve()
    manifest = json.loads((bundle / "manifest.json").read_text())
    if manifest.get("format_version") != 1 or manifest.get("framework") != "SurfloDiT":
        raise ValueError("Unsupported Surflo bundle manifest")
    code_root = bundle / "code"
    if Path(__file__).resolve().parents[3] != code_root:
        raise RuntimeError(f"Run with PYTHONPATH={code_root} to use the bundled architecture code")
    for relative, expected in manifest["files"].items():
        path = _bundle_file(bundle, relative)
        if not path.is_file() or sha256_file(path) != expected:
            raise ValueError(f"Bundle file missing or changed: {relative}")
    backbone = manifest["backbone"]
    if surflo_source_digest(root) != backbone["source_sha256"]:
        raise ValueError("RTX Surflo source differs from the training source; install the matching revision")
    mode = backbone["mode"]
    if mode == "reference":
        model_config = root / backbone["model_config_relative_path"]
        release = root / backbone["release_checkpoint_relative_path"]
        for path, expected in ((model_config, backbone["model_config_sha256"]),
                               (release, backbone["release_checkpoint_sha256"])):
            if not path.is_file() or sha256_file(path) != expected:
                raise ValueError(f"RTX Surflo file missing or checksum mismatch: {path}")
        checkpoint_override = f"framework.surflo.checkpoint_path={release}"
    elif mode == "embedded":
        model_config = _bundle_file(bundle, backbone["model_config_path"])
        checkpoint_override = "framework.surflo.checkpoint_path=null"
    else:
        raise ValueError(f"Unsupported backbone mode: {mode}")
    overrides = [
        f"framework.surflo.source_root={root}",
        f"framework.surflo.model_config_path={model_config}",
        checkpoint_override,
    ]
    return _bundle_file(bundle, manifest["checkpoint_path"]), overrides, manifest
