import json
import subprocess
import sys
from pathlib import Path

import torch
from omegaconf import OmegaConf

from tools.package_surflo_checkpoint import package_checkpoint


def test_reference_bundle_omits_frozen_backbone_and_verifies(tmp_path):
    surflo_root = tmp_path / "surflo_install"
    (surflo_root / "surflo").mkdir(parents=True)
    (surflo_root / "surflo/model.py").write_text("version = 1\n")
    (surflo_root / "configs/model").mkdir(parents=True)
    (surflo_root / "configs/model/surflo.yaml").write_text("model: test\n")
    (surflo_root / "checkpoints").mkdir()
    (surflo_root / "checkpoints/surflo_v0.pt").write_bytes(b"release weights")
    run_dir = tmp_path / "run"
    (run_dir / "checkpoints").mkdir(parents=True)
    (run_dir / "dataset_statistics.json").write_text('{"new_embodiment": {}}\n')
    config = {
        "run_id": "test_run",
        "framework": {
            "name": "SurfloDiT",
            "state_conditioning": "parts",
            "action_model": {"num_condition_tokens": 138},
            "surflo": {
                "source_root": str(surflo_root),
                "model_config_path": str(surflo_root / "configs/model/surflo.yaml"),
                "checkpoint_path": str(surflo_root / "checkpoints/surflo_v0.pt"),
                "freeze_backbone": True,
                "append_per_view_camera_tokens": True,
            },
        },
    }
    OmegaConf.save(OmegaConf.create(config), run_dir / "config.yaml")
    checkpoint = run_dir / "checkpoints/steps_20_pytorch_model.pt"
    torch.save({"surflo_backbone.weight": torch.ones(2), "visual_proj.weight": torch.zeros(2)}, checkpoint)
    bundle = package_checkpoint(checkpoint, tmp_path / "bundle")
    manifest = json.loads((bundle / "manifest.json").read_text())
    weights = torch.load(bundle / "checkpoints/model.pt", weights_only=True)

    assert manifest["backbone"]["mode"] == "reference"
    assert list(weights) == ["visual_proj.weight"]
    assert not (bundle / "backbone").exists()
    result = subprocess.run([
        sys.executable, str(bundle / "code/tools/verify_surflo_bundle.py"),
        "--bundle", str(bundle), "--surflo-root", str(surflo_root), "--check-files-only",
    ], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    (surflo_root / "checkpoints/surflo_v0.pt").write_bytes(b"wrong release")
    result = subprocess.run([
        sys.executable, str(bundle / "code/tools/verify_surflo_bundle.py"),
        "--bundle", str(bundle), "--surflo-root", str(surflo_root), "--check-files-only",
    ], capture_output=True, text=True)
    assert result.returncode != 0
    assert "checksum mismatch" in result.stderr


def test_embedded_bundle_keeps_trained_backbone(tmp_path):
    surflo_root = tmp_path / "surflo_install"
    (surflo_root / "surflo").mkdir(parents=True)
    (surflo_root / "surflo/model.py").write_text("version = 1\n")
    (surflo_root / "configs/model").mkdir(parents=True)
    (surflo_root / "configs/model/surflo.yaml").write_text("model: test\n")
    run_dir = tmp_path / "run"
    (run_dir / "checkpoints").mkdir(parents=True)
    (run_dir / "dataset_statistics.json").write_text('{"new_embodiment": {}}\n')
    OmegaConf.save(OmegaConf.create({
        "run_id": "trained_run",
        "framework": {
            "name": "SurfloDiT",
            "action_model": {"num_condition_tokens": 129},
            "surflo": {
                "source_root": str(surflo_root),
                "model_config_path": str(surflo_root / "configs/model/surflo.yaml"),
                "checkpoint_path": str(surflo_root / "checkpoints/unneeded.pt"),
                "freeze_backbone": False,
            },
        },
    }), run_dir / "config.yaml")
    checkpoint = run_dir / "checkpoints/steps_20_pytorch_model.pt"
    torch.save({"surflo_backbone.weight": torch.ones(2), "visual_proj.weight": torch.zeros(2)}, checkpoint)
    bundle = package_checkpoint(checkpoint, tmp_path / "bundle")
    manifest = json.loads((bundle / "manifest.json").read_text())
    weights = torch.load(bundle / "checkpoints/model.pt", weights_only=True)

    assert manifest["backbone"]["mode"] == "embedded"
    assert "surflo_backbone.weight" in weights
    assert (bundle / "backbone/surflo.yaml").is_file()
    result = subprocess.run([
        sys.executable, str(bundle / "code/tools/verify_surflo_bundle.py"),
        "--bundle", str(bundle), "--surflo-root", str(surflo_root), "--check-files-only",
    ], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
