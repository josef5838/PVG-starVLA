"""Register matched A/C 10K rollouts on three test and five training scenes."""

import argparse
import datetime
import io
import json
from pathlib import Path

import numpy as np
import pandas as pd
from omegaconf import OmegaConf
from PIL import Image

from prepare_omega_joint24_gaas import digest

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-study", type=Path, required=True)
    parser.add_argument("--highway-study", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--result-root", type=Path, required=True)
    args = parser.parse_args()
    registry_path = ROOT / "docs/experiment_records/20261012/evidence/omega_camera_stream_training/rotate_qrcode_20261010_135404/manifest.json"
    registry = json.loads(registry_path.read_text())
    scenes = [dict(row) for split, count in (("test", 3), ("train", 5))
              for row in sorted((r for r in registry["rollout_scenes"] if r["split"] == split),
                                key=lambda r: r["episode"])[:count]]
    assert len(scenes) == 8 and len({r["episode"] for r in scenes}) == 8
    runs, checksums = [], {}
    statistics = None
    for variant, study in (("A", args.baseline_study), ("C", args.highway_study)):
        study = study.resolve()
        training = json.loads((study / "manifest.json").read_text())
        output = Path(training["output_dir"])
        checkpoint = output / "checkpoints/steps_10000_pytorch_model.pt"
        assert checkpoint.is_file()
        cfg_path = output / "config.full.yaml"
        cfg = OmegaConf.load(cfg_path)
        assert cfg.framework.name == "OmegaCameraStreamDiT"
        assert cfg.framework.streams.encoding == "joint24" and cfg.framework.streams.conditioning == "direct"
        assert cfg.framework.omega.train_last_blocks == 0
        assert bool(cfg.framework.get("spatial_highway", {}).get("enabled", False)) == (variant == "C")
        norm = json.loads((output / "dataset_statistics.json").read_text())
        if statistics is not None:
            assert norm == statistics
        statistics = norm
        for scene in scenes:
            assert scene["episode"] in training["split_ids"][scene["split"]]
            assert scene["episode"] in registry[scene["split"]+"_episode_ids"]
            assert scene["observation_matches_source_t"] and scene["action_matches_source_t_plus_1"]
        runs.append(dict(variant=variant, training_study=str(study), checkpoint=str(checkpoint),
                         checkpoint_step=10000, checkpoint_sha256=digest(checkpoint),
                         config=str(cfg_path), config_sha256=digest(cfg_path),
                         normalization=str(output / "dataset_statistics.json"),
                         normalization_sha256=digest(output / "dataset_statistics.json")))
        if variant == "A":
            data_root = Path(cfg.datasets.vla_data.data_root_dir) / "rotate_qrcode"
            dataset_hashes = training["dataset_checksums"]
    args.out = args.out.resolve()
    args.out.mkdir(parents=True, exist_ok=False)
    args.result_root = args.result_root.resolve()
    args.result_root.mkdir(parents=True, exist_ok=False)
    payload = {}
    image_keys = ["observation.images.cam_high", "observation.images.cam_left_wrist", "observation.images.cam_right_wrist"]
    for scene in scenes:
        relative = f"data/chunk-000/episode_{scene['episode']:06d}.parquet"
        path = data_root / relative
        checksum = digest(path)
        assert checksum == dataset_hashes["data/rotate_qrcode/"+relative]
        checksums[str(path)] = checksum
        frame = pd.read_parquet(path, columns=["observation.state"]+image_keys)
        assert len(frame) == scene["length"]
        row = frame.iloc[0]
        prefix = f"episode_{scene['episode']}"
        payload[prefix+"_state"] = np.asarray(row["observation.state"], dtype=np.float32)
        images = [np.asarray(Image.open(io.BytesIO(row[key]["bytes"])).convert("RGB")) for key in image_keys]
        assert all(image.shape == (480, 640, 3) for image in images)
        payload[prefix+"_images"] = np.stack(images)
    reference = args.out / "scene_references.npz"
    np.savez_compressed(reference, **payload)
    manifest = dict(status="prepared", task="rotate_qrcode", checkpoint_step=10000, runs=runs,
                    result_root=str(args.result_root), rollout_scenes=scenes,
                    scene_reference_data=str(reference), scene_reference_sha256=digest(reference),
                    source=registry["source"], source_registry=str(registry_path),
                    source_registry_sha256=digest(registry_path), selected_dataset_checksums=checksums,
                    native_camera_contract=True, initial_image_validation=registry["initial_image_validation"],
                    camera_order=registry["camera_order"], diffusion_seed=4000, num_ddim_steps=20,
                    execute_horizon=8, history_length=8, history_stride=1, image_size=[640,480],
                    selection_policy="First three test and first five registered train scenes by episode ID, before policy evaluation",
                    total_policy_rollouts=16, expert_controls=False,
                    simulator_root="/home/n2506021c/RoboTwin_PVG",
                    simulator_python="/home/n2506021c/miniconda3/envs/robotwin-gaas/bin/python",
                    policy_python="/home/n2506021c/miniconda3/envs/surflo-dit-cu124/bin/python",
                    runtime_note="GAAS simulator differs from source workstation; verify every initial state, object model/arm/pose and grayscale image against registry/local data before rollout",
                    created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
    print(json.dumps(dict(manifest=str(args.out/"manifest.json"), scenes=[{k:r[k] for k in ("split","episode","seed")} for r in scenes]),indent=2))


if __name__ == "__main__":
    main()
