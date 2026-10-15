"""Register C state-dropout 10K rollouts against the existing A/C scene suite."""

import argparse
import datetime
import json
from pathlib import Path

import yaml

from prepare_omega_conditioning_rollouts_gaas import digest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-study", type=Path, required=True)
    parser.add_argument("--training-study", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--result-root", type=Path, required=True)
    args = parser.parse_args()
    reference = args.reference_study.resolve() / "manifest.json"
    manifest = json.loads(reference.read_text())
    training_path = args.training_study.resolve() / "manifest.json"
    training = json.loads(training_path.read_text())
    output = Path(training["output_dir"])
    status = json.loads((output / "stream_training_status.json").read_text())
    assert training["status"] == status["status"] == "completed"
    assert status["completed_updates"] == 10000 and status["consumed_observations"] == 80000
    config = output / "config.full.yaml"
    cfg = yaml.safe_load(config.read_text())
    control = next(run for run in manifest["runs"] if run["variant"] == "C")
    assert digest(control["config"]) == control["config_sha256"]
    reference_cfg = yaml.safe_load(Path(control["config"]).read_text())
    framework = dict(cfg["framework"])
    assert framework["state_dropout_prob"] == 0.2
    framework["state_dropout_prob"] = 0.0
    assert framework == reference_cfg["framework"], "Architecture or conditioning changed"
    assert cfg["framework"]["state_conditioning"] == "full"
    assert cfg["framework"]["visual_conditioning"]
    assert cfg["framework"]["spatial_highway"]["enabled"]
    for key in ("history_length", "execute_horizon", "num_ddim_steps", "diffusion_seed"):
        assert manifest[key] == dict(history_length=8, execute_horizon=8,
                                     num_ddim_steps=20, diffusion_seed=4000)[key]
    scenes = manifest["rollout_scenes"]
    assert len(scenes) == 8 and sum(s["split"] == "test" for s in scenes) == 3
    assert sum(s["split"] == "train" for s in scenes) == 5
    for scene in scenes:
        assert scene["episode"] in training["split_ids"][scene["split"]]
    assert digest(manifest["scene_reference_data"]) == manifest["scene_reference_sha256"]
    normalization = output / "dataset_statistics.json"
    assert digest(normalization) == control["normalization_sha256"]
    checkpoint = output / "checkpoints/steps_10000_pytorch_model.pt"
    run = dict(variant="C_dropout20", model="C", training_study=str(training_path.parent),
               checkpoint=str(checkpoint), checkpoint_step=10000, checkpoint_sha256=digest(checkpoint),
               config=str(config), config_sha256=digest(config), normalization=str(normalization),
               normalization_sha256=digest(normalization), state_conditioning="full",
               visual_conditioning=True, port=5794)
    for key in ("pbs_jobs", "source_commit", "source_snapshot", "source_hashes", "source_file_count",
                "cpu_validation", "pbs_script", "runs"):
        manifest.pop(key, None)
    manifest.update(status="prepared", experiment="C dropout20 matched 10K rollouts", runs=[run],
                    total_policy_rollouts=8, result_root=str(args.result_root.resolve()),
                    reference_study=str(reference), reference_manifest_sha256=digest(reference),
                    control_run=control, training_completion=status,
                    created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                    preflight_tool="preflight_omega_conditioning_rollouts.py",
                    record_conditioning_geometry=True,
                    evaluation_note="Same eight scenes, resets, cameras, history, DDIM seeds and execution horizon as prior A/C 10K. Full state and vision; training dropout disabled during inference. Use the 10K milestone, not best-validation selection.")
    args.out.resolve().mkdir(parents=True, exist_ok=False)
    args.result_root.resolve().mkdir(parents=True, exist_ok=False)
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(dict(study=str(args.out.resolve()), checkpoint=str(checkpoint),
                         checkpoint_sha256=run["checkpoint_sha256"], total_rollouts=8), indent=2))


if __name__ == "__main__":
    main()
