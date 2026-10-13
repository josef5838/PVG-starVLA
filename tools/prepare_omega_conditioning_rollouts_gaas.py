"""Register four conditioning interventions on the completed A 20K checkpoint."""

import argparse
import datetime
import hashlib
import json
from pathlib import Path

import yaml


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-study", type=Path, required=True)
    parser.add_argument("--baseline-study", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--result-root", type=Path, required=True)
    args = parser.parse_args()
    reference = args.reference_study.resolve() / "manifest.json"
    manifest = json.loads(reference.read_text())
    training = json.loads((args.baseline_study / "manifest.json").read_text())
    output = Path(training["output_dir"])
    status = json.loads((output / "stream_training_status.json").read_text())
    assert status["status"] == "completed" and status["completed_updates"] == 20000
    assert status["consumed_observations"] == 160000
    checkpoint = output / "final_model/pytorch_model.pt"
    assert checkpoint.is_file()
    config = output / "config.full.yaml"
    cfg = yaml.safe_load(config.read_text())["framework"]
    assert cfg["name"] == "OmegaCameraStreamDiT"
    assert cfg["streams"]["encoding"] == "joint24" and cfg["streams"]["conditioning"] == "direct"
    assert not cfg.get("spatial_highway", {}).get("enabled", False)
    assert cfg["state_conditioning"] == "full" and cfg["visual_conditioning"]
    scenes = manifest["rollout_scenes"]
    assert len(scenes) == 8 and sum(r["split"] == "test" for r in scenes) == 3
    for scene in scenes:
        assert scene["episode"] in training["split_ids"][scene["split"]]
    assert digest(manifest["scene_reference_data"]) == manifest["scene_reference_sha256"]
    checkpoint_hash = digest(checkpoint)
    runs = []
    for index, (variant, state, vision) in enumerate((("full", "full", True), ("vision_only", "none", True),
                                                     ("state_only", "full", False), ("task_only", "none", False))):
        runs.append(dict(variant=variant, model="A", training_study=str(args.baseline_study.resolve()),
                         checkpoint=str(checkpoint), checkpoint_step=20000,
                         checkpoint_sha256=checkpoint_hash, config=str(config), config_sha256=digest(config),
                         normalization=str(output / "dataset_statistics.json"),
                         normalization_sha256=digest(output / "dataset_statistics.json"),
                         state_conditioning=state, visual_conditioning=vision, port=5790+index))
    for key in ("pbs_jobs", "source_commit", "source_snapshot", "source_hashes", "runs"):
        manifest.pop(key, None)
    manifest.update(status="prepared", experiment="A20K conditioning interventions",
                    checkpoint_step=20000, runs=runs, total_policy_rollouts=32,
                    result_root=str(args.result_root.resolve()), reference_study=str(reference),
                    reference_manifest_sha256=digest(reference),
                    training_completion=status, created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                    preflight_tool="preflight_omega_conditioning_rollouts.py",
                    summary_tool="summarize_omega_conditioning_rollouts.py", record_conditioning_geometry=True,
                    intervention_note="Inference-time route masking; no retraining. State masks all four projected state tokens and embeddings while retaining task. Vision masking bypasses Omega and uses constant zero visual memory. Images still contain robot configuration when vision is enabled.",
                    interpretation="Assess explicit-route dependence with paired progress/geometry/action diagnostics. Masking is out of training distribution; zero successes or state sensitivity alone do not prove a harmful shortcut.")
    args.out.resolve().mkdir(parents=True, exist_ok=False)
    args.result_root.resolve().mkdir(parents=True, exist_ok=False)
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(dict(study=str(args.out.resolve()), checkpoint=str(checkpoint),
                         checkpoint_sha256=checkpoint_hash, variants=[r["variant"] for r in runs],
                         total_rollouts=32), indent=2))


if __name__ == "__main__":
    main()
