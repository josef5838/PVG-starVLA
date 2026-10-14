"""Prepare a fresh C state-dropout control with a matched first-10K LR schedule."""

import argparse
import datetime
import hashlib
import json
from pathlib import Path

import yaml


def digest(file):
    value = hashlib.sha256()
    with Path(file).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def flatten(value, prefix=""):
    if isinstance(value, dict):
        result = {}
        for key, child in value.items():
            result.update(flatten(child, f"{prefix}.{key}" if prefix else key))
        return result
    return {prefix: value}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-study", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    args = parser.parse_args()
    args.out = args.out.resolve()
    args.checkpoint_root = args.checkpoint_root.resolve()
    reference = args.reference_study.resolve()
    original = reference / "production.yaml"
    prior = json.loads((reference / "manifest.json").read_text())
    assert prior["status"] == "completed" and prior["experiment"] == "C"
    preflight = json.loads((reference / "preflight.json").read_text())
    assert preflight["status"] == "passed" and preflight["trainable_parameters"] == 358505492
    cfg = yaml.safe_load(original.read_text())
    before = flatten(cfg)
    assert cfg["seed"] == 42
    assert cfg["framework"]["spatial_highway"] == dict(enabled=True, selection="current")
    assert cfg["framework"]["streams"]["encoding"] == "joint24"
    assert cfg["framework"]["streams"]["conditioning"] == "direct"
    assert cfg["framework"]["state_dropout_prob"] == 0
    assert cfg["framework"]["state_conditioning"] == "full" and cfg["framework"]["visual_conditioning"]
    assert cfg["framework"]["omega"]["train_last_blocks"] == 0
    assert cfg["trainer"]["max_train_steps"] == 20000 and cfg["trainer"]["sample_budget"] == 160000
    assert cfg["trainer"]["gradient_accumulation_steps"] == 2
    assert cfg["datasets"]["vla_data"]["per_device_batch_size"] == 4
    assert cfg["trainer"]["learning_rate"]["base"] == 3e-4
    assert "resume_checkpoint" not in cfg["trainer"]
    args.checkpoint_root.mkdir(parents=True, exist_ok=True)
    probe = args.checkpoint_root / ".dropout_write_probe"
    probe.write_text("C20 state dropout checkpoint storage\n")
    probe.unlink()
    args.out.mkdir(parents=True, exist_ok=False)
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d_%H%M%S")
    cfg["run_id"] = f"omega_joint24_rotate_qrcode_highway_state_dropout20_10000steps_gaas_{stamp}"
    cfg["run_root_dir"] = str(args.checkpoint_root)
    cfg["output_dir"] = str(args.checkpoint_root / cfg["run_id"])
    cfg["config_yaml"] = str(args.out / "production.yaml")
    cfg["framework"]["state_dropout_prob"] = 0.2
    cfg["trainer"].update(max_train_steps=10000, step_limit=10000, sample_budget=80000,
                          planned_training_samples=80000, lr_schedule_steps=20000,
                          training_state_root=str(Path(cfg["output_dir"]) / "training_states"))
    cfg["datasets"]["vla_data"]["statistics_cache_dir"] = str(Path(cfg["output_dir"]) / "data_split")
    after = flatten(cfg)
    differences = {key: dict(reference=before.get(key), prepared=after.get(key))
                   for key in before.keys() | after.keys() if before.get(key) != after.get(key)}
    allowed = {"run_id", "run_root_dir", "output_dir", "config_yaml", "framework.state_dropout_prob",
               "trainer.max_train_steps", "trainer.step_limit", "trainer.sample_budget",
               "trainer.planned_training_samples", "trainer.lr_schedule_steps",
               "trainer.training_state_root", "datasets.vla_data.statistics_cache_dir"}
    assert set(differences) <= allowed and "framework.state_dropout_prob" in differences, differences
    config = args.out / "C_dropout20.yaml"
    config.write_text(yaml.safe_dump(cfg, sort_keys=False))
    manifest = dict(status="prepared", experiment="C_state_dropout20_10K", run_id=cfg["run_id"],
        config=str(config), config_sha256=digest(config), output_dir=cfg["output_dir"],
        reference_study=str(reference), reference_config=str(original), reference_config_sha256=digest(original),
        reference_pbs_job_id=prior["pbs_job_id"], config_differences=differences,
        fixed_batch=True, durable_state=True, preflight_tool="preflight_omega_spatial_highway.py",
        reference_preflight=prior["reference_preflight"], reference_statistics=prior["reference_statistics"],
        expected_trainable_parameters=prior["expected_trainable_parameters"],
        expected_additional_trainable_parameters=prior["expected_additional_trainable_parameters"],
        dataset_checksums=prior["dataset_checksums"], split_ids=prior["split_ids"],
        verify_dataset_checksums=True, dataset_bundle_root=str(Path(cfg["datasets"]["vla_data"]["data_root_dir"]).parent),
        reference_effective_batch=8, target_updates=10000, total_training_observations=80000,
        state_dropout_prob=0.2, lr_schedule_steps=20000, mixed_precision="bf16", initial_seed=42,
        warm_start=False, created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        control_checkpoint="Original C steps_10000_pytorch_model.pt, same 80,000-observation budget and LR curve",
        dropout_semantics="Per-example 20% removal of all four projected state tokens including part embeddings. Task, registers and spatial highway retained. Validation/inference uses full state.",
        scheduler_note="Stop at 10K but retain the original 20K cosine schedule, warmup100 and floor1e-5, so dropout is the training-condition change relative to the prior C 10K milestone.")
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({k:manifest[k] for k in ["config", "output_dir", "target_updates", "state_dropout_prob", "lr_schedule_steps"]}))


if __name__ == "__main__":
    main()
