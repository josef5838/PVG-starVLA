"""Prepare experiment C from the running A production config, with a fixed budget."""

import argparse
import datetime
import json
from pathlib import Path

from omegaconf import OmegaConf

from prepare_omega_joint24_gaas import digest, flatten


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-study", type=Path, required=True)
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.baseline_study = args.baseline_study.resolve()
    args.out = args.out.resolve()
    args.checkpoint_root = args.checkpoint_root.resolve()
    original = args.baseline_study / "production.yaml"
    baseline = json.loads((args.baseline_study / "manifest.json").read_text())
    reference = json.loads((args.baseline_study / "preflight.json").read_text())
    assert reference["status"] == "passed" and baseline["preflight_passed"]
    cfg = OmegaConf.load(original)
    before = flatten(OmegaConf.to_container(cfg, resolve=True))
    assert cfg.framework.streams.encoding == "joint24" and cfg.framework.streams.conditioning == "direct"
    assert cfg.framework.streams.temporal_embedding == "learned"
    assert cfg.framework.omega.train_last_blocks == 0
    assert cfg.framework.action_model.action_hidden_dim == 1024 and cfg.framework.streams.width == 512
    assert cfg.framework.action_model.depth == 24 and cfg.framework.action_model.cross_attention_every == 4
    assert cfg.seed == 42 and cfg.trainer.learning_rate.base == 3e-4
    assert cfg.datasets.vla_data.per_device_batch_size == 4 and cfg.trainer.gradient_accumulation_steps == 2
    assert cfg.trainer.max_train_steps == 20000 and cfg.trainer.sample_budget == 160000
    assert "resume_checkpoint" not in cfg.trainer
    args.checkpoint_root.mkdir(parents=True, exist_ok=True)
    probe = args.checkpoint_root / ".highway_write_probe"
    probe.write_text("C checkpoint storage\n")
    probe.unlink()
    args.out.mkdir(parents=True, exist_ok=False)
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d_%H%M%S")
    cfg.run_id = f"omega_joint24_rotate_qrcode_current_patch_highway_20000steps_gaas_{stamp}"
    cfg.run_root_dir = str(args.checkpoint_root)
    cfg.output_dir = str(args.checkpoint_root / cfg.run_id)
    cfg.config_yaml = str(args.out / "C.yaml")
    cfg.framework.spatial_highway = dict(enabled=True, selection="current")
    cfg.datasets.vla_data.statistics_cache_dir = str(Path(cfg.output_dir) / "data_split")
    cfg.trainer.training_state_root = str(Path(cfg.output_dir) / "training_states")
    after = flatten(OmegaConf.to_container(cfg, resolve=True))
    differences = {key: dict(baseline=before.get(key), prepared=after.get(key))
                   for key in before.keys() | after.keys() if before.get(key) != after.get(key)}
    expected = {"run_id", "run_root_dir", "output_dir", "config_yaml",
                "framework.spatial_highway.enabled", "framework.spatial_highway.selection",
                "datasets.vla_data.statistics_cache_dir", "trainer.training_state_root"}
    assert set(differences) == expected, differences
    OmegaConf.save(cfg, args.out / "C.yaml", resolve=True)
    # Six decoder branches: query, KV, output projections plus norms and scalar gates.
    w, d, n = 512, 1024, 6
    additional = (2*2048 + 2048*w+w + 2*w+w+w*w+w + w + w*d+d + 2*d
                  + n*(4*d*d+4*d+2*d) + 2*d+n)
    manifest = dict(
        status="prepared", experiment="C", run_id=cfg.run_id, config=cfg.config_yaml,
        output_dir=cfg.output_dir, baseline_study=str(args.baseline_study),
        baseline_pbs_job_id=baseline["pbs_job_id"], baseline_config=str(original),
        baseline_config_sha256=digest(original), config_sha256=digest(args.out / "C.yaml"),
        config_differences=differences, fixed_batch=True, durable_state=True,
        preflight_tool="preflight_omega_spatial_highway.py",
        reference_preflight=str(args.baseline_study / "preflight.json"),
        reference_statistics=str(args.baseline_study / "preflight_dataset_statistics.json"),
        expected_common_trainable_parameters=reference["trainable_parameters"],
        expected_additional_trainable_parameters=additional,
        expected_trainable_parameters=reference["trainable_parameters"]+additional,
        dataset_checksums=baseline["dataset_checksums"], split_ids=baseline["split_ids"],
        reference_effective_batch=8, reference_updates=20000, total_training_observations=160000,
        mixed_precision="bf16", initial_seed=42, warm_start=False,
        state_storage="Latest full optimizer/RNG/cursor state and policy weights on shared project storage",
        architecture="A register path plus latest three-camera patch memory, XY/time/camera/type embeddings, six zero-gated attention branches with cached KV",
        preflight_policy="Exact common initialization/data/statistics parity; nonvacuous zero-gate equivalence; four disposable updates opening spatial gradients; fresh training discards probe weights",
    )
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
    print(json.dumps({key: manifest[key] for key in ("config", "output_dir", "expected_trainable_parameters", "expected_additional_trainable_parameters")}))


if __name__ == "__main__":
    main()
