"""Prepare a fresh joint24 direct run by relocating the retained three-call config."""

import argparse
import datetime
import hashlib
import json
import sys
from pathlib import Path

from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from starVLA.dataloader.episode_split import split_train_validation_test  # noqa: E402


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def flatten(value, prefix=""):
    if not isinstance(value, dict):
        return {prefix: value}
    result = {}
    for key, item in value.items():
        result.update(flatten(item, f"{prefix}.{key}" if prefix else key))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    evidence = ROOT / "docs/experiment_records/20261012/evidence"
    original = evidence / "Checkpoints/omega_stream8_rotate_qrcode_direct_20000steps_20261010_135404/config.full.yaml"
    study = evidence / "omega_camera_stream_training/rotate_qrcode_20261010_135404"
    cfg = OmegaConf.load(original)
    before = flatten(OmegaConf.to_container(cfg, resolve=True))
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d_%H%M%S")
    cfg.run_id = f"omega_joint24_rotate_qrcode_direct_20000steps_gaas_{stamp}"
    cfg.run_root_dir = str(ROOT / "results/Checkpoints")
    cfg.output_dir = str(Path(cfg.run_root_dir) / cfg.run_id)
    cfg.config_yaml = str(args.out / "joint24.yaml")
    cfg.framework.omega.source_root = str(args.bundle / "omega")
    cfg.framework.omega.checkpoint_path = str(args.bundle / "omega/ckpt/vggt_omega_1b_512.pt")
    cfg.framework.streams.encoding = "joint24"
    cfg.datasets.vla_data.data_root_dir = str(args.bundle / "data")
    cfg.datasets.vla_data.statistics_cache_dir = str(Path(cfg.output_dir) / "data_split")
    # Node-local resumable state follows the original run's /tmp storage convention.
    cfg.trainer.training_state_root = f"/tmp/{cfg.run_id}/training_states"
    del cfg.trainer.resume_checkpoint
    after = flatten(OmegaConf.to_container(cfg, resolve=True))
    differences = {key: dict(original=before.get(key), prepared=after.get(key))
                   for key in before.keys() | after.keys() if before.get(key) != after.get(key)}
    expected = {
        "run_id", "run_root_dir", "output_dir", "config_yaml", "framework.omega.source_root",
        "framework.omega.checkpoint_path", "framework.streams.encoding", "datasets.vla_data.data_root_dir",
        "datasets.vla_data.statistics_cache_dir", "trainer.training_state_root", "trainer.resume_checkpoint",
    }
    assert set(differences) == expected, differences
    assert cfg.framework.streams.conditioning == "direct" and cfg.framework.omega.train_last_blocks == 0
    assert cfg.trainer.max_train_steps == 20000 and cfg.datasets.vla_data.per_device_batch_size == 4
    assert cfg.trainer.gradient_accumulation_steps == 2
    dataset = args.bundle / "data/rotate_qrcode"
    split = split_train_validation_test(dataset, 0.1, 0.1, 42)
    reference = json.loads((study / "manifest.json").read_text())
    for ids, key in zip(split, ["train_episode_ids", "validation_episode_ids", "test_episode_ids"], strict=True):
        assert ids == reference[key], key
    bundle_manifest = json.loads((args.bundle / "manifest.json").read_text())
    hashes = {}
    for relative, expected_hash in bundle_manifest["checksums"].items():
        if relative.startswith("data/rotate_qrcode/"):
            actual = digest(args.bundle / relative)
            assert actual == expected_hash, relative
            hashes[relative] = actual
    assert sum(path.endswith(".parquet") for path in hashes) == 100
    assert digest(cfg.framework.omega.checkpoint_path) == cfg.framework.omega.checkpoint_sha256
    OmegaConf.save(cfg, args.out / "joint24.yaml", resolve=True)
    manifest = dict(
        status="prepared", run_id=cfg.run_id, config=cfg.config_yaml, output_dir=cfg.output_dir,
        original_config=str(original), original_config_sha256=digest(original),
        config_sha256=digest(args.out / "joint24.yaml"), config_differences=differences,
        reference_preflight=str(study / "direct_preflight.json"),
        reference_statistics=str(evidence / "Checkpoints/omega_stream8_rotate_qrcode_direct_20000steps_20261010_135404/dataset_statistics.json"),
        dataset_checksums=hashes, split_ids=dict(zip(["train", "validation", "test"], split, strict=True)),
        expected_trainable_parameters=331455502, mixed_precision="bf16",
        reference_updates=20000, reference_effective_batch=8, total_training_observations=160000,
        batch_policy="Benchmark 4x2, 8x1, 16x1; use a >5% throughput improvement with 15% memory headroom",
        schedule_policy="Keep 160K observations; scale update/event counts by selected effective batch; LR remains 3e-4",
        input_order="time-major: head,left,right for each chronological timestamp",
        decoder_memory_order="camera-major: eight chronological timestamps, 17 tokens per image",
        runtime_difference="GAAS GPU and installed software versions; fresh run, no resume from other machine",
        state_storage="Latest full state on compute-node /tmp; copied to durable output when PBS exits normally or with a catchable signal",
    )
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(str(args.out / "manifest.json"))


if __name__ == "__main__":
    main()
