"""Larger microbatches must preserve training budget and validation noise grouping."""

import sys
from pathlib import Path

from omegaconf import OmegaConf

from starVLA.training.action_validation import configure_sample_budget

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from tune_omega_joint24_batch import apply_batch_schedule  # noqa: E402


def test_batch_changes_keep_observation_budget_and_schedule_events():
    source = Path(__file__).resolve().parents[1] / "docs/experiment_records/20261012/evidence/Checkpoints/omega_stream8_rotate_qrcode_direct_20000steps_20261010_135404/config.full.yaml"
    original = OmegaConf.load(source)
    for microbatch, accumulation in ((4, 2), (8, 1), (16, 1)):
        cfg = OmegaConf.create(OmegaConf.to_container(original, resolve=True))
        effective = microbatch * accumulation
        apply_batch_schedule(cfg, dict(microbatch=microbatch, accumulation=accumulation, effective_batch=effective))
        assert configure_sample_budget(cfg, 1) == accumulation
        assert cfg.trainer.max_train_steps * effective == 160000
        assert cfg.trainer.validation_batch_size == 4
        for key in ("num_warmup_steps", "save_interval", "eval_interval"):
            assert int(cfg.trainer[key])*effective == int(original.trainer[key])*8
        assert list(cfg.trainer.milestone_updates) == [80000//effective]
        assert cfg.trainer.learning_rate == original.trainer.learning_rate
        assert cfg.trainer.optimizer == original.trainer.optimizer
        assert cfg.datasets.vla_data.validation_samples == original.datasets.vla_data.validation_samples
