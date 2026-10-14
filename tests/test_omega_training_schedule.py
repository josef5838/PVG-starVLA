"""Actual scheduler construction keeps the first 10K C learning rates identical."""

import copy
import os
from unittest.mock import patch

os.environ["STARVLA_DISABLE_DEEPSPEED"] = "1"

import torch
from omegaconf import OmegaConf
from starVLA.training.trainer_utils import trainer_tools
from starVLA.training import train_starvla


def test_short_run_matches_reference_cosine_curve_in_both_scheduler_builders():
    cfg = OmegaConf.create(dict(trainer=dict(max_train_steps=20000, num_warmup_steps=100,
        learning_rate=dict(base=3e-4), optimizer=dict(betas=[0.9, 0.95], eps=1e-8, weight_decay=1e-8),
        lr_scheduler_type="cosine_with_min_lr", scheduler_specific_kwargs=dict(min_lr=1e-5))))
    short = copy.deepcopy(cfg)
    short.trainer.max_train_steps = 10000
    short.trainer.lr_schedule_steps = 20000
    adamw = torch.optim.AdamW

    def cpu_optimizer(*args, **kwargs):
        kwargs["fused"] = False
        return adamw(*args, **kwargs)

    with patch("torch.optim.AdamW", side_effect=cpu_optimizer):
        _, reference = train_starvla.setup_optimizer_and_scheduler(torch.nn.Linear(2, 1), cfg)
        _, production = train_starvla.setup_optimizer_and_scheduler(torch.nn.Linear(2, 1), short)
        _, shared = trainer_tools.setup_optimizer_and_scheduler(torch.nn.Linear(2, 1), short)
    for step in range(10001):
        expected = reference.lr_lambdas[0](step)
        assert production.lr_lambdas[0](step) == expected
        assert shared.lr_lambdas[0](step) == expected
    assert 3e-4 * production.lr_lambdas[0](10000) > 1e-5
