"""Compare the next stochastic update after a real Accelerate save/load."""

import os

os.environ["STARVLA_DISABLE_DEEPSPEED"] = "1"
os.environ["WANDB_MODE"] = "disabled"

import tempfile
import unittest
from pathlib import Path

import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from starVLA.training.train_starvla import VLATrainer, accelerator


class ResumeTests(unittest.TestCase):
    def test_next_update_and_data_cursor_match_uninterrupted_training(self):
        class Policy(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.tensor(0.0))

            def forward(self, batch):
                return {
                    "action_loss": (
                        (self.weight * batch.float().mean() - torch.rand((), device=self.weight.device)).square()
                    )
                }

        accelerator.gradient_accumulation_steps = 1
        model = Policy()
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)
        loader = DataLoader(torch.arange(1, 13), batch_size=2)
        model, optimizer, loader = accelerator.prepare(model, optimizer, loader)
        trainer = VLATrainer.__new__(VLATrainer)
        trainer.model, trainer.optimizer, trainer.lr_scheduler = model, optimizer, scheduler
        trainer.accelerator, trainer.vla_train_dataloader = accelerator, loader
        trainer.config = OmegaConf.create({"trainer": {"save_training_state": True, "gradient_clipping": None}})
        trainer.completed_steps, trainer.consumed_batches, trainer.total_batch_size = 0, 0, 2
        trainer.best_validation_mse = 0.123
        accelerator.register_for_checkpointing(scheduler, trainer)
        trainer._create_data_iterators()
        trainer._train_step(trainer._get_next_batch())
        trainer.completed_steps += 1
        with tempfile.TemporaryDirectory() as directory:
            trainer._save_training_state(directory)
            self.assertTrue((Path(directory) / "complete.json").exists())
            saved_lr = optimizer.param_groups[0]["lr"]
            batch = trainer._get_next_batch()
            trainer._train_step(batch)
            expected_weight = model.weight.detach().clone()
            expected_lr = optimizer.param_groups[0]["lr"]
            expected_rng = torch.rand(3, device=accelerator.device)
            expected_optimizer_step = optimizer.state[model.weight]["step"].item()

            accelerator.load_state(directory)
            self.assertEqual(trainer.completed_steps, 1)
            self.assertEqual(trainer.consumed_batches, 1)
            self.assertEqual(trainer.best_validation_mse, 0.123)
            self.assertEqual(optimizer.param_groups[0]["lr"], saved_lr)
            trainer._create_data_iterators()
            resumed_batch = trainer._get_next_batch()
            torch.testing.assert_close(resumed_batch, batch)
            trainer._train_step(resumed_batch)
            torch.testing.assert_close(model.weight, expected_weight, rtol=0, atol=0)
            self.assertEqual(optimizer.param_groups[0]["lr"], expected_lr)
            self.assertEqual(optimizer.state[model.weight]["step"].item(), expected_optimizer_step)
            torch.testing.assert_close(torch.rand(3, device=accelerator.device), expected_rng, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
