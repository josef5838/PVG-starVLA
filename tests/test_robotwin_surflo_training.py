"""Regression checks for Surflo's RoboTwin training/evaluation contract."""

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
from omegaconf import OmegaConf

from deployment.model_server.policy_norm_processor import PolicyNormProcessor
from deployment.model_server.policy_wrapper import PolicyServerWrapper
from examples.simBenchmarks.Robotwin.eval_files.model2robotwin_interface import ModelClient
from starVLA.dataloader.episode_split import split_episodes
from starVLA.model.framework.VLM4A.SurfloDiT import SurfloDiT
from starVLA.training.action_validation import configure_sample_budget, evaluate_action_batches


class TrainingContractTests(unittest.TestCase):
    def test_fixed_instruction_bypasses_dataset_language_annotations(self):
        from starVLA.dataloader.gr00t_lerobot.datasets import LeRobotSingleDataset

        dataset = SimpleNamespace(
            data_cfg={"instruction_override": "rotate_qrcode"}, delta_indices={"annotation.human.task": np.array([0, 1])}
        )
        self.assertEqual(
            LeRobotSingleDataset.get_language(dataset, 0, "annotation.human.task", 0), ["rotate_qrcode", "rotate_qrcode"]
        )
        dataset.data_cfg["instruction_override"] = ""
        with self.assertRaises(ValueError):
            LeRobotSingleDataset.get_language(dataset, 0, "annotation.human.task", 0)

    def test_episode_split_is_reproducible_and_disjoint(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "meta"
            path.mkdir()
            (path / "episodes.jsonl").write_text("\n".join(json.dumps({"episode_index": i}) for i in range(100)))
            train, val = split_episodes(directory, 0.1, 42)
            self.assertEqual((len(train), len(val)), (90, 10))
            self.assertFalse(set(train) & set(val))
            self.assertEqual(set(train) | set(val), set(range(100)))
            self.assertEqual((train, val), split_episodes(directory, 0.1, 42))

    def test_budget_counts_accumulation_and_processes(self):
        config = OmegaConf.create(
            {
                "trainer": {"gradient_accumulation_steps": 2, "sample_budget": 20000, "max_train_steps": 2500},
                "datasets": {"vla_data": {"per_device_batch_size": 4}},
            }
        )
        self.assertEqual(configure_sample_budget(config, 2), 2)
        self.assertEqual(config.trainer.max_train_steps, 1250)
        self.assertEqual(config.trainer.planned_training_samples, 20000)
        config.datasets.vla_data.per_device_batch_size = 1
        configure_sample_budget(config, 1)
        self.assertEqual(config.trainer.max_train_steps, 10000)
        self.assertEqual(config.trainer.planned_training_samples, 20000)
        config.datasets.vla_data.per_device_batch_size = 4
        config.trainer.step_limit = 2
        configure_sample_budget(config, 1)
        self.assertEqual(config.trainer.planned_training_samples, 16)

    def test_validation_disables_dropout_restores_modes_and_rng(self):
        class Policy(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.layer = torch.nn.Linear(1, 1)
                self.dropout = torch.nn.Dropout(0.9)
                self.frozen = torch.nn.Linear(1, 1).eval()

            def predict_action(self, examples, **kwargs):
                self.assert_steps = kwargs["num_ddim_steps"]
                return {"normalized_actions": self.dropout(torch.ones(len(examples), 16, 14)).numpy()}

        policy = Policy()
        before = torch.get_rng_state().clone()
        batch = [{"action": np.zeros((16, 14))}] * 4
        error, count, samples = evaluate_action_batches(policy, [batch], num_ddim_steps=20)
        self.assertEqual(error / count, 1.0)
        self.assertEqual(samples, 4)
        self.assertTrue(policy.training)
        self.assertFalse(policy.frozen.training)
        self.assertTrue(torch.equal(before, torch.get_rng_state()))
        self.assertEqual(policy.assert_steps, 20)

    def test_frozen_surflo_stays_in_eval_mode(self):
        model = SurfloDiT.__new__(SurfloDiT)
        torch.nn.Module.__init__(model)
        model.surflo_backbone = torch.nn.Linear(1, 1)
        model.freeze_surflo = True
        model.eval()
        model.train()
        self.assertTrue(model.training)
        self.assertFalse(model.surflo_backbone.training)

    def test_old_checkpoint_metadata_keeps_surflo_observation_contract(self):
        wrapper = PolicyServerWrapper.__new__(PolicyServerWrapper)
        wrapper._model_cfg = {
            "framework": {"name": "SurfloDiT", "surflo": {"image_size": 518}},
            "evaluation": {"execute_horizon": 8, "num_ddim_steps": 20},
        }
        wrapper._ckpt_path = "unused"
        wrapper._action_chunk_size = 16
        wrapper._available_unnorm_keys = []
        wrapper._default_unnorm_key = None
        self.assertTrue(wrapper.metadata["include_state"])
        self.assertEqual(wrapper.metadata["training_obs_image_size"], [518, 518])
        self.assertEqual(wrapper.metadata["execute_horizon"], 8)

    def test_ddim_schedule_changes_when_requested(self):
        model = SurfloDiT.__new__(SurfloDiT)
        torch.nn.Module.__init__(model)
        model.action_dim, model.action_horizon = 14, 16
        model._condition = lambda examples, **kwargs: torch.zeros(len(examples), 136, 4)
        calls = []

        class Action:
            net = torch.nn.Linear(4, 4)
            ddim_diffusion = None

            def create_ddim(self, ddim_step):
                calls.append(ddim_step)
                self.ddim_diffusion = SimpleNamespace(ddim_sample_loop=lambda fn, shape, noise, **kwargs: noise)

        model.action_model = Action()
        for steps in (20, 10, 10):
            out = model.predict_action([{}], num_ddim_steps=steps)
            self.assertEqual(out["normalized_actions"].shape, (1, 16, 14))
        self.assertEqual(calls, [20, 10])

    def test_state_normalization_matches_training_transform(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "checkpoints").mkdir()
            checkpoint = root / "checkpoints/model.pt"
            checkpoint.touch()
            config = OmegaConf.load(
                "examples/simBenchmarks/Robotwin/train_files/surflo_camera3_body_parts_robotwin.yaml"
            )
            OmegaConf.save(config, root / "config.yaml")
            stats = {
                name: {
                    "min": [0.0] * 14,
                    "max": [2.0] * 14,
                    "mean": [1.0] * 14,
                    "std": [1.0] * 14,
                    "q01": [0.0] * 14,
                    "q99": [2.0] * 14,
                }
                for name in ("action", "state")
            }
            (root / "dataset_statistics.json").write_text(json.dumps({"robotwin_surflo": stats}))
            proc = PolicyNormProcessor(str(checkpoint))
            raw = np.array([[1.0] * 12 + [0.2, 0.8]], dtype=np.float32)
            normalized = proc.apply_state(raw)
            np.testing.assert_array_equal(normalized[:, :12], np.zeros((1, 12)))
            np.testing.assert_array_equal(normalized[:, 12:], [[0.0, 1.0]])
            self.assertFalse(np.array_equal(raw, normalized))
            np.testing.assert_array_equal(raw, np.array([[1.0] * 12 + [0.2, 0.8]], dtype=np.float32))

    def test_client_reorders_state_and_replans_after_eight_actions(self):
        class Client:
            def __init__(self, *args):
                self.requests = []

            def get_server_metadata(self):
                return {
                    "action_chunk_size": 16,
                    "include_state": True,
                    "training_obs_image_size": [518, 518],
                    "execute_horizon": 8,
                    "num_ddim_steps": 20,
                }

            def predict_action(self, request):
                self.requests.append(request)
                values = np.arange(16 * 14).reshape(1, 16, 14)
                return {"actions": values}

        with patch("examples.simBenchmarks.Robotwin.eval_files.model2robotwin_interface.WebsocketClientPolicy", Client):
            model = ModelClient("unused")
            image = np.zeros((32, 32, 3), dtype=np.uint8)
            example = {"image": [image] * 3, "state": np.arange(14), "lang": "task"}
            for step in range(9):
                model.step(example, step)
            self.assertEqual(len(model.client.requests), 2)
            sent = model.client.requests[0]
            np.testing.assert_array_equal(sent["examples"][0]["state"], [[0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12, 6, 13]])
            self.assertEqual(sent["examples"][0]["image"][0].shape, (518, 518, 3))
            self.assertEqual(sent["num_ddim_steps"], 20)
            self.assertEqual(example["image"][0].shape, (32, 32, 3))
            model.reset("task")
            model.step(example, 0)
            self.assertEqual(len(model.client.requests), 3)
            self.assertNotIn("diffusion_seed", sent)
            model.diffusion_seed = 4000
            model.reset("task")
            model.step(example, 0)
            model.step(example, 8)
            self.assertEqual(model.client.requests[-2]["diffusion_seed"], 4000)
            self.assertEqual(model.client.requests[-1]["diffusion_seed"], 4001)

    def test_two_batches_accumulate_before_optimizer_update(self):
        import os

        os.environ["STARVLA_DISABLE_DEEPSPEED"] = "1"
        from starVLA.training.train_starvla import VLATrainer, accelerator

        previous_steps = accelerator.gradient_accumulation_steps
        accelerator.gradient_accumulation_steps = 2

        class Policy(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.tensor(0.0))

            def forward(self, example):
                x, y = example
                return {"action_loss": (self.weight * x - y).square()}

        model = Policy()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        model, optimizer = accelerator.prepare(model, optimizer)
        trainer = VLATrainer.__new__(VLATrainer)
        trainer.model, trainer.optimizer, trainer.accelerator = model, optimizer, accelerator
        trainer.completed_steps, trainer.total_batch_size = 0, 2
        trainer.config = SimpleNamespace(trainer=SimpleNamespace(gradient_clipping=None))
        updates = []
        trainer.lr_scheduler = SimpleNamespace(step=lambda: updates.append(1))
        try:
            trainer.optimizer.zero_grad()
            trainer._train_step((torch.tensor(1.0), torch.tensor(1.0)))
            self.assertEqual(model.weight.item(), 0.0)
            trainer._train_step((torch.tensor(3.0), torch.tensor(1.0)))
            # Combined-batch loss has gradient -4 at w=0, so SGD yields .4.
            self.assertAlmostEqual(model.weight.item(), 0.4, places=6)
            self.assertEqual(len(updates), 1)
        finally:
            accelerator.gradient_accumulation_steps = previous_steps

    def test_server_normalizes_state_once_without_mutating_request(self):
        wrapper = PolicyServerWrapper.__new__(PolicyServerWrapper)
        wrapper._default_unnorm_key = "robotwin_surflo"
        normalized = []

        class Processor:
            def apply_state(self, state):
                normalized.append(state)
                return np.asarray(state) + 10

            def unapply_actions(self, action):
                return action

        class Framework:
            def predict_action(self, examples, **kwargs):
                np.testing.assert_array_equal(examples[0]["state"], [[10.0] * 14])
                return {"normalized_actions": np.zeros((1, 16, 14))}

        wrapper._get_processor = lambda key: Processor()
        wrapper._framework = Framework()
        example = {"state": np.zeros((1, 14))}
        wrapper.predict_action([example])
        self.assertEqual(len(normalized), 1)
        np.testing.assert_array_equal(example["state"], np.zeros((1, 14)))


if __name__ == "__main__":
    unittest.main()
