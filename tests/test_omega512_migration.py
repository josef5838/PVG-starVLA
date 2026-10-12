"""Deployment regressions encountered while moving Omega policies between hosts."""

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from omegaconf import OmegaConf

from deployment.model_server.policy_wrapper import PolicyServerWrapper
from deployment.model_server.tools.websocket_policy_client import _expected_image_hw
from starVLA.model.framework.share_tools import read_mode_config


class MigrationTests(unittest.TestCase):
    def test_target_configs_are_fresh_with_matched_budget_and_separate_outputs(self):
        from tools.omega512_cluster_bundle import configure

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bundle, repo = root / "bundle", root / "repo"
            baseline = OmegaConf.load(
                Path(__file__).resolve().parents[1]
                / "examples/simBenchmarks/Robotwin/train_files/omega512_robotwin.yaml"
            )
            runs = []
            for variant in ("dense", "state_query_local"):
                saved = bundle / "runs" / variant
                saved.mkdir(parents=True)
                config = OmegaConf.create(OmegaConf.to_container(baseline, resolve=True))
                config.framework.readout.type = variant
                # A source resume path must never leak into independent experiments.
                config.trainer.resume_checkpoint = "/source/running/training_state"
                OmegaConf.save(config, saved / "config.full.yaml")
                runs.append({"variant": variant, "source_run_id": "old_local_job"})
            (bundle / "manifest.json").write_text(json.dumps({"runs": runs}))
            configure(SimpleNamespace(bundle=bundle, repo=repo))
            outputs = []
            for variant in ("dense", "state_query_local"):
                config = OmegaConf.load(bundle / "configured" / f"{variant}_fresh.yaml")
                self.assertIsNone(config.trainer.get("resume_checkpoint"))
                self.assertIsNone(config.trainer.get("pretrained_checkpoint"))
                self.assertEqual(config.trainer.max_train_steps, 10000)
                self.assertEqual(config.trainer.sample_budget, 80000)
                self.assertEqual(config.framework.readout.type, variant)
                self.assertEqual(config.datasets.vla_data.data_root_dir, str(bundle / "data"))
                outputs.append(config.output_dir)
                smoke = OmegaConf.load(bundle / "configured" / f"{variant}_smoke.yaml")
                self.assertEqual(smoke.trainer.max_train_steps, 16)
                self.assertEqual(smoke.trainer.sample_budget, 128)
            self.assertEqual(len(set(outputs)), 2)
            adaptation = OmegaConf.load(bundle / "configured/adapt_last2_fresh.yaml")
            self.assertEqual(adaptation.framework.omega.train_last_blocks, 2)
            self.assertEqual(adaptation.trainer.learning_rate.omega, 1e-5)

    def test_complete_config_preserves_deployment_fields_with_legacy_fallback(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "checkpoints").mkdir()
            checkpoint = root / "checkpoints/model.pt"
            checkpoint.touch()
            config = {
                "framework": {"name": "Omega512DiT", "action_model": {"action_horizon": 16, "action_dim": 14}},
                "version_id": "0.21",
            }
            OmegaConf.save(OmegaConf.create(config), root / "config.yaml")
            (root / "dataset_statistics.json").write_text("{}")
            self.assertEqual(read_mode_config(checkpoint)[0]["framework"]["name"], "Omega512DiT")
            config["datasets"] = {"vla_data": {"preserve_native_images": True, "instruction_override": "rotate_qrcode"}}
            OmegaConf.save(OmegaConf.create(config), root / "config.full.yaml")
            loaded, _ = read_mode_config(checkpoint)
            self.assertEqual(loaded["datasets"], config["datasets"])

    def test_bfloat16_serving_keeps_backbone_parameters_in_float32(self):
        framework = torch.nn.Module()
        framework.omega = torch.nn.Linear(2, 2)
        framework.action_model = torch.nn.Linear(2, 2)
        config = {"framework": {"name": "Omega512DiT", "action_model": {"action_horizon": 16}}}
        processor = SimpleNamespace(
            unnorm_key="rotate_qrcode", available_unnorm_keys=["rotate_qrcode"], action_keys=[], state_keys=[]
        )
        with (
            patch("deployment.model_server.policy_wrapper.baseframework.from_pretrained", return_value=framework),
            patch(
                "deployment.model_server.policy_wrapper.read_mode_config", return_value=(config, {"rotate_qrcode": {}})
            ),
            patch.object(PolicyServerWrapper, "_get_processor", return_value=processor),
        ):
            wrapper = PolicyServerWrapper("unused.pt", device="cpu", use_bf16=True)
        self.assertEqual(wrapper._framework.omega.weight.dtype, torch.float32)
        self.assertEqual(wrapper._framework.action_model.weight.dtype, torch.bfloat16)

    def test_non_square_metadata_uses_width_height_order(self):
        self.assertEqual(_expected_image_hw({"training_obs_image_size": [640, 480]}), (480, 640))


if __name__ == "__main__":
    unittest.main()
