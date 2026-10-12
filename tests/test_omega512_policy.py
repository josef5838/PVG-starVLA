"""Visual grounding, readout gradients, preprocessing and resumable Omega policies."""

import copy
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from omegaconf import OmegaConf
from PIL import Image
from torch import nn
from torch.nn import functional as F

from starVLA.model.framework.VLM4A.Omega512DiT import Omega512DiT, Omega512DiTDefaultConfig

sys.path.insert(0, Omega512DiTDefaultConfig().omega["source_root"])
from vggt_omega.utils.load_fn import load_and_preprocess_images


class TinyAggregator(nn.Module):
    depth = 2

    def __init__(self):
        super().__init__()
        self.patch_embed = nn.Linear(3, 2048)
        self.frame_blocks = nn.ModuleList([nn.Linear(1, 1) for _ in range(2)])
        self.inter_frame_blocks = nn.ModuleList([nn.Linear(1, 1) for _ in range(2)])

    def forward(self, images):
        b, views, channels, height, width = images.shape
        patches = F.avg_pool2d(images.flatten(0, 1), 16).flatten(2).transpose(1, 2)
        features = self.patch_embed(patches).reshape(b, views, -1, 2048)
        for frame, cross in zip(self.frame_blocks, self.inter_frame_blocks, strict=False):
            features = features + frame(features.mean(-1, keepdim=True))
            features = features + cross(features.mean(-1, keepdim=True))
        tokens = torch.cat([features[:, :, :1], features], dim=2)
        return [tokens, tokens], 1


class TinyOmega(nn.Module):
    def __init__(self):
        super().__init__()
        self.aggregator = TinyAggregator()
        self.camera_head = nn.Linear(1, 1)


def make_policy(readout="dense", adapt=0, state_dropout=0):
    cfg = OmegaConf.create(
        dict(
            framework=dict(
                omega=dict(feature_layer=1, train_last_blocks=adapt),
                readout=dict(type=readout, queries_per_view=4, local_grid=4, depth=1),
                action_model=dict(action_hidden_dim=16, depth=2, num_heads=4, cross_attention_every=1),
                state_dropout_prob=state_dropout,
                condition_dropout_prob=0,
            )
        )
    )
    with patch.object(Omega512DiT, "_load_omega", side_effect=TinyOmega):
        return Omega512DiT(cfg)


def examples():
    rng = np.random.default_rng(19)
    return [
        dict(
            image=[Image.fromarray(rng.integers(0, 256, (24, 32, 3), dtype=np.uint8)) for _ in range(3)],
            state=rng.normal(size=14).astype(np.float32),
            lang="rotate_qrcode",
            action=rng.normal(size=(16, 14)).astype(np.float32),
        )
    ]


class OmegaPolicyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(4)

    def test_official_loader_identity_and_non_square_shape(self):
        model, batch = make_policy(), examples()
        with tempfile.TemporaryDirectory() as directory:
            paths = []
            for index, image in enumerate(batch[0]["image"]):
                path = Path(directory) / f"{index}.png"
                image.save(path)
                paths.append(str(path))
            actual = model._prepare_images(batch)
            expected = load_and_preprocess_images(paths, image_resolution=512)
            torch.testing.assert_close(actual[0], expected, rtol=0, atol=0)
            self.assertEqual(tuple(actual.shape), (1, 3, 3, 448, 592))

    def test_native_images_survive_dataset_and_evaluation_client(self):
        from types import SimpleNamespace

        from examples.simBenchmarks.Robotwin.eval_files.model2robotwin_interface import ModelClient
        from starVLA.dataloader.gr00t_lerobot.datasets import LeRobotSingleDataset

        image = np.zeros((48, 64, 3), dtype=np.uint8)
        dataset = SimpleNamespace(
            data_cfg={"preserve_native_images": True, "obs_image_size": [224, 224]},
            modality_keys={"video": ["video.head"], "language": ["lang"], "action": ["action"]},
            tag="robot",
        )
        packed = LeRobotSingleDataset._pack_sample(
            dataset, {"video.head": [image], "lang": ["rotate_qrcode"], "action": np.zeros((16, 14), dtype=np.float32)}
        )
        self.assertEqual(packed["image"][0].size, (64, 48))
        client = ModelClient.__new__(ModelClient)
        client.preserve_native_images, client.image_size = True, [224, 224]
        np.testing.assert_array_equal(client._resize_image(image), image)

    def test_visual_and_readout_gradients_with_frozen_backbone(self):
        for readout in ("dense", "state_query_local"):
            model = make_policy(readout).train()
            nn.init.normal_(model.action_model.net.final_layer.linear.weight, std=0.02)
            loss = model(examples())["action_loss"]
            self.assertTrue(torch.isfinite(loss))
            loss.backward()
            self.assertGreater(model.visual_proj.weight.grad.abs().sum().item(), 0)
            self.assertGreater(model.action_model.net.visual_attentions[0].key_value.weight.grad.abs().sum().item(), 0)
            self.assertTrue(all(parameter.grad is None for parameter in model.omega.parameters()))
            self.assertFalse(model.omega.training)
            if model.visual_readout is not None:
                self.assertGreater(model.visual_readout.queries.grad.abs().sum().item(), 0)
                self.assertGreater(model.visual_readout.state_encoder[-1].weight.grad.abs().sum().item(), 0)
                self.assertEqual(model.last_visual_info["memory_tokens"], 60)
            else:
                self.assertEqual(model.last_visual_info["memory_tokens"], 3108)

    def test_image_sensitivity_and_cached_visual_attention(self):
        model = make_policy().eval()
        nn.init.normal_(model.action_model.net.final_layer.linear.weight, std=0.02)
        first = examples()
        second = copy.deepcopy(first)
        second[0]["image"][0] = Image.new("RGB", (32, 24), (255, 0, 0))
        condition, memory = model._condition(first)
        other_condition, other_memory = model._condition(second)
        torch.testing.assert_close(condition, other_condition, rtol=0, atol=0)
        self.assertFalse(torch.allclose(memory, other_memory))
        noise = torch.randn(1, 16, 14)
        time = torch.tensor([50])
        predicted = model.action_model.net(noise, time, condition, model.action_model.net.prepare_visual(memory))
        other = model.action_model.net(noise, time, condition, model.action_model.net.prepare_visual(other_memory))
        self.assertFalse(torch.allclose(predicted, other))
        with patch.object(
            model.action_model.net, "prepare_visual", wraps=model.action_model.net.prepare_visual
        ) as prepare:
            result = model.predict_action(first, num_ddim_steps=2)
            self.assertEqual(prepare.call_count, 1)
            self.assertEqual(result["normalized_actions"].shape, (1, 16, 14))

    def test_state_dropout_covers_queries_and_state_tokens(self):
        model = make_policy("state_query_local", state_dropout=1).train()
        nn.init.normal_(model.visual_readout.state_encoder[-1].weight)
        nn.init.ones_(model.visual_readout.state_encoder[-1].bias)
        first = examples()
        second = copy.deepcopy(first)
        second[0]["state"] += 10
        condition, memory = model._condition(first, apply_dropout=True)
        other_condition, other_memory = model._condition(second, apply_dropout=True)
        torch.testing.assert_close(condition, other_condition, rtol=0, atol=0)
        torch.testing.assert_close(memory, other_memory, rtol=0, atol=0)
        self.assertEqual(condition[:, :4].count_nonzero().item(), 0)

    def test_shared_initialization_and_explicit_task_validation(self):
        torch.manual_seed(42)
        dense = make_policy()
        rng = torch.get_rng_state()
        torch.manual_seed(42)
        query = make_policy("state_query_local")
        torch.testing.assert_close(torch.get_rng_state(), rng, rtol=0, atol=0)
        for name, value in dense.state_dict().items():
            torch.testing.assert_close(value, query.state_dict()[name], rtol=0, atol=0)
        batch = examples()
        batch[0]["lang"] = "unregistered task"
        with self.assertRaisesRegex(ValueError, "Unknown task"):
            dense._condition(batch)

    def test_interventions_remove_state_routes_without_zeroing_pose(self):
        model = make_policy("state_query_local").eval()
        nn.init.normal_(model.visual_readout.state_encoder[-1].weight)
        first, second = examples(), examples()
        second[0]["state"] += 2
        for mode in ("full", "none", "decoder_only", "readout_only"):
            condition, memory = model._condition(first, state_conditioning=mode)
            other_condition, other_memory = model._condition(second, state_conditioning=mode)
            self.assertEqual(torch.equal(condition, other_condition), mode in ("none", "readout_only"))
            self.assertEqual(torch.equal(memory, other_memory), mode in ("none", "decoder_only"))
        condition, memory = model._condition(first, state_conditioning="none")
        self.assertEqual(condition[:, :4].count_nonzero().item(), 0)
        self.assertGreater(condition[:, 4:].count_nonzero().item(), 0)
        with self.assertRaisesRegex(ValueError, "Invalid state"):
            model._condition(first, state_conditioning="typo")

    def test_state_only_control_has_no_image_or_query_state_leak(self):
        model = make_policy("state_query_local").eval()
        first, second = examples(), examples()
        second[0]["image"] = [Image.new("RGB", im.size) for im in second[0]["image"]]
        condition, memory = model._condition(first, visual_conditioning=False)
        other_condition, other_memory = model._condition(second, visual_conditioning=False)
        torch.testing.assert_close(condition, other_condition, rtol=0, atol=0)
        torch.testing.assert_close(memory, other_memory, rtol=0, atol=0)
        self.assertEqual(memory.count_nonzero().item(), 0)

    def test_fixed_diffusion_noise_does_not_depend_on_global_rng(self):
        model = make_policy().eval()
        nn.init.normal_(model.action_model.net.final_layer.linear.weight, std=0.02)
        batch = examples()
        first = model.predict_action(batch, num_ddim_steps=2, diffusion_seed=91)["normalized_actions"]
        torch.randn(123)
        second = model.predict_action(batch, num_ddim_steps=2, diffusion_seed=91)["normalized_actions"]
        np.testing.assert_array_equal(first, second)
        other = model.predict_action(batch, num_ddim_steps=2, diffusion_seed=92)["normalized_actions"]
        self.assertFalse(np.array_equal(first, other))

    def test_adapted_blocks_are_saved_and_missing_policy_weights_fail(self):
        model = make_policy(adapt=1).train()
        nn.init.normal_(model.action_model.net.final_layer.linear.weight, std=0.02)
        model(examples())["action_loss"].backward()
        self.assertIsNotNone(model.omega.aggregator.frame_blocks[-1].weight.grad)
        self.assertIsNone(model.omega.aggregator.frame_blocks[0].weight.grad)
        state = model.state_dict()
        self.assertIn("omega.aggregator.frame_blocks.1.weight", state)
        self.assertNotIn("omega.aggregator.frame_blocks.0.weight", state)
        with self.assertRaisesRegex(RuntimeError, "Incomplete Omega"):
            model.load_state_dict({name: value for name, value in state.items() if name != "visual_proj.weight"})
        with self.assertRaisesRegex(RuntimeError, "Incomplete Omega"):
            model.load_state_dict(
                {name: value for name, value in state.items() if name != "omega.aggregator.frame_blocks.1.weight"}
            )

    def test_accelerate_resume_matches_next_stochastic_update(self):
        from accelerate import Accelerator

        for readout in ("dense", "state_query_local"):
            torch.manual_seed(7)
            model = make_policy(readout).train()
            optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=0.001)
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)
            accelerator = Accelerator(cpu=True)
            model, optimizer = accelerator.prepare(model, optimizer)
            accelerator.register_for_checkpointing(scheduler)
            batch = examples()

            def update(accelerator=accelerator, batch=batch, model=model, optimizer=optimizer, scheduler=scheduler):
                optimizer.zero_grad()
                accelerator.backward(model(batch)["action_loss"])
                optimizer.step()
                scheduler.step()

            update()
            with tempfile.TemporaryDirectory() as directory:
                accelerator.save_state(directory, safe_serialization=False)
                saved = torch.load(Path(directory) / "pytorch_model.bin", weights_only=True)
                self.assertFalse(any(key.startswith("omega.") for key in saved))
                update()
                expected = {key: value.clone() for key, value in model.state_dict().items()}
                expected_rng = torch.rand(3)
                expected_lr = optimizer.param_groups[0]["lr"]
                accelerator.load_state(directory)
                update()
                for key, value in model.state_dict().items():
                    torch.testing.assert_close(value, expected[key], rtol=0, atol=0)
                torch.testing.assert_close(torch.rand(3), expected_rng, rtol=0, atol=0)
                self.assertEqual(optimizer.param_groups[0]["lr"], expected_lr)
            accelerator.free_memory()


if __name__ == "__main__":
    unittest.main()
