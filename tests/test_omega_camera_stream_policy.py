"""Camera/history isolation, gradient flow and matched stream-policy initialization."""

import copy
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
from omegaconf import OmegaConf
from PIL import Image
from torch import nn

from starVLA.dataloader.gr00t_lerobot.datasets import LeRobotSingleDataset
from starVLA.model.framework.VLM4A.OmegaCameraStreamDiT import OmegaCameraStreamDiT


class TinyAggregator(nn.Module):
    depth = 2

    def __init__(self):
        super().__init__()
        self.projection = nn.Linear(3, 2048)
        self.frame_blocks = nn.ModuleList([nn.Linear(1, 1) for _ in range(2)])
        self.inter_frame_blocks = nn.ModuleList([nn.Linear(1, 1) for _ in range(2)])
        self.calls = []

    def forward(self, images):
        self.calls.append(images.detach().clone())
        features = self.projection(images.mean((-1, -2)))
        features = features + features.mean(1, keepdim=True)
        features = features[:, :, None].expand(-1, -1, 18, -1)
        return [features, features], 17


class TinyOmega(nn.Module):
    def __init__(self):
        super().__init__()
        self.aggregator = TinyAggregator()


def prepare_images(self, rows):
    pixels = [
        [torch.tensor(np.asarray(im)[0, 0].copy(), dtype=torch.float32) / 255 for im in row["image"]] for row in rows
    ]
    return torch.stack([torch.stack(row) for row in pixels])[:, :, :, None, None].expand(-1, -1, -1, 2, 2)


def policy(kind, encoding="independent"):
    config = OmegaConf.create(
        dict(
            framework=dict(
                omega=dict(feature_layer=1),
                task_names=["adjust_bottle"],
                streams=dict(
                    history_length=8,
                    conditioning=kind,
                    encoding=encoding,
                    width=16,
                    depth=2,
                    heads=4,
                    output_queries=4,
                    temporal_rope=False,
                ),
                action_model=dict(action_hidden_dim=16, depth=2, num_heads=4, cross_attention_every=1),
                state_dropout_prob=0.0,
                condition_dropout_prob=0.0,
            )
        )
    )
    with patch.object(OmegaCameraStreamDiT, "_load_omega", side_effect=TinyOmega):
        return OmegaCameraStreamDiT(config)


def example(offset=0):
    history = [
        [Image.new("RGB", (4, 4), (20 + offset + 40 * c + 2 * t, 30 + 3 * t, 60 + 10 * c)) for t in range(8)]
        for c in range(3)
    ]
    return dict(
        image=[view[-1] for view in history],
        image_history=history,
        state=np.arange(14, dtype=np.float32) / 14,
        action=np.ones((16, 14), dtype=np.float32) * 0.2,
        lang="adjust_bottle",
    )


class StreamPolicyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_history_packing_is_chronological_and_preserves_current_images(self):
        videos = ["video.head", "video.left", "video.right"]
        dataset = SimpleNamespace(
            modality_keys=dict(video=videos, language=["lang"], action=["action"]),
            delta_indices={v: np.arange(0, -8, -1) for v in videos},
            tag="robot",
            data_cfg=dict(pack_image_history=True, preserve_native_images=True),
        )
        data = {v: np.stack([np.full((3, 4, 3), 100 - t, dtype=np.uint8) for t in range(8)]) for v in videos}
        data.update(lang=["adjust_bottle"], action=np.zeros((16, 14)))
        sample = LeRobotSingleDataset._pack_sample(dataset, data)
        self.assertEqual([np.asarray(im)[0, 0, 0] for im in sample["image_history"][0]], list(range(93, 101)))
        self.assertEqual(np.asarray(sample["image"][0])[0, 0, 0], 100)

    @patch("starVLA.model.framework.VLM4A.Omega512DiT.Omega512DiT._prepare_images", prepare_images)
    def test_calls_separate_physical_cameras_and_batch_elements(self):
        model = policy("direct").eval()
        a, b = example(), example(5)
        batched = model._history_tokens([a, b])
        calls = model.omega.aggregator.calls
        self.assertEqual(len(calls), 3)
        self.assertEqual(tuple(batched.shape), (2, 3, 8, 17, 2048))
        for c in range(3):
            self.assertEqual(tuple(calls[c].shape), (2, 8, 3, 2, 2))
            self.assertAlmostEqual(float(calls[c][0, 0, 0, 0, 0]), (20 + 40 * c) / 255, places=6)
        alone = model._history_tokens([a])
        torch.testing.assert_close(batched[0], alone[0])

    @patch("starVLA.model.framework.VLM4A.Omega512DiT.Omega512DiT._prepare_images", prepare_images)
    def test_gradients_and_historical_image_sensitivity(self):
        for kind, count in [("direct", 408), ("fused", 4)]:
            model = policy(kind).train()
            nn.init.normal_(model.action_model.net.final_layer.linear.weight, std=0.02)
            loss = model([example()])["action_loss"]
            loss.backward()
            self.assertTrue(torch.isfinite(loss))
            self.assertGreater(model.visual_proj.weight.grad.abs().sum(), 0)
            self.assertGreater(model.time_embeddings.grad.abs().sum(), 0)
            self.assertTrue(all(p.grad is None for p in model.omega.parameters()))
            self.assertEqual(model.last_visual_info["memory_tokens"], count)
            if kind == "fused":
                self.assertGreater(model.visual_readout.queries.grad.abs().sum(), 0)
            model.eval()
            a = example()
            b = copy.deepcopy(a)
            b["image_history"][1][0] = Image.new("RGB", (4, 4), (255, 0, 0))
            z, x = model._condition([a])
            zz, xx = model._condition([b])
            torch.testing.assert_close(z, zz)
            self.assertFalse(torch.allclose(x, xx))
            weights = model.state_dict()
            self.assertFalse(any(k.startswith("omega.") for k in weights))
            model.load_state_dict(weights)

    def test_direct_and_fused_share_all_common_initial_weights_and_rng(self):
        torch.manual_seed(42)
        direct = policy("direct")
        rng = torch.get_rng_state()
        torch.manual_seed(42)
        fused = policy("fused")
        torch.testing.assert_close(rng, torch.get_rng_state(), rtol=0, atol=0)
        for name, value in direct.state_dict().items():
            torch.testing.assert_close(value, fused.state_dict()[name], rtol=0, atol=0)

    @patch("starVLA.model.framework.VLM4A.Omega512DiT.Omega512DiT._prepare_images", prepare_images)
    def test_joint24_order_context_and_batch_isolation(self):
        model = policy("direct", "joint24").eval()
        a, b = example(), example(5)
        raw = model._history_tokens([a, b])
        calls = model.omega.aggregator.calls
        self.assertEqual(len(calls), 1)
        self.assertEqual(tuple(calls[0].shape), (2, 24, 3, 2, 2))
        self.assertEqual(tuple(raw.shape), (2, 3, 8, 17, 2048))
        for t in range(8):
            for c in range(3):
                self.assertAlmostEqual(float(calls[0][0, 3*t+c, 0, 0, 0]), (20+40*c+2*t)/255, places=6)
        layers, _ = model.omega.aggregator(calls[0])
        expected = layers[1][:, :, :17].reshape(2, 8, 3, 17, 2048).transpose(1, 2)
        torch.testing.assert_close(raw, expected)
        torch.testing.assert_close(raw[0], model._history_tokens([a])[0])
        changed = copy.deepcopy(a)
        changed["image_history"][1][0] = Image.new("RGB", (4, 4), (255, 0, 0))
        # Joint Omega attention must allow a different camera to change head tokens.
        self.assertFalse(torch.allclose(raw[0, 0], model._history_tokens([changed])[0, 0]))

    def test_joint24_keeps_identical_parameters_initialization_and_rng(self):
        torch.manual_seed(42)
        independent = policy("direct")
        rng = torch.get_rng_state()
        torch.manual_seed(42)
        joint = policy("direct", "joint24")
        torch.testing.assert_close(rng, torch.get_rng_state(), rtol=0, atol=0)
        self.assertEqual(set(independent.state_dict()), set(joint.state_dict()))
        for name, value in independent.state_dict().items():
            torch.testing.assert_close(value, joint.state_dict()[name], rtol=0, atol=0)

    @patch("starVLA.model.framework.VLM4A.Omega512DiT.Omega512DiT._prepare_images", prepare_images)
    def test_joint24_backward_preserves_frozen_omega_and_direct_memory(self):
        model = policy("direct", "joint24").train()
        nn.init.normal_(model.action_model.net.final_layer.linear.weight, std=0.02)
        loss = model([example()])["action_loss"]
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertGreater(model.visual_proj.weight.grad.abs().sum(), 0)
        self.assertGreater(model.time_embeddings.grad.abs().sum(), 0)
        self.assertTrue(all(p.grad is None for p in model.omega.parameters()))
        self.assertEqual(model.last_visual_info["memory_tokens"], 408)
        self.assertEqual(model.last_visual_info["calls_per_batch"], 1)
        self.assertEqual(model.last_visual_info["images_per_call"], 24)


if __name__ == "__main__":
    unittest.main()
