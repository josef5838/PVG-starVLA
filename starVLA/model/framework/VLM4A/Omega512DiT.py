"""VGGT-Omega policy with dense or state-query/local visual memory."""

import hashlib
import logging
import os
import sys
from contextlib import nullcontext
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import nn
from torchvision.transforms.functional import to_pil_image, to_tensor

from starVLA.model.framework.base_framework import baseframework
from starVLA.model.framework.share_tools import merge_framework_config
from starVLA.model.modules.action_model.OmegaCrossAttentionHeader import OmegaCrossAttentionActionModel
from starVLA.model.modules.projector.omega_readout import StateQueryLocalReadout
from starVLA.model.tools import FRAMEWORK_REGISTRY

OFFICIAL_SHA256 = "c02da418b18bb01d0392598d3f6147366bcde1bb70fd08a5e3bf7925b0667934"
DEFAULT_OMEGA_ROOT = Path(__file__).resolve().parents[4] / "third_party" / "vggt-omega"
STATE_PARTS = (slice(0, 6), slice(12, 13), slice(6, 12), slice(13, 14))


@dataclass
class Omega512DiTDefaultConfig:
    name: str = "Omega512DiT"
    omega: dict = field(
        default_factory=lambda: dict(
            source_root=os.environ.get("OMEGA_ROOT", str(DEFAULT_OMEGA_ROOT)),
            checkpoint_path=os.environ.get(
                "OMEGA_CHECKPOINT",
                str(Path(os.environ.get("OMEGA_ROOT", str(DEFAULT_OMEGA_ROOT))) / "ckpt/vggt_omega_1b_512.pt"),
            ),
            checkpoint_sha256=OFFICIAL_SHA256,
            image_resolution=512,
            num_views=3,
            feature_layer=23,
            train_last_blocks=0,
            weights_mode="reference",
        )
    )
    readout: dict = field(default_factory=lambda: dict(type="dense", queries_per_view=64, local_grid=16, depth=2))
    action_model: dict = field(
        default_factory=lambda: dict(
            action_model_type="DiT-L",
            action_dim=14,
            action_hidden_dim=1024,
            action_horizon=16,
            state_dim=14,
            num_condition_tokens=5,
            depth=24,
            num_heads=16,
            cross_attention_every=4,
        )
    )
    task_names: list = field(default_factory=lambda: ["rotate_qrcode"])
    state_scale: float = 1.0
    state_dropout_prob: float = 0.2
    condition_dropout_prob: float = 0.1
    # These switches mask complete conditioning routes, including projection
    # biases. They leave initialization and checkpoint parameter names intact.
    state_conditioning: str = "full"
    visual_conditioning: bool = True


@FRAMEWORK_REGISTRY.register("Omega512DiT")
class Omega512DiT(baseframework):
    def __init__(self, config=None, **kwargs):
        super().__init__()
        self.config = merge_framework_config(Omega512DiTDefaultConfig, config)
        cfg = self.config.framework
        if cfg.omega.weights_mode not in {"reference", "embedded"}:
            raise ValueError("Omega weights_mode must be reference or embedded")
        if int(cfg.omega.image_resolution) != 512:
            raise ValueError("The Omega512 policy requires the official 512 resolution budget")
        if int(cfg.action_model.state_dim) != 14 or int(cfg.omega.num_views) != 3:
            raise ValueError("This policy requires the RoboTwin 14-D state and three simultaneous cameras")
        for probability in (cfg.state_dropout_prob, cfg.condition_dropout_prob):
            if not 0 <= probability <= 1:
                raise ValueError("Condition dropout probabilities must be in [0,1]")
        if cfg.state_conditioning not in {"full", "none", "decoder_only", "readout_only"}:
            raise ValueError("Invalid state_conditioning route")
        self.omega = self._load_omega()
        self.omega.requires_grad_(False).eval()
        count = int(cfg.omega.train_last_blocks)
        depth = self.omega.aggregator.depth
        if count < 0 or count > depth:
            raise ValueError("Invalid number of Omega blocks to adapt")
        self.adapted_prefixes = []
        for kind in ("frame_blocks", "inter_frame_blocks"):
            for index in range(depth - count, depth):
                getattr(self.omega.aggregator, kind)[index].requires_grad_(True)
                self.adapted_prefixes.append(f"omega.aggregator.{kind}.{index}.")
        self.reference_keys = {
            "omega." + key
            for key in self.omega.state_dict()
            if not any(("omega." + key).startswith(prefix) for prefix in self.adapted_prefixes)
        }
        width = int(cfg.action_model.action_hidden_dim)
        self.visual_norm = nn.LayerNorm(2048)
        self.visual_proj = nn.Linear(2048, width)
        self.position_proj = nn.Sequential(nn.Linear(2, width), nn.SiLU(), nn.Linear(width, width))
        self.view_embeddings = nn.Parameter(torch.randn(3, width) * 0.02)
        self.state_proj = nn.ModuleList([nn.Linear(part.stop - part.start, width) for part in STATE_PARTS])
        self.state_part_embeddings = nn.Parameter(torch.randn(4, width) * 0.02)
        self.task_names = [self._task_key(name) for name in cfg.task_names]
        if not self.task_names or len(set(self.task_names)) != len(self.task_names):
            raise ValueError("task_names must contain distinct nonempty task names")
        if int(cfg.action_model.num_condition_tokens) != 5:
            raise ValueError("Four body-part state tokens plus one task token require num_condition_tokens=5")
        self.task_embedding = nn.Embedding(len(self.task_names), width)
        nn.init.normal_(self.task_embedding.weight, std=0.02)
        self.action_model = OmegaCrossAttentionActionModel(self.config)
        self.readout_type = str(cfg.readout.type)
        self.visual_readout = None
        if self.readout_type == "state_query_local":
            # Shared adapter/decoder weights and subsequent diffusion noise are seed-matched.
            with torch.random.fork_rng(devices=[]):
                self.visual_readout = StateQueryLocalReadout(
                    width=width,
                    queries_per_view=int(cfg.readout.queries_per_view),
                    local_grid=int(cfg.readout.local_grid),
                    depth=int(cfg.readout.depth),
                    heads=int(cfg.action_model.num_heads),
                )
        elif self.readout_type != "dense":
            raise ValueError(f"Unknown Omega readout: {self.readout_type}")
        self.action_horizon = int(cfg.action_model.action_horizon)
        self.action_dim = int(cfg.action_model.action_dim)
        self.last_visual_info = {}

    def _load_omega(self):
        cfg = self.config.framework.omega
        source = Path(cfg.source_root).expanduser().resolve()
        if not source.is_dir():
            raise FileNotFoundError(source)
        if str(source) not in sys.path:
            sys.path.insert(0, str(source))
        from vggt_omega.models import VGGTOmega

        path = Path(cfg.checkpoint_path).expanduser().resolve()
        hasher = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                hasher.update(block)
        sha = hasher.hexdigest()
        if sha != cfg.checkpoint_sha256:
            raise ValueError("Omega checkpoint SHA256 differs from the configured reference")
        model = VGGTOmega(enable_camera=True, enable_depth=True, enable_alignment=False)
        weights = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
        model.load_state_dict(weights, strict=True)
        logging.getLogger(__name__).info("Loaded official Omega512 with strict=True; SHA256=%s", sha)
        return model

    def train(self, mode=True):
        super().train(mode)
        self.omega.eval()
        count = int(self.config.framework.omega.train_last_blocks)
        for kind in ("frame_blocks", "inter_frame_blocks"):
            if count:
                for block in getattr(self.omega.aggregator, kind)[-count:]:
                    block.train(mode)
        return self

    def state_dict(self, *args, **kwargs):
        result = super().state_dict(*args, **kwargs)
        if self.config.framework.omega.weights_mode == "reference":
            prefix = kwargs.get("prefix", args[1] if len(args) > 1 else "")
            for key in self.reference_keys:
                result.pop(prefix + key, None)
        return result

    def load_state_dict(self, state_dict, strict=True, assign=False):
        supplied = {key for key in state_dict if key.startswith("omega.")}
        if self.config.framework.omega.weights_mode == "embedded" or supplied.intersection(self.reference_keys):
            return super().load_state_dict(state_dict, strict=strict, assign=assign)
        loaded = super().load_state_dict(state_dict, strict=False, assign=assign)
        if set(loaded.missing_keys) != self.reference_keys or loaded.unexpected_keys:
            raise RuntimeError(
                "Incomplete Omega policy/adapted weights: missing "
                f"{set(loaded.missing_keys) - self.reference_keys}; unexpected {loaded.unexpected_keys}"
            )
        return torch.nn.modules.module._IncompatibleKeys([], [])

    def load_pretrained_state_dict(self, state_dict):
        self.load_state_dict(state_dict, strict=True)

    @staticmethod
    def _task_key(value):
        if not isinstance(value, str) or not value.strip():
            raise ValueError("An explicit nonempty task instruction is required")
        return "_".join(value.strip().lower().split())

    def _prepare_images(self, examples):
        from vggt_omega.utils.load_fn import (
            _balanced_target_shape,
            _crop_to_supported_aspect_ratio,
            _pad_images_to_common_size,
        )

        tensors, shapes = [], set()
        for example in examples:
            if len(example["image"]) != 3:
                raise ValueError("Omega requires exactly three simultaneous camera images")
            for image in example["image"]:
                if isinstance(image, torch.Tensor):
                    image = to_pil_image(image.detach().cpu())
                elif not isinstance(image, Image.Image):
                    image = Image.fromarray(np.asarray(image))
                if image.mode == "RGBA":
                    background = Image.new("RGBA", image.size, (255, 255, 255, 255))
                    image = Image.alpha_composite(background, image)
                image = _crop_to_supported_aspect_ratio(image.convert("RGB"))
                width, height = image.size
                target_h, target_w = _balanced_target_shape(height / width, 512, 16)
                tensor = to_tensor(image.resize((target_w, target_h), Image.Resampling.BICUBIC))
                shapes.add(tuple(tensor.shape[-2:]))
                tensors.append(tensor)
        if len(shapes) > 1:
            tensors = _pad_images_to_common_size(tensors, shapes)
        return (
            torch.stack(tensors).reshape(len(examples), 3, 3, *tensors[0].shape[-2:]).to(self.visual_proj.weight.device)
        )

    def _condition(self, examples, apply_dropout=False, state_conditioning=None, visual_conditioning=None):
        state_mode = self.config.framework.state_conditioning if state_conditioning is None else state_conditioning
        if state_mode not in {"full", "none", "decoder_only", "readout_only"}:
            raise ValueError("Invalid state_conditioning route")
        use_visual = self.config.framework.visual_conditioning if visual_conditioning is None else visual_conditioning
        device, dtype = self.visual_proj.weight.device, self.visual_proj.weight.dtype
        state = torch.as_tensor(
            np.stack([np.asarray(row["state"], dtype=np.float32).reshape(-1) for row in examples]),
            device=device,
            dtype=dtype,
        )
        if state.shape != (len(examples), 14) or not torch.isfinite(state).all():
            raise ValueError("Expected a finite normalized 14-D RoboTwin state")
        tasks = [self._task_key(row.get("lang", row.get("instruction"))) for row in examples]
        unknown = set(tasks) - set(self.task_names)
        if unknown:
            raise ValueError(f"Unknown task instruction(s) {unknown}; expected {self.task_names}")
        task = self.task_embedding(torch.tensor([self.task_names.index(name) for name in tasks], device=device))
        keep_state = torch.ones(len(examples), device=device, dtype=dtype)
        if apply_dropout:
            keep_state = (torch.rand(len(examples), device=device) >= self.config.framework.state_dropout_prob).to(dtype)
        decoder_keep = keep_state if state_mode in {"full", "decoder_only"} else torch.zeros_like(keep_state)
        readout_keep = keep_state if state_mode in {"full", "readout_only"} else torch.zeros_like(keep_state)
        parts = torch.stack(
            [projection(state[:, part]) for projection, part in zip(self.state_proj, STATE_PARTS, strict=False)], dim=1
        )
        parts = (parts + self.state_part_embeddings) * decoder_keep[:, None, None] * self.config.framework.state_scale
        condition = torch.cat([parts, task[:, None]], dim=1)
        images = self._prepare_images(examples)
        amp = torch.autocast("cuda", dtype=torch.bfloat16) if device.type == "cuda" else nullcontext()
        gradients = nullcontext() if self.config.framework.omega.train_last_blocks else torch.no_grad()
        with amp, gradients:
            layers, patch_start = self.omega.aggregator(images)
            raw_features = layers[int(self.config.framework.omega.feature_layer)]
            if raw_features is None:
                raise ValueError("Requested Omega feature layer was not cached")
            raw_features = raw_features[:, :, patch_start:]
        grid = (images.shape[-2] // 16, images.shape[-1] // 16)
        if raw_features.shape[2:] != (grid[0] * grid[1], 2048):
            raise ValueError("Omega dense feature dimensions do not match the image patch grid")
        features = self.visual_proj(self.visual_norm(raw_features.to(dtype)))
        y, x = torch.meshgrid(
            torch.linspace(-1, 1, grid[0], device=device, dtype=dtype),
            torch.linspace(-1, 1, grid[1], device=device, dtype=dtype),
            indexing="ij",
        )
        positions = self.position_proj(torch.stack([x, y], dim=-1).reshape(-1, 2))
        features = features + positions[None, None] + self.view_embeddings[None, :, None]
        if self.visual_readout is None:
            memory = features.flatten(1, 2)
        else:
            memory = self.visual_readout(features, state, grid, keep_state=readout_keep)
        if not use_visual:
            # Mask query residuals as well as image features: queries themselves
            # contain state, so blanking RGB alone is not a state-only control.
            memory = memory * 0.0
        if apply_dropout:
            keep = (torch.rand(len(examples), device=device) >= self.config.framework.condition_dropout_prob).to(dtype)
            condition = condition * keep[:, None, None]
            memory = memory * keep[:, None, None]
        self.last_visual_info = dict(
            input_shape=list(images.shape),
            patch_grid=list(grid),
            dense_patch_tokens=int(features.shape[1] * features.shape[2]),
            memory_tokens=int(memory.shape[1]),
            readout=self.readout_type,
        )
        return condition, memory

    def forward(self, examples, **kwargs):
        condition, memory = self._condition(examples, apply_dropout=self.training)
        actions = torch.as_tensor(
            np.stack([row["action"] for row in examples]), device=condition.device, dtype=condition.dtype
        )
        if actions.shape[1:] != (self.action_horizon, self.action_dim):
            raise ValueError(f"Expected action chunks {(self.action_horizon, self.action_dim)}, got {actions.shape}")
        kv = self.action_model.net.prepare_visual(memory)
        prediction, noise, _ = self.action_model(actions, condition, kv)
        return dict(action_loss=self.action_model.loss(prediction, noise))

    @torch.inference_mode()
    def predict_action(self, examples, **kwargs):
        if not isinstance(examples, list):
            examples = [examples]
        steps = int(kwargs.get("num_ddim_steps", 20))
        if not 1 <= steps <= 100:
            raise ValueError("DDIM steps must be between 1 and 100")
        device = self.visual_proj.weight.device
        amp = torch.autocast("cuda", dtype=torch.bfloat16) if device.type == "cuda" else nullcontext()
        with amp:
            condition, memory = self._condition(
                examples,
                state_conditioning=kwargs.get("state_conditioning"),
                visual_conditioning=kwargs.get("visual_conditioning"),
            )
            kv = self.action_model.net.prepare_visual(memory)
            generator = None
            if kwargs.get("diffusion_seed") is not None:
                generator = torch.Generator(device=device).manual_seed(int(kwargs["diffusion_seed"]))
            noise = torch.randn(
                len(examples),
                self.action_horizon,
                self.action_dim,
                device=device,
                dtype=self.visual_proj.weight.dtype,
                generator=generator,
            )
            if self.action_model.ddim_diffusion is None or getattr(self, "_ddim_steps", None) != steps:
                self.action_model.create_ddim(steps)
                self._ddim_steps = steps
            samples = self.action_model.ddim_diffusion.ddim_sample_loop(
                self.action_model.net,
                noise.shape,
                noise,
                clip_denoised=False,
                model_kwargs=dict(z=condition, visual_kv=kv),
                progress=False,
                device=device,
                eta=0.0,
            )
        return dict(normalized_actions=samples.float().cpu().numpy())
