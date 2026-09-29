"""Surflo-conditioned DiT policy for BEHAVIOR action-chunk generation."""

from __future__ import annotations

import importlib
import sys
from contextlib import nullcontext
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision.transforms import functional as VF

from starVLA.model.framework.base_framework import baseframework
from starVLA.model.framework.share_tools import merge_framework_config
from starVLA.model.modules.action_model.DiTActionHeader import get_action_model
from starVLA.model.tools import FRAMEWORK_REGISTRY
from starVLA.training.trainer_utils import initialize_overwatch

logger = initialize_overwatch(__name__)

R1PRO_STATE_PARTS = (
    ("base", slice(0, 3)),
    ("torso", slice(53, 61)),
    ("left_arm", slice(3, 24)),
    ("left_gripper", slice(24, 28)),
    ("right_arm", slice(28, 49)),
    ("right_gripper", slice(49, 53)),
)


@dataclass
class SurfloDiTDefaultConfig:
    name: str = "SurfloDiT"
    surflo: dict = field(default_factory=lambda: {
        "source_root": "/export/home/haoliang/projects/Surflo",
        "checkpoint_path": "/export/home/haoliang/projects/Surflo/checkpoints/surflo_v0.pt",
        "model_config_path": "/export/home/haoliang/projects/Surflo/configs/model/surflo.yaml",
        "use_ema": True,
        "freeze_backbone": True,
        "image_size": 518,
        "num_views": 3,
        "append_camera_token": True,
        "append_per_view_camera_tokens": False,
        "weights_mode": "reference",
    })
    action_model: dict = field(default_factory=lambda: {
        "action_model_type": "DiT-L",
        "action_dim": 23,
        "action_hidden_dim": 1024,
        "action_horizon": 16,
        "state_dim": 61,
        "num_condition_tokens": 129,
    })
    state_scale: float = 1.0
    state_dropout_prob: float = 0.2
    state_conditioning: str = "add"


@FRAMEWORK_REGISTRY.register("SurfloDiT")
class SurfloDiT(baseframework):
    """Use frozen Surflo compressor tokens as conditions for StarVLA's DiT."""

    def __init__(self, config=None, **kwargs) -> None:
        super().__init__()
        self.config = merge_framework_config(SurfloDiTDefaultConfig, config)
        self.surflo_backbone = self._load_surflo()
        self.freeze_surflo = bool(self.config.framework.surflo.freeze_backbone)
        if self.freeze_surflo:
            for parameter in self.surflo_backbone.parameters():
                parameter.requires_grad_(False)
            self.surflo_backbone.eval()

        surface_net = self.surflo_backbone.surface_net
        self.append_camera_token = bool(self.config.framework.surflo.append_camera_token)
        if self.append_camera_token and not surface_net.encode_camera_tokens_separately:
            raise ValueError("The official Surflo config must expose a separate camera token.")
        self.append_per_view_camera_tokens = bool(self.config.framework.surflo.append_per_view_camera_tokens)
        self.state_conditioning = str(self.config.framework.state_conditioning)
        if self.state_conditioning not in {"add", "token", "parts"}:
            raise ValueError(f"Unsupported state_conditioning: {self.state_conditioning}")
        visual_dim = int(surface_net.compressor.latent_tokens.shape[-1])
        hidden_dim = int(self.config.framework.action_model.action_hidden_dim)
        self.visual_proj = nn.Linear(visual_dim, hidden_dim)
        state_dim = int(getattr(self.config.framework.action_model, "state_dim", 0))
        if self.state_conditioning == "parts":
            if state_dim != 61:
                raise ValueError(f"R1Pro body-part tokens require a 61-D state, got {state_dim}.")
            self.state_proj = nn.ModuleDict({
                name: nn.Linear(part.stop - part.start, hidden_dim)
                for name, part in R1PRO_STATE_PARTS
            })
            self.state_part_embeddings = nn.Parameter(torch.empty(len(R1PRO_STATE_PARTS), hidden_dim))
            nn.init.normal_(self.state_part_embeddings, std=0.02)
        else:
            self.state_proj = nn.Linear(state_dim, hidden_dim) if state_dim > 0 else None
        self.action_model = get_action_model(config=self.config)
        self.action_horizon = int(self.config.framework.action_model.action_horizon)
        self.action_dim = int(self.config.framework.action_model.action_dim)
        self.image_size = int(self.config.framework.surflo.image_size)
        self.num_views = int(self.config.framework.surflo.num_views)
        expected_tokens = 128 + int(self.append_camera_token)
        expected_tokens += self.num_views if self.append_per_view_camera_tokens else 0
        expected_tokens += {"add": 0, "token": 1, "parts": len(R1PRO_STATE_PARTS)}[self.state_conditioning]
        configured_tokens = int(self.config.framework.action_model.num_condition_tokens)
        if configured_tokens != expected_tokens:
            raise ValueError(f"num_condition_tokens must be {expected_tokens}, got {configured_tokens}.")

    def _load_surflo(self) -> nn.Module:
        cfg = self.config.framework.surflo
        weights_mode = str(getattr(cfg, "weights_mode", "reference"))
        if weights_mode not in {"reference", "embedded"}:
            raise ValueError(f"Unsupported Surflo weights_mode: {weights_mode}")
        source_root = Path(cfg.source_root).expanduser().resolve()
        if not source_root.is_dir():
            raise FileNotFoundError(f"Surflo source_root does not exist: {source_root}")
        if str(source_root) not in sys.path:
            sys.path.insert(0, str(source_root))
        try:
            from omegaconf import OmegaConf
            load_model = importlib.import_module("surflo.model.loader").load_model
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "SurfloDiT needs Surflo's runtime dependencies in the active Python "
                f"environment. Missing module: {exc.name}"
            ) from exc

        checkpoint_path = Path(cfg.checkpoint_path).expanduser() if weights_mode == "reference" else None
        model_config_path = Path(cfg.model_config_path).expanduser()
        if checkpoint_path is not None and not checkpoint_path.is_file():
            raise FileNotFoundError(f"Surflo checkpoint not found: {checkpoint_path}")
        if not model_config_path.is_file():
            raise FileNotFoundError(f"Surflo model config not found: {model_config_path}")
        model = load_model(
            OmegaConf.load(str(model_config_path)),
            ckpt_path=str(checkpoint_path) if checkpoint_path is not None else None,
            device="cuda" if torch.cuda.is_available() else "cpu",
            use_ema=bool(cfg.use_ema),
        )
        if checkpoint_path is not None:
            logger.info("Loaded Surflo release checkpoint from %s", checkpoint_path)
        return model

    def load_pretrained_state_dict(self, state_dict: dict) -> None:
        backbone_keys = {key for key in self.state_dict() if key.startswith("surflo_backbone.")}
        supplied_backbone = {key for key in state_dict if key.startswith("surflo_backbone.")}
        mode = str(getattr(self.config.framework.surflo, "weights_mode", "reference"))
        if supplied_backbone:
            if mode == "reference" and supplied_backbone != backbone_keys:
                raise RuntimeError("Checkpoint contains only some Surflo backbone weights.")
            self.load_state_dict(state_dict, strict=True)
            return
        if mode != "reference":
            raise RuntimeError("Embedded-backbone checkpoint is missing Surflo weights.")
        incompatible = self.load_state_dict(state_dict, strict=False)
        if set(incompatible.missing_keys) != backbone_keys or incompatible.unexpected_keys:
            raise RuntimeError(
                "Policy checkpoint has missing or unexpected non-backbone keys: "
                f"missing={set(incompatible.missing_keys) - backbone_keys}, "
                f"unexpected={incompatible.unexpected_keys}"
            )

    def _image_tensor(self, image: Image.Image | np.ndarray | torch.Tensor) -> torch.Tensor:
        if isinstance(image, torch.Tensor):
            if image.ndim != 3:
                raise ValueError(f"Expected image tensor [C,H,W], got {tuple(image.shape)}")
            tensor = image.float().div(255.0) if image.dtype == torch.uint8 else image.float()
        else:
            if not isinstance(image, Image.Image):
                image = Image.fromarray(np.asarray(image))
            tensor = VF.to_tensor(image.convert("RGB"))
        return VF.resize(tensor, [self.image_size, self.image_size], antialias=True)

    def _prepare_images(self, examples: List[dict]) -> torch.Tensor:
        groups = [example["image"] for example in examples]
        bad_counts = [len(group) for group in groups if len(group) != self.num_views]
        if bad_counts:
            raise ValueError(
                f"SurfloDiT expects {self.num_views} simultaneous views, got {bad_counts}. "
                "Configure one observation timestep."
            )
        images = torch.stack([torch.stack([self._image_tensor(image) for image in group]) for group in groups])
        return images.to(self.visual_proj.weight.device)

    def _encode_visual(self, examples: List[dict]) -> torch.Tensor:
        images = self._prepare_images(examples)
        grad_context = torch.no_grad() if self.freeze_surflo else nullcontext()
        amp_context = torch.autocast("cuda", dtype=torch.bfloat16) if images.device.type == "cuda" else nullcontext()
        with grad_context, amp_context:
            predictions = self.surflo_backbone.vggt(images, return_aggregated_tokens=True)
            tokens, _ = self.surflo_backbone.surface_net.get_compressed_tokens(
                predictions["aggregated_tokens_list"],
                predictions["patch_start_idx"],
                vggt_world_points=predictions["world_points"],
            )
            if self.append_camera_token:
                camera_tokens = self.surflo_backbone.surface_net.get_compressed_camera_tokens(
                    predictions["aggregated_tokens_list"], predictions["patch_start_idx"]
                )
                tokens = torch.cat([tokens, camera_tokens], dim=1)
            if self.append_per_view_camera_tokens:
                per_view_camera_tokens = self.surflo_backbone.surface_net._build_camera_tokens(
                    predictions["aggregated_tokens_list"], predictions["patch_start_idx"]
                )
                if per_view_camera_tokens.shape[1] != self.num_views:
                    raise ValueError("SurfLo returned an unexpected number of per-view camera tokens.")
                tokens = torch.cat([tokens, per_view_camera_tokens], dim=1)
        return self.visual_proj(tokens.to(dtype=self.visual_proj.weight.dtype))

    def _encode_state(self, states: List[np.ndarray | torch.Tensor | None]) -> torch.Tensor | None:
        if self.state_proj is None or any(state is None for state in states):
            return None
        values = np.stack([np.asarray(state, dtype=np.float32).reshape(-1) for state in states])
        expected_dim = 61 if self.state_conditioning == "parts" else self.state_proj.in_features
        if values.shape[-1] != expected_dim:
            raise ValueError(f"State has dimension {values.shape[-1]}, expected {expected_dim}.")
        tensor = torch.as_tensor(values, device=self.visual_proj.weight.device, dtype=self.visual_proj.weight.dtype)
        if self.state_conditioning == "parts":
            part_tokens = torch.stack([
                self.state_proj[name](tensor[:, part])
                for name, part in R1PRO_STATE_PARTS
            ], dim=1)
            return part_tokens + self.state_part_embeddings.unsqueeze(0)
        return self.state_proj(tensor)

    def _apply_state_dropout(self, state_features: torch.Tensor, *, enabled: bool) -> torch.Tensor:
        """Drop complete per-example state conditions without perturbing visual tokens."""
        probability = float(getattr(self.config.framework, "state_dropout_prob", 0.0))
        if not 0.0 <= probability <= 1.0:
            raise ValueError(f"state_dropout_prob must be in [0, 1], got {probability}.")
        if not enabled or probability == 0.0:
            return state_features
        keep_state = torch.rand(state_features.shape[0], device=state_features.device) >= probability
        keep_shape = (state_features.shape[0],) + (1,) * (state_features.ndim - 1)
        return state_features * keep_state.to(dtype=state_features.dtype).reshape(keep_shape)

    def _condition(self, examples: List[dict], *, apply_state_dropout: bool = False) -> torch.Tensor:
        condition = self._encode_visual(examples)
        state_features = self._encode_state([example.get("state") for example in examples])
        if self.state_conditioning in {"token", "parts"} and state_features is None:
            raise ValueError("State-token conditioning requires a state for every example.")
        if state_features is not None:
            state_features = self._apply_state_dropout(state_features, enabled=apply_state_dropout)
            if self.state_conditioning != "parts":
                state_features = state_features.unsqueeze(1)
            state_features = float(self.config.framework.state_scale) * state_features
            if self.state_conditioning in {"token", "parts"}:
                condition = torch.cat([condition, state_features], dim=1)
            else:
                condition = condition + state_features
        return condition

    def forward(self, examples: List[dict], **kwargs) -> dict:
        condition = self._condition(examples, apply_state_dropout=self.training)
        values = np.stack([np.asarray(example["action"], dtype=np.float32) for example in examples])
        actions = torch.as_tensor(values, device=condition.device, dtype=condition.dtype)
        if actions.shape[-1] != self.action_dim:
            raise ValueError(f"Action has dimension {actions.shape[-1]}, expected {self.action_dim}.")
        if actions.shape[1] < self.action_horizon:
            raise ValueError(f"Action sequence is shorter than action_horizon={self.action_horizon}.")
        noise_pred, noise, _ = self.action_model(actions[:, -self.action_horizon :], condition)
        return {"action_loss": self.action_model.loss(noise_pred, noise)}

    @torch.inference_mode()
    def predict_action(self, examples: List[dict], **kwargs) -> dict:
        if not isinstance(examples, list):
            examples = [examples]
        condition = self._condition(examples, apply_state_dropout=False)
        model_dtype = next(self.action_model.net.parameters()).dtype
        condition = condition.to(dtype=model_dtype)
        noise = torch.randn(condition.shape[0], self.action_horizon, self.action_dim, device=condition.device, dtype=model_dtype)
        if self.action_model.ddim_diffusion is None:
            self.action_model.create_ddim(ddim_step=10)

        def model_fn(x, t, **model_kwargs):
            return self.action_model.net.forward(x.to(dtype=model_dtype), t, **model_kwargs)

        samples = self.action_model.ddim_diffusion.ddim_sample_loop(
            model_fn, noise.shape, noise, clip_denoised=False, model_kwargs={"z": condition},
            progress=False, device=condition.device, eta=0.0,
        )
        return {"normalized_actions": samples.float().cpu().numpy()}
