"""
VGGTOmega-OFT Framework

Uses VGGT-Omega as the visual backbone for multi-view robot action prediction.
- VGGT-Omega Aggregator: encodes T camera views via alternating-attention transformer
- CLIP text encoder: encodes language instructions (frozen)
- Linear projections: fuse visual and text features into a shared hidden space
- DiT diffusion action head: DDIM-based action chunk generation (replaces L1 MLP)
"""

import os
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms as TF
from transformers import CLIPTextModel, CLIPTokenizer

from starVLA.model.framework.base_framework import baseframework
from starVLA.model.framework.share_tools import merge_framework_config
from starVLA.model.modules.action_model.DiTActionHeader import get_action_model as get_dit_action_model
from starVLA.model.modules.projector.memory_pool import (
    GatedMemoryPool,
    GatedRecurrentMemoryPool,
    HierarchicalGatedMemoryPool,
    LegacyCrossAttnPool,
    ResidualCrossAttnPool,
    StreamingMemoryPool,
    VisualMemoryPool,
)
from starVLA.model.tools import FRAMEWORK_REGISTRY
from starVLA.training.trainer_utils import initialize_overwatch

logger = initialize_overwatch(__name__)

IGNORE_INDEX = -100


@dataclass
class VGGTOmegaOFTDefaultConfig:
    name: str = "VGGTOmegaOFT"

    # VGGT-Omega visual backbone and temporal-memory configuration.
    visual_backbone: dict = field(
        default_factory=lambda: {
            "type": "vggt_omega",
            "freeze_backbone": True,
            # "cross_self_attn" (new): cross-attention pooling + self-attention
            # refinement, letting pooled memory tokens reconcile info across
            # views. "cross_only": original plain single-shot cross-attention
            # pool, kept selectable for A/B comparison. "streaming": recurrent
            # pooling over a bounded sliding window of timesteps (see
            # StreamingMemoryPool in memory_pool.py) — pooled memory tokens
            # persist across timesteps instead of being recomputed from a flat,
            # non-causal attention over the whole history window every call.
            # "residual_cross_attn": cross-attention + residual + final norm,
            # no self-attention (ResidualCrossAttnPool) — step 1 of staged
            # re-introduction of cross_self_attn's self-attention after it
            # was found to grow the residual stream ~3x in magnitude with no
            # final norm and fail completely in closed-loop rollout.
            # "gated_self_attn": residual_cross_attn + one LayerScale-gated
            # self-attention layer (GatedMemoryPool) — step 2. Beat cross_self_attn
            # AND the cross_only baseline in closed-loop rollout (45.5% vs
            # 28.9%/0%) — the validated default going forward.
            # "gated_recurrent": GatedRecurrentMemoryPool — same cross-attn/
            # self-attn/LayerScale/final-norm as gated_self_attn, but the
            # memory update is a gated convex combination
            # (memory = (1-z)*memory + z*candidate) applied once per timestep
            # instead of one flat pass over the whole window — bounded by
            # construction, unlike StreamingMemoryPool's unbounded residual
            # recurrence. Step 2b of the streaming-memory plan: trains against
            # the existing fixed window (no data pipeline change) to test
            # whether recurrence itself, now properly stabilized, helps/hurts/
            # is neutral vs. flat gated_self_attn, before committing to real
            # incremental deployment (client/backbone changes).
            "pool_type": "cross_self_attn",
            # Used when pool_type in {"cross_self_attn", "streaming"}.
            "pool_self_attn_layers": 2,
            # Used when pool_type in {"streaming", "gated_recurrent"}. Must
            # equal the number of history timesteps the paired dataset config
            # produces (i.e. len(video_observation_indices) in data_config.py
            # — e.g. 3 for [0, -8, -16]), since backbone tokens are split into
            # that many equal timestep chunks before being fed through the
            # recurrence.
            "pool_window_steps": 3,
            # "hierarchical_gated": HierarchicalGatedMemoryPool — two-level
            # memory. Each chunk of pool_chunk_timesteps consecutive
            # timesteps (e.g. 8, matching execute_horizon) gets its own joint
            # VGGT-Omega backbone pass, then a GatedMemoryPool-style flat
            # pool compresses it to a chunk summary; pool_num_chunks such
            # summaries fold into a persistent global memory via
            # GatedRecurrentMemoryPool's bounded gated update. Unlike every
            # other pool_type, this is NOT capped at a fixed 16-real-step
            # lookback — offline-validated stable over 25 chunks (~200
            # frames). The paired dataset config must produce
            # pool_num_chunks * pool_chunk_timesteps consecutive timestep
            # offsets (e.g. [0,-1,...,-23] for 3 chunks of 8).
            "pool_num_chunks": 3,
            "pool_chunk_timesteps": 8,
        }
    )

    vggt_omega: dict = field(
        default_factory=lambda: {
            "checkpoint_path": "./playground/Pretrained_models/vggt_omega_1b_256_text.pt",
            "freeze_backbone": True,
        }
    )

    clip_text: dict = field(
        default_factory=lambda: {
            "model_name": "openai/clip-vit-base-patch32",
            "freeze": True,
        }
    )

    action_model: dict = field(
        default_factory=lambda: {
            "action_model_type": "DiT-L",
            "action_dim": 14,
            "action_hidden_dim": 1024,
            "action_horizon": 16,
            "state_dim": 14,  # proprioceptive state dim; set 0 to disable
        }
    )


@FRAMEWORK_REGISTRY.register("VGGTOmegaOFT")
class VGGTOmega_OFT(baseframework):
    """
    Multi-view VLA framework using VGGT-Omega as visual backbone.

    Data flow (training):
      example["image"]  = [cam_high, cam_left_wrist, cam_right_wrist]  List[PIL.Image]
                           (time-major: current-timestep cameras first, then
                           older timesteps, if the data config buffers history)
      example["lang"]   = str  task instruction
      example["action"] = np.ndarray (action_horizon, 14)  abs_qpos targets

    Forward:
      1. Stack T PIL images (T = num_cameras * num_history_steps) → (B, T, 3, H, W) tensor
      2. VGGT-Omega Aggregator → aggregated_tokens (B, T, N_tokens, 2048)
      3. Cross-attention pooling over all tokens → visual feat (B, 2048)
      4. CLIP text encoder → text feat (B, 512)
      5. Project both to hidden_dim, add → fused (B, hidden_dim)
      6. Expand fused to (B, n_cond_tokens, hidden_dim) — DiT condition
      7. DiT diffusion head: noise prediction MSE loss during training,
         DDIM sampling (10 steps) during inference
    """

    def __init__(self, config=None, **kwargs) -> None:
        super().__init__()
        self.config = merge_framework_config(VGGTOmegaOFTDefaultConfig, config)

        # ── VGGT-Omega visual backbone ─────────────────────────────────────
        self.visual_backbone_type = self.config.framework.visual_backbone.type
        self.freeze_visual_backbone = bool(
            getattr(self.config.framework.visual_backbone, "freeze_backbone", True)
        )
        visual_dim = self._build_visual_backbone()
        self.patch_token_start = 17  # 1 camera token + 16 register tokens (VGGT only)

        # ── CLIP text encoder ─────────────────────────────────────────────
        clip_name = self.config.framework.clip_text.model_name
        clip_local_only = bool(
            getattr(self.config.framework.clip_text, "local_files_only", False)
        )
        self.clip_tokenizer = CLIPTokenizer.from_pretrained(
            clip_name, local_files_only=clip_local_only
        )
        self.clip_text_model = CLIPTextModel.from_pretrained(
            clip_name, local_files_only=clip_local_only
        )
        if self.config.framework.clip_text.freeze:
            for p in self.clip_text_model.parameters():
                p.requires_grad_(False)
        text_dim = self.clip_text_model.config.hidden_size  # 512 for clip-vit-base-patch32

        # ── Visual pooling: 64 learned memory tokens cross-attend once over all
        # T×N_tokens backbone tokens, then self-attend among themselves so
        # pooled tokens can reconcile information gathered across views,
        # producing 64 diverse condition tokens for the DiT head.
        hidden_dim = int(self.config.framework.action_model.action_hidden_dim)
        pool_type = getattr(self.config.framework.visual_backbone, "pool_type", "cross_self_attn")
        if pool_type == "cross_only":
            self.visual_pool = LegacyCrossAttnPool(dim=visual_dim, num_query_tokens=64, num_heads=8)
        elif pool_type == "residual_cross_attn":
            self.visual_pool = ResidualCrossAttnPool(dim=visual_dim, num_query_tokens=64, num_heads=8)
        elif pool_type == "gated_self_attn":
            self.visual_pool = GatedMemoryPool(dim=visual_dim, num_query_tokens=64, num_heads=8)
        elif pool_type == "streaming":
            self.visual_pool = StreamingMemoryPool(
                dim=visual_dim,
                num_query_tokens=64,
                num_heads=8,
                num_self_attn_layers=int(
                    getattr(self.config.framework.visual_backbone, "pool_self_attn_layers", 2)
                ),
                window_steps=int(
                    getattr(self.config.framework.visual_backbone, "pool_window_steps", 3)
                ),
            )
        elif pool_type == "gated_recurrent":
            self.visual_pool = GatedRecurrentMemoryPool(
                dim=visual_dim,
                num_query_tokens=64,
                num_heads=8,
                window_steps=int(
                    getattr(self.config.framework.visual_backbone, "pool_window_steps", 3)
                ),
            )
        elif pool_type == "hierarchical_gated":
            self.visual_pool = HierarchicalGatedMemoryPool(
                dim=visual_dim,
                num_query_tokens=64,
                num_heads=8,
                num_chunks=int(
                    getattr(self.config.framework.visual_backbone, "pool_num_chunks", 3)
                ),
            )
            self.pool_chunk_timesteps = int(
                getattr(self.config.framework.visual_backbone, "pool_chunk_timesteps", 8)
            )
        else:
            self.visual_pool = VisualMemoryPool(
                dim=visual_dim,
                num_query_tokens=64,
                num_heads=8,
                num_self_attn_layers=int(
                    getattr(self.config.framework.visual_backbone, "pool_self_attn_layers", 2)
                ),
            )
        self.visual_proj = nn.Linear(visual_dim, hidden_dim)
        self.text_proj = nn.Linear(text_dim, hidden_dim)

        # ── Proprioception projection (optional) ──────────────────────────
        state_dim = int(getattr(self.config.framework.action_model, "state_dim", 14))
        self.state_proj = nn.Linear(state_dim, hidden_dim) if state_dim > 0 else None

        # ── DiT diffusion action head ─────────────────────────────────────
        self.action_model = get_dit_action_model(config=self.config)
        self.action_horizon = int(self.config.framework.action_model.action_horizon)
        self.action_dim = int(self.config.framework.action_model.action_dim)

        self.to_tensor = TF.ToTensor()

    # ──────────────────────────────────────────────────────────────────────
    # Construction helpers
    # ──────────────────────────────────────────────────────────────────────

    def _build_visual_backbone(self) -> int:
        """
        Build the VGGT-Omega visual backbone and return its token dimension.

        Returns:
            visual_dim (int): per-token feature dimension of the backbone.
        """
        btype = self.visual_backbone_type

        if btype != "vggt_omega":
            raise ValueError(
                f"VGGTOmegaOFT only supports visual_backbone.type='vggt_omega', got {btype!r}"
            )

        self.vggt_omega = self._build_vggt_omega()
        # Aggregator output dim: 2*embed_dim (frame + inter-frame tokens concat)
        return 2048

    def _build_vggt_omega(self) -> nn.Module:
        """Instantiate and optionally freeze VGGT-Omega Aggregator."""
        from vggt_omega.models import VGGTOmega

        model = VGGTOmega(
            patch_size=16,
            embed_dim=1024,
            enable_camera=False,
            enable_depth=False,
            enable_alignment=False,
        )

        ckpt_path = self.config.framework.vggt_omega.checkpoint_path
        if ckpt_path:
            import os
            if os.path.isfile(ckpt_path):
                state_dict = torch.load(ckpt_path, map_location="cpu")
                # Checkpoint includes camera/dense/text-alignment head weights we don't use
                missing, unexpected = model.load_state_dict(state_dict, strict=False)
                if missing:
                    logger.warning(f"VGGTOmega: {len(missing)} missing keys (first 5: {missing[:5]})")
                logger.info(f"Loaded VGGT-Omega from {ckpt_path}")
            else:
                logger.warning(f"VGGT-Omega checkpoint not found at {ckpt_path}, using random weights.")

        if self.config.framework.vggt_omega.freeze_backbone:
            for p in model.parameters():
                p.requires_grad_(False)
            logger.info("VGGT-Omega backbone frozen.")

        return model

    # ──────────────────────────────────────────────────────────────────────
    # Encoding helpers
    # ──────────────────────────────────────────────────────────────────────

    def _pil_to_tensor_stack(self, pil_images: List[Image.Image]) -> torch.Tensor:
        """Convert a list of PIL images (T views) to (T, 3, H, W) float tensor [0,1]."""
        return torch.stack([self.to_tensor(img.convert("RGB")) for img in pil_images])

    def _encode_visual_tokens(self, images: torch.Tensor) -> torch.Tensor:
        """
        Run the frozen visual backbone and return a flat token sequence.

        Args:
            images: (B, T, 3, H, W) float tensor in [0, 1]

        Returns:
            (B, T*N_tokens, visual_dim) backbone token features
        """
        B, T = images.shape[:2]
        amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        no_grad_ctx = torch.no_grad() if self.freeze_visual_backbone else torch.enable_grad()

        # VGGT-Omega returns cached outputs at layers 4, 11, 17, and 23.
        # Concatenating them exposes both geometric and semantic features to
        # the learned memory queries.
        with no_grad_ctx, torch.autocast("cuda", dtype=amp_dtype):
            aggregated_tokens_list, _ = self.vggt_omega.aggregator(images)
        layers = [layer for layer in aggregated_tokens_list if layer is not None]
        tokens = torch.cat(layers, dim=2)                  # (B, T, 4*N_tokens, 2048)
        return tokens.reshape(B, -1, tokens.shape[-1])     # (B, T*4*N_tokens, 2048)

    def _pool_visual(self, kv: torch.Tensor) -> torch.Tensor:
        """
        Cross-attention + self-attention pooling: 64 learned memory tokens
        cross-attend over every backbone token (across views/structural
        roles), then self-attend among themselves, producing 64 diverse
        condition tokens for the DiT head.

        Args:
            kv: (B, N, visual_dim) backbone token features (live or cached)

        Returns:
            (B, 64, hidden_dim) projected visual condition tokens
        """
        kv = kv.to(self.visual_proj.weight.device)

        if isinstance(self.visual_pool, (StreamingMemoryPool, GatedRecurrentMemoryPool)):
            num_steps = self.visual_pool.window_steps
            n_tok = kv.shape[1]
            if n_tok % num_steps != 0:
                raise ValueError(
                    f"visual tokens ({n_tok}) not evenly divisible by "
                    f"pool_window_steps ({num_steps}); pool_window_steps must "
                    f"equal the number of history timesteps the dataset config "
                    f"produces (len(video_observation_indices))."
                )
            chunk = n_tok // num_steps
            kv_steps = [kv[:, i * chunk:(i + 1) * chunk] for i in range(num_steps)]
            pooled = self.visual_pool.forward_sequence(kv_steps)  # (B, 64, visual_dim)
        else:
            pooled = self.visual_pool(kv)  # (B, 64, visual_dim)

        proj_dtype = self.visual_proj.weight.dtype
        return self.visual_proj(pooled.to(dtype=proj_dtype))  # (B, 64, hidden_dim)

    def _encode_visual(self, batch_images: List[List[Image.Image]]) -> torch.Tensor:
        """Run the live backbone over images, then pool. (B, hidden_dim)."""
        device = self.visual_proj.weight.device
        images = torch.stack([self._pil_to_tensor_stack(imgs) for imgs in batch_images]).to(device)

        if isinstance(self.visual_pool, HierarchicalGatedMemoryPool):
            return self._encode_visual_hierarchical(images)

        kv = self._encode_visual_tokens(images)              # (B, N, visual_dim)
        return self._pool_visual(kv)

    def _encode_visual_hierarchical(self, images: torch.Tensor) -> torch.Tensor:
        """
        For HierarchicalGatedMemoryPool: unlike every other pool_type, a
        single joint backbone pass over the whole window is wrong here — each
        chunk needs its own separate VGGT-Omega call, so cross-frame
        attention happens within a chunk's timesteps only, never across
        chunks (that reconciliation is the global memory's job, not the
        backbone's). images is time-major with index 0 = current timestep
        (see _pack_sample / model2robotwin_interface.py), so contiguous
        blocks along dim=1 are contiguous timestep ranges; chunk 0 (first
        block) is therefore the most-recent chunk.

        Args:
            images: (B, T_total, 3, H, W), T_total = pool_num_chunks *
                pool_chunk_timesteps * num_views.
        Returns:
            (B, 64, hidden_dim) projected global-memory condition tokens.
        """
        num_chunks = self.visual_pool.num_chunks
        T_total = images.shape[1]
        if T_total % num_chunks != 0:
            raise ValueError(
                f"visual images ({T_total}) not evenly divisible by pool_num_chunks "
                f"({num_chunks}); the dataset config's video_observation_indices must "
                f"produce pool_num_chunks * pool_chunk_timesteps consecutive timestep "
                f"offsets (e.g. [0,-1,...,-23] for 3 chunks of 8)."
            )
        images_per_chunk = T_total // num_chunks
        chunk_kvs = [
            self._encode_visual_tokens(images[:, c * images_per_chunk:(c + 1) * images_per_chunk])
            for c in range(num_chunks)
        ]  # each (B, images_per_chunk-worth-of-tokens, visual_dim); index 0 = most recent chunk
        pooled = self.visual_pool.forward_sequence(chunk_kvs)  # (B, 64, visual_dim)

        proj_dtype = self.visual_proj.weight.dtype
        return self.visual_proj(pooled.to(dtype=proj_dtype))

    def _visual_features(self, examples: List[dict]) -> torch.Tensor:
        """
        Produce pooled visual features for a batch, using precomputed token
        caches when present (skips the frozen backbone forward) and otherwise
        running the live backbone over the images.
        """
        if examples[0].get("visual_tokens", None) is not None:
            if isinstance(self.visual_pool, HierarchicalGatedMemoryPool):
                raise NotImplementedError(
                    "HierarchicalGatedMemoryPool requires per-chunk separate backbone "
                    "passes; a precomputed visual_tokens cache (built as one joint call) "
                    "can't be reused correctly. Disable visual_feature_cache for this pool_type."
                )
            kv = torch.stack([
                torch.as_tensor(np.asarray(e["visual_tokens"])) for e in examples
            ])                                               # (B, N, visual_dim)
            return self._pool_visual(kv)
        return self._encode_visual([e["image"] for e in examples])

    def _encode_text(self, instructions: List[str]) -> torch.Tensor:
        """
        Encode instructions with frozen CLIP text encoder.

        Args:
            instructions: List[B] of str

        Returns:
            projected text features (B, hidden_dim)
        """
        device = self.text_proj.weight.device

        tokens = self.clip_tokenizer(
            instructions,
            padding=True,
            truncation=True,
            max_length=77,
            return_tensors="pt",
        ).to(device)

        with torch.no_grad() if self.config.framework.clip_text.freeze else torch.enable_grad():
            text_outputs = self.clip_text_model(**tokens)

        # pooler_output: (B, text_dim) — EOS token projected to CLIP embedding space
        proj_dtype = self.text_proj.weight.dtype
        return self.text_proj(text_outputs.pooler_output.to(dtype=proj_dtype))  # (B, hidden_dim)

    def _encode_state(self, states: List[np.ndarray]) -> torch.Tensor:
        """
        Project proprioceptive joint state to hidden_dim.

        Args:
            states: List[B] of np.ndarray shaped (1, state_dim) — current qpos

        Returns:
            (B, hidden_dim) tensor
        """
        device = self.state_proj.weight.device
        proj_dtype = self.state_proj.weight.dtype
        state_tensor = torch.tensor(
            np.stack([s[0] for s in states], axis=0),  # (B, state_dim)
            device=device,
            dtype=proj_dtype,
        )
        return self.state_proj(state_tensor)  # (B, hidden_dim)

    # ──────────────────────────────────────────────────────────────────────
    # Training forward
    # ──────────────────────────────────────────────────────────────────────

    def forward(self, examples: List[dict], external_memory: torch.Tensor = None, **kwargs) -> dict:
        """
        Training forward pass.

        Each example must contain:
            image:  List[PIL.Image]  — camera views, optionally x history steps
            lang:   str              — task instruction
            action: np.ndarray       — (action_horizon, 14) abs_qpos targets

        external_memory dispatches to the Option B streaming path (see
        _streaming_forward) instead of the standard i.i.d. path below.
        Callers under DistributedDataParallel MUST go through this method
        (i.e. call `self.model(examples, external_memory=...)`, not
        `raw_model._streaming_forward(...)` on the unwrapped module) — DDP's
        gradient-sync hooks are only armed by its own forward()/__call__;
        invoking a method directly on `.module` silently skips
        synchronization and each rank trains an independent, unsynced
        replica. Confirmed empirically (a minimal 2-process DDP test: calling
        a custom method directly on `.module` produced different post-
        backward gradients per rank; calling through `model(x)` produced
        identical, correctly-averaged gradients) after both `streaming_step`
        training runs on stack_blocks_three ran this way — each one actually
        trained two diverging, non-communicating replicas per step, with
        only rank 0's ever saved.
        """
        if external_memory is not None:
            return self._streaming_forward(examples, external_memory)

        instructions = [example["lang"] for example in examples]
        actions = [example["action"] for example in examples]
        states = [example.get("state", None) for example in examples]

        visual_feats = self._visual_features(examples)     # (B, 64, hidden_dim)
        text_feats = self._encode_text(instructions)        # (B, hidden_dim)

        # Broadcast text/state as a global offset across all 64 visual tokens
        condition = visual_feats + text_feats.unsqueeze(1)  # (B, 64, hidden_dim)

        # state_scale controls state contribution magnitude:
        #   0.0 = disabled (visual-only), 0.1 = suppressed, 1.0 = full
        state_scale = float(getattr(self.config.framework, "state_scale", 1.0))
        if state_scale > 0.0 and self.state_proj is not None and states[0] is not None:
            condition = condition + state_scale * self._encode_state(states).unsqueeze(1)

        # Ground truth actions
        actions_tensor = torch.tensor(
            np.array(actions), device=condition.device, dtype=condition.dtype
        )  # (B, T_full, 14)
        gt_action = actions_tensor[:, -self.action_horizon:, :]  # (B, L, 14)

        # DiT training step: add noise, predict noise, MSE loss
        noise_pred, noise, _ = self.action_model(gt_action, condition)
        action_loss = self.action_model.loss(noise_pred, noise)

        return {"action_loss": action_loss}

    def _streaming_forward(self, examples: List[dict], external_memory: torch.Tensor) -> Tuple[dict, torch.Tensor]:
        """
        Option B / true stateful streaming training: process exactly ONE new
        chunk per example (see episode_chunk_stream.EpisodeChunkStream),
        using external_memory as this call's starting global-memory state.
        The caller (a custom trainer, not the standard VLATrainer._train_step)
        owns the persistent per-slot memory across calls — this method is a
        pure function of (examples, external_memory) w.r.t. memory
        bookkeeping (it sets self.visual_pool.global_pool._memory only as an
        implementation detail of reusing GatedRecurrentMemoryPool.step();
        that attribute is not read before being overwritten here).

        Dispatched from forward() (never call this directly on a DDP-wrapped
        model's `.module` — see forward()'s docstring for why that silently
        breaks gradient sync under multi-GPU training).

        Args:
            examples: list[B] of packed samples, each "image" being exactly
                one chunk's worth of frames (chunk_timesteps x num_views).
            external_memory: (B, 64, visual_dim) — carried-over memory for
                continuing slots, or self.visual_pool.global_pool.query
                (expanded) for slots whose episode just reset. Caller builds
                this by combining stored per-slot memory with reset_mask.
        Returns:
            (loss_dict, new_memory): new_memory is graph-connected (NOT
            detached) — caller detaches after backward, before storing it
            for the next call.
        """
        assert isinstance(self.visual_pool, HierarchicalGatedMemoryPool), (
            "streaming_step requires pool_type: hierarchical_gated"
        )
        instructions = [example["lang"] for example in examples]
        actions = [example["action"] for example in examples]
        states = [example.get("state", None) for example in examples]

        device = self.visual_proj.weight.device
        images = torch.stack([self._pil_to_tensor_stack(e["image"]) for e in examples]).to(device)
        kv = self._encode_visual_tokens(images)                       # one backbone call, (B, chunk_tokens, visual_dim)
        chunk_summary = self.visual_pool.chunk_pool(kv)                # (B, 64, visual_dim)

        self.visual_pool.global_pool._memory = external_memory.to(device)
        new_memory = self.visual_pool.global_pool.step(chunk_summary)  # (B, 64, visual_dim), graph-connected

        proj_dtype = self.visual_proj.weight.dtype
        visual_feats = self.visual_proj(new_memory.to(dtype=proj_dtype))  # (B, 64, hidden_dim)
        text_feats = self._encode_text(instructions)

        state_scale = float(getattr(self.config.framework, "state_scale", 1.0))
        use_state = state_scale > 0.0 and self.state_proj is not None and states[0] is not None
        state_feats = self._encode_state(states) if use_state else None

        condition = visual_feats + text_feats.unsqueeze(1)
        if use_state:
            condition = condition + state_scale * state_feats.unsqueeze(1)

        actions_tensor = torch.tensor(np.array(actions), device=condition.device, dtype=condition.dtype)
        gt_action = actions_tensor[:, -self.action_horizon:, :]

        noise_pred, noise, _ = self.action_model(gt_action, condition)
        action_loss = self.action_model.loss(noise_pred, noise)

        loss_dict = {"action_loss": action_loss}

        # Auxiliary chunk-level action loss: predicts this chunk's own
        # ground-truth action directly from chunk_summary, bypassing
        # global_pool/the recurrence entirely. Counters a diagnosed failure
        # mode where chunk_pool collapses to a near content-independent
        # output (measured: >0.99 cosine similarity of chunk_summary across
        # genuinely different real scenes, despite the raw backbone tokens
        # differing substantially) — the recurrent gate's gradient back into
        # chunk_pool is too weak on its own to keep it discriminative. A
        # collapsed chunk_summary can only reach the loss floor of a
        # content-blind predictor here (no memory to lean on as a crutch),
        # so this term directly penalizes the collapse every single step,
        # independent of however global_pool ends up using the result.
        chunk_aux_loss_weight = float(getattr(self.config.framework, "chunk_aux_loss_weight", 0.0))
        if chunk_aux_loss_weight > 0.0:
            aux_visual_feats = self.visual_proj(chunk_summary.to(dtype=proj_dtype))
            aux_condition = aux_visual_feats + text_feats.unsqueeze(1)
            if use_state:
                aux_condition = aux_condition + state_scale * state_feats.unsqueeze(1)

            aux_noise_pred, aux_noise, _ = self.action_model(gt_action, aux_condition)
            aux_loss = self.action_model.loss(aux_noise_pred, aux_noise)

            loss_dict["action_loss_main"] = action_loss.detach()
            loss_dict["action_loss_aux"] = aux_loss.detach()
            loss_dict["action_loss"] = action_loss + chunk_aux_loss_weight * aux_loss

        return loss_dict, new_memory

    # ──────────────────────────────────────────────────────────────────────
    # Inference
    # ──────────────────────────────────────────────────────────────────────

    @torch.inference_mode()
    def predict_action(self, examples: List[dict], **kwargs) -> dict:
        """
        Inference: DDIM diffusion sampling to produce normalized action chunk.

        Returns:
            dict with "normalized_actions": np.ndarray (B, chunk_len, 14)
        """
        if not isinstance(examples, list):
            examples = [examples]

        # Normalize images to PIL in-place (eval has no cache → live backbone).
        for example in examples:
            example["image"] = [
                img if isinstance(img, Image.Image) else Image.fromarray(img)
                for img in example["image"]
            ]

        instructions = [example["lang"] for example in examples]
        states = [example.get("state", None) for example in examples]

        visual_feats = self._visual_features(examples)      # (B, 64, hidden_dim)
        text_feats = self._encode_text(instructions)         # (B, hidden_dim)

        condition = visual_feats + text_feats.unsqueeze(1)  # (B, 64, hidden_dim)

        state_scale = float(getattr(self.config.framework, "state_scale", 1.0))
        if state_scale > 0.0 and self.state_proj is not None and states[0] is not None:
            condition = condition + state_scale * self._encode_state(states).unsqueeze(1)

        # Cast to model dtype (DiT net parameters dtype)
        model_dtype = next(self.action_model.net.parameters()).dtype
        condition = condition.to(dtype=model_dtype)

        # Initial noise for DDIM denoising
        B = condition.shape[0]
        noise = torch.randn(
            B, self.action_horizon, self.action_dim,
            device=condition.device, dtype=model_dtype,
        )

        # Lazily build the DDIM sampler on first inference call
        if self.action_model.ddim_diffusion is None:
            self.action_model.create_ddim(ddim_step=10)

        # The DDIM loop uses .float() internally for schedule coefficients, which
        # promotes intermediate x to float32. Wrap forward to re-cast x each step.
        def _model_fn(x, t, **kwargs):
            return self.action_model.net.forward(x.to(dtype=model_dtype), t, **kwargs)

        samples = self.action_model.ddim_diffusion.ddim_sample_loop(
            _model_fn,
            noise.shape,
            noise,
            clip_denoised=False,
            model_kwargs=dict(z=condition),
            progress=False,
            device=condition.device,
            eta=0.0,
        )  # (B, action_horizon, action_dim)

        vals_all = samples[0, 0, :].float().tolist()
        logger.info(
            "[action-debug] samples step0: %s",
            " ".join(f"a{i}={value:+.3f}" for i, value in enumerate(vals_all)),
        )

        return {"normalized_actions": samples.float().cpu().numpy()}

    @torch.inference_mode()
    def predict_action_streaming(self, examples: List[dict], external_memory=None, **kwargs) -> dict:
        """
        Streaming counterpart to predict_action: samples an action chunk from
        ONE new chunk_timesteps-worth of frames plus carried-over global
        memory, instead of re-encoding a whole fixed window from scratch
        every call. Mirrors streaming_step's memory update (chunk_pool then
        global_pool.step) but runs DDIM sampling instead of computing a loss,
        and needs no ground-truth actions.

        Args:
            examples: list[B] of samples, each "image" = exactly one
                chunk's worth of frames (chunk_timesteps x num_views).
            external_memory: (B, 64, visual_dim) array/tensor carried from
                the previous call, or None to start from the model's learned
                initial query (episode reset) — caller (e.g. ModelClient)
                owns reset bookkeeping and simply passes None on a new
                episode's first chunk.

        Returns:
            dict with "normalized_actions": np.ndarray (B, chunk_len, 14)
            and "memory": np.ndarray (B, 64, visual_dim) — the caller must
            carry this into the next chunk's external_memory.
        """
        assert isinstance(self.visual_pool, HierarchicalGatedMemoryPool), (
            "predict_action_streaming requires pool_type: hierarchical_gated"
        )
        if not isinstance(examples, list):
            examples = [examples]

        for example in examples:
            example["image"] = [
                img if isinstance(img, Image.Image) else Image.fromarray(img)
                for img in example["image"]
            ]

        instructions = [example["lang"] for example in examples]
        states = [example.get("state", None) for example in examples]

        device = self.visual_proj.weight.device
        images = torch.stack([self._pil_to_tensor_stack(e["image"]) for e in examples]).to(device)
        kv = self._encode_visual_tokens(images)                       # one backbone call, (B, chunk_tokens, visual_dim)
        chunk_summary = self.visual_pool.chunk_pool(kv)                # (B, 64, visual_dim)

        if external_memory is None:
            B = len(examples)
            memory_in = self.visual_pool.global_pool.query.expand(B, -1, -1).to(device)
        else:
            memory_in = torch.as_tensor(
                np.asarray(external_memory), device=device, dtype=chunk_summary.dtype
            )

        self.visual_pool.global_pool._memory = memory_in
        new_memory = self.visual_pool.global_pool.step(chunk_summary)  # (B, 64, visual_dim)

        proj_dtype = self.visual_proj.weight.dtype
        visual_feats = self.visual_proj(new_memory.to(dtype=proj_dtype))
        text_feats = self._encode_text(instructions)

        condition = visual_feats + text_feats.unsqueeze(1)
        state_scale = float(getattr(self.config.framework, "state_scale", 1.0))
        if state_scale > 0.0 and self.state_proj is not None and states[0] is not None:
            condition = condition + state_scale * self._encode_state(states).unsqueeze(1)

        model_dtype = next(self.action_model.net.parameters()).dtype
        condition = condition.to(dtype=model_dtype)

        B = condition.shape[0]
        noise = torch.randn(
            B, self.action_horizon, self.action_dim,
            device=condition.device, dtype=model_dtype,
        )

        if self.action_model.ddim_diffusion is None:
            self.action_model.create_ddim(ddim_step=10)

        def _model_fn(x, t, **kwargs):
            return self.action_model.net.forward(x.to(dtype=model_dtype), t, **kwargs)

        samples = self.action_model.ddim_diffusion.ddim_sample_loop(
            _model_fn,
            noise.shape,
            noise,
            clip_denoised=False,
            model_kwargs=dict(z=condition),
            progress=False,
            device=condition.device,
            eta=0.0,
        )  # (B, action_horizon, action_dim)

        return {
            "normalized_actions": samples.float().cpu().numpy(),
            "memory": new_memory.float().cpu().numpy(),
        }
