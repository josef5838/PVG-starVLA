# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License");

"""
Option B trainer: true stateful streaming training for
HierarchicalGatedMemoryPool's global memory. Subclasses VLATrainer,
reusing its config/checkpoint/wandb/distributed-setup scaffolding unchanged
(_save_initial_configs, _init_checkpointing, freeze_backbones,
_finalize_training, _log_metrics, etc.) and overriding only the parts that
depend on the standard i.i.d.-random-window dataloader: data iteration and
the single training step.

Each training step advances `num_slots` independent, parallel episode
streams by exactly one 8-timestep chunk each, carrying a persistent
per-slot global-memory tensor across steps (detached after every step —
truncated BPTT with truncation length 1: gradients flow through the current
step's cross-attn/gate/self-attn weights, which are shared and therefore
still receive signal accumulated across the whole training run, but not
backward through earlier chunks within a single backward() call). See
episode_chunk_stream.py and VGGTOmegaOFT.streaming_step for the mechanics.
"""

import argparse
import os
from pathlib import Path
from typing import Tuple

import torch
import torch.distributed as dist
from accelerate import Accelerator, DistributedDataParallelKwargs
from accelerate.logging import get_logger
from omegaconf import OmegaConf

from starVLA.dataloader.lerobot_datasets import make_LeRobotSingleDataset
from starVLA.model.framework.base_framework import build_framework
from starVLA.model.framework.share_tools import apply_config_compat
from starVLA.training.episode_chunk_stream import EpisodeChunkStream
from starVLA.training.train_starvla import VLATrainer, setup_directories, setup_optimizer_and_scheduler
from starVLA.training.trainer_utils.config_tracker import wrap_config
from starVLA.training.trainer_utils.trainer_tools import normalize_dotlist_args

# Never use DeepSpeed here: this environment has no CUDA toolkit (CUDA_HOME
# unset), which DeepSpeed's op builder requires just to import, and the
# model comfortably fits on one GPU anyway so ZeRO partitioning buys nothing.
# Plain accelerate multi-GPU DDP (accelerate launch --num_processes N
# --multi_gpu) handles the 2-GPU case instead. Importing VLATrainer below
# from train_starvla triggers THAT module's own top-level Accelerator
# construction first; STARVLA_DISABLE_DEEPSPEED (set by this script's
# launch command) makes it also skip DeepSpeed, so the two don't collide
# ("cannot pass a deepspeed_plugin when creating a second Accelerator").
#
# find_unused_parameters=True: required for correct gradient sync in the
# streaming path specifically. Confirmed via a minimal 2-process repro on
# this exact model+checkpoint: with the default False, DDP silently (no
# error) failed to all-reduce gradients at all (each rank's post-backward
# grad reflected only its own local data); with True, both ranks' grads
# became bit-identical. Some parameters genuinely go unused per streaming
# forward call (root cause not further isolated; VGGTOmegaOFT's DiT action
# head is large enough that this is plausible without being a correctness
# concern in itself). Costs an extra autograd-graph traversal per step to
# detect unused parameters -- a real but acceptable overhead given the
# alternative is silently wrong (unsynced) training.
ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
accelerator = Accelerator(kwargs_handlers=[ddp_kwargs])
accelerator.print(accelerator.state)

os.environ["TOKENIZERS_PARALLELISM"] = "false"
logger = get_logger(__name__)


def prepare_stream(cfg, rank: int = 0) -> EpisodeChunkStream:
    """
    Build the episode-ordered chunk stream in place of the standard
    dataloader. robot_type is hardcoded to the *_hist8stride1 variant of
    whatever base data_mix is configured, since streaming needs raw
    8-consecutive-timestep chunks (AgilexDataConfigHist8Stride1-style),
    not the data_mix's own (possibly wide-stride) video_observation_indices.

    rank offsets the RNG seed so that under distributed data-parallel
    training (accelerate launch --num_processes N), each rank's
    EpisodeChunkStream samples a DIFFERENT sequence of episodes rather than
    redundantly duplicating rank 0's — otherwise N GPUs would compute
    gradients on identical data every step, wasting N-1 GPUs' worth of
    compute for zero additional data coverage.
    """
    data_cfg = cfg.datasets.vla_data
    num_slots = int(data_cfg.per_device_batch_size)
    chunk_timesteps = int(getattr(cfg.framework.visual_backbone, "pool_chunk_timesteps", 8))

    single_dataset = make_LeRobotSingleDataset(
        data_root_dir=Path(data_cfg.data_root_dir),
        data_name=data_cfg.streaming_data_name,
        robot_type=data_cfg.streaming_robot_type,
        data_cfg={
            "video_backend": data_cfg.get("video_backend", "torchvision_av"),
            # _pack_sample reads this from the runtime data_cfg dict, NOT
            # from the robot_type class's own config (datasets.py:1453) —
            # must be set explicitly here or state is silently omitted.
            "include_state": bool(data_cfg.get("include_state", True)),
        },
    )
    logger.info(
        f"[rank {rank}] EpisodeChunkStream: {len(single_dataset.trajectory_ids)} episodes, "
        f"{num_slots} parallel slots, {chunk_timesteps}-timestep chunks, seed={int(cfg.seed) + rank}."
    )

    # The standard pipeline (build_dataloader -> LeRobotMixtureDataset.save_dataset_statistics,
    # starVLA/dataloader/__init__.py:56) writes this as a side effect of dataloader
    # construction; prepare_stream bypasses that path entirely, so it must be
    # written explicitly here or checkpoints from this trainer fail
    # baseframework.from_pretrained's `dataset_statistics.json` existence check at
    # eval time (share_tools.py read_mode_config).
    if not dist.is_initialized() or dist.get_rank() == 0:
        single_dataset._save_dataset_statistics_(Path(cfg.output_dir) / "dataset_statistics.json")

    return EpisodeChunkStream(
        single_dataset, num_slots=num_slots, chunk_timesteps=chunk_timesteps, seed=int(cfg.seed) + rank,
    )


class StreamingTrainer(VLATrainer):
    def prepare_training(self):
        from accelerate.utils import set_seed

        rank = dist.get_rank() if dist.is_initialized() else 0
        seed = self.config.seed + rank if hasattr(self.config, "seed") else rank + 3047
        set_seed(seed)

        self._save_initial_configs()
        self._init_checkpointing()
        self._adjust_lr_scheduler_for_resume()

        freeze_modules = (
            self.config.trainer.freeze_modules
            if (self.config and hasattr(self.config.trainer, "freeze_modules"))
            else None
        )
        self.model = self.freeze_backbones(self.model, freeze_modules=freeze_modules)
        self.print_trainable_parameters(self.model)

        # Only model + optimizer go through accelerator.prepare(): the
        # stream is a plain Python object, not a torch DataLoader, and must
        # keep its own single-process, in-order iteration semantics intact.
        self.model, self.optimizer = self.setup_distributed_training(
            self.accelerator, self.model, self.optimizer,
        )
        self._init_wandb()

        self._slot_memory = None  # (num_slots, 64, visual_dim), lazily initialized on first step

    def _create_data_iterators(self):
        pass  # EpisodeChunkStream needs no separate iterator object

    def _get_next_batch(self):
        return self.vla_train_dataloader.next_batch()  # (examples, reset_mask)

    def _build_external_memory(self, raw_model, reset_mask, device):
        num_slots = len(reset_mask)
        query_default = raw_model.visual_pool.global_pool.query.expand(num_slots, -1, -1).to(device)
        if self._slot_memory is None:
            self._slot_memory = query_default.clone()
        reset_mask_t = torch.tensor(reset_mask, device=device).view(-1, 1, 1)
        return torch.where(reset_mask_t, query_default, self._slot_memory.to(device))

    def _train_step(self, batch_vla, batch_vlm=None):
        examples, reset_mask = batch_vla
        raw_model = getattr(self.model, "module", self.model)
        device = raw_model.visual_proj.weight.device

        with self.accelerator.accumulate(self.model):
            self.optimizer.zero_grad()

            external_memory = self._build_external_memory(raw_model, reset_mask, device)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                # Must go through self.model(...) (DDP.__call__), NOT
                # raw_model.<method>(...) on the unwrapped module — DDP only
                # arms its gradient-sync hooks via its own forward()/__call__.
                # Calling a custom method directly on `.module` silently
                # skips synchronization (confirmed empirically; see
                # VGGTOmegaOFT.forward's docstring) and each rank trains an
                # independent, unsynced replica under multi-GPU.
                output_dict, new_memory = self.model(examples, external_memory=external_memory)
                action_loss = output_dict["action_loss"]

            self.accelerator.backward(action_loss)

            if self.config.trainer.gradient_clipping is not None:
                self.accelerator.clip_grad_norm_(self.model.parameters(), self.config.trainer.gradient_clipping)

            self.optimizer.step()
            self.lr_scheduler.step()

        # Truncated BPTT (length 1): detach before storing for next step.
        self._slot_memory = new_memory.detach()

        metrics = {"action_dit_loss": action_loss.item()}
        if "action_loss_main" in output_dict:
            # Only present when chunk_aux_loss_weight > 0 (see
            # VGGTOmegaOFT.streaming_step): action_dit_loss above is the
            # combined total actually backpropped; these are the two
            # components, logged separately so the auxiliary loss's effect
            # on chunk_pool collapse can be tracked during training rather
            # than only inferred after the fact.
            metrics["action_loss_main"] = output_dict["action_loss_main"].item()
            metrics["action_loss_aux"] = output_dict["action_loss_aux"].item()

        return metrics

    def eval_action_model(self, step_metrics: dict = None) -> dict:
        # predict_action's fixed-window replay path isn't compatible with
        # streaming state; real evaluation happens via the RoboTwin rollout
        # eval after training, same as every other run tonight.
        return step_metrics


def main(cfg) -> None:
    logger.info("VLA Streaming Training :: Warming Up")
    cfg = wrap_config(cfg)

    output_dir = setup_directories(cfg=cfg)
    vla = build_framework(cfg)
    vla_stream = prepare_stream(cfg, rank=accelerator.process_index)
    optimizer, lr_scheduler = setup_optimizer_and_scheduler(model=vla, cfg=cfg)

    accelerator.gradient_accumulation_steps = int(getattr(cfg.trainer, "gradient_accumulation_steps", 1))

    trainer = StreamingTrainer(
        cfg=cfg,
        model=vla,
        vla_train_dataloader=vla_stream,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        accelerator=accelerator,
    )

    trainer.prepare_training()
    trainer.train()

    logger.info("... and that's all, folks!")
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_yaml", type=str, required=True, help="Path to YAML config")
    args, clipargs = parser.parse_known_args()

    cfg = OmegaConf.load(args.config_yaml)
    dotlist = normalize_dotlist_args(clipargs)
    cli_cfg = OmegaConf.from_dotlist(dotlist)
    cfg = OmegaConf.merge(cfg, cli_cfg)
    cfg = apply_config_compat(cfg)

    main(cfg)
