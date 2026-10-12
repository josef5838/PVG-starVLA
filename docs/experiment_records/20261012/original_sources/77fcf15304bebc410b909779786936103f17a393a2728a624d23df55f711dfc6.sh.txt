#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${repo_root}"

: "${BEHAVIOR_DATA_PARENT:?Set BEHAVIOR_DATA_PARENT to the directory containing behavior-1k-2026}"

config_yaml="examples/simBenchmarks/Behavior/train_files/vggt_omega_gated_self_attn_2026.yaml"
run_id="${RUN_ID:-vggt_omega_behavior2026_$(date +%Y%m%d_%H%M%S)}"

export WANDB_MODE="${WANDB_MODE:-offline}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

accelerate launch \
  --config_file starVLA/config/single_gpu.yaml \
  starVLA/training/train_starvla.py \
  --config_yaml "${config_yaml}" \
  --datasets.vla_data.data_root_dir "${BEHAVIOR_DATA_PARENT}" \
  --datasets.vla_data.per_device_batch_size "${BATCH_SIZE:-4}" \
  --trainer.max_train_steps "${MAX_TRAIN_STEPS:-10000}" \
  --run_id "${run_id}"
