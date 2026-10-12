#!/usr/bin/env bash
set -euo pipefail
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
cd "$repo_root"
export PYTHONPATH="$repo_root${PYTHONPATH:+:$PYTHONPATH}"
export STARVLA_DISABLE_DEEPSPEED=1
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export WANDB_MODE="${WANDB_MODE:-offline}"
python_bin="${PYTHON_BIN:-/export/home/haoliang/miniconda3/envs/starVLA/bin/python}"
task="${TASK:-rotate_qrcode}"
case "$task" in adjust_bottle|rotate_qrcode|stack_blocks_three) ;; *) echo "Unsupported TASK: $task" >&2; exit 2 ;; esac
run_id="${RUN_ID:-surflo_camera3_body_parts_robotwin_${task}_20ksamples_$(date +%Y%m%d_%H%M%S)}"
exec "$python_bin" starVLA/training/train_starvla.py \
  --config_yaml examples/simBenchmarks/Robotwin/train_files/surflo_camera3_body_parts_robotwin.yaml \
  --datasets.vla_data.data_mix "robotwin_surflo_${task}" \
  --run_id "$run_id" "$@"
