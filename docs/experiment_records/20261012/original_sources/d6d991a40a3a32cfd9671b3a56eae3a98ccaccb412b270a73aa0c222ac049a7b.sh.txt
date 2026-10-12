#!/usr/bin/env bash
# RoboCasa365 (PandaOmron) — single-task walk-through training with Qwen3VL-OFT.
# Run from the repo root inside the `starVLA` conda env.
set -eo pipefail

# ==============================================================
#  Platform-specific settings — adjust these for YOUR cluster
# ==============================================================
# WandB — disabled for the walk-through; remove this line and set
# WANDB_API_KEY (or run `wandb login`) to enable logging.
export WANDB_MODE=disabled

# NCCL networking — uncomment and edit to match your InfiniBand / RoCE setup:
#   export NCCL_SOCKET_IFNAME=<your_network_interface>   # e.g. eth0, bond0
#   export NCCL_IB_HCA=<your_ib_devices>                # e.g. mlx5_0,mlx5_1
#   export NCCL_BLOCKING_WAIT=1
#   export NCCL_ASYNC_ERROR_HANDLING=1
#   export NCCL_TIMEOUT=1000

# CUDA / nvcc — DeepSpeed requires a real nvcc; point to your toolkit if needed:
#   export CUDA_HOME=/path/to/cuda                       # e.g. /usr/local/cuda-12.2
#   export PATH=${CUDA_HOME}/bin:${PATH}
# ==============================================================

# Activate conda env if not already active.
if [[ "${CONDA_DEFAULT_ENV:-}" != "starVLA" ]]; then
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate starVLA
fi

# How many GPUs to use; falls back to "all visible".
NUM_GPUS=${NUM_GPUS:-$(python -c "import torch;print(torch.cuda.device_count())")}

# Pick accelerate config: single GPU gets the simple config, multi-GPU uses DeepSpeed ZeRO-2.
if [[ "${NUM_GPUS}" -eq 1 ]]; then
  ACCEL_CONFIG=starVLA/config/single_gpu.yaml
else
  ACCEL_CONFIG=starVLA/config/deepseeds/deepspeed_zero2.yaml
fi

# Auto-select a free port for distributed rendezvous.
MASTER_PORT=$(python -c "import socket; s=socket.socket(); s.bind(('',0)); p=s.getsockname()[1]; s.close(); print(p)")

# ---- training knobs (edit here) ----
TASK=OpenDrawer
BATCH=8
# Walk-through: 100 steps to validate the pipeline end-to-end.
# For real training raise to e.g. 100000 and adjust SAVE_EVERY accordingly.
MAX_STEPS=100
SAVE_EVERY=100
EVAL_EVERY=1000
LOG_EVERY=10

run_root_dir=./playground/Checkpoints
run_id=robocasa365_qwenoft_${TASK}_100step
output_dir=${run_root_dir}/${run_id}
mkdir -p "${output_dir}"
cp "$0" "${output_dir}/"

accelerate launch \
  --config_file "${ACCEL_CONFIG}" \
  --num_processes "${NUM_GPUS}" \
  --main_process_port "${MASTER_PORT}" \
  starVLA/training/train_starvla.py \
  --config_yaml ./examples/simBenchmarks/Robocasa_365/train_files/starvla_qwenoft_robocasa365.yaml \
  --datasets.vla_data.per_device_batch_size "${BATCH}" \
  --trainer.max_train_steps "${MAX_STEPS}" \
  --trainer.save_interval "${SAVE_EVERY}" \
  --trainer.logging_frequency "${LOG_EVERY}" \
  --trainer.eval_interval "${EVAL_EVERY}" \
  --run_root_dir "${run_root_dir}" \
  --run_id "${run_id}" \
  --wandb_project starVLA_robocasa365
