#!/bin/bash
# Single-GPU launcher for the VGGT-Omega baseline.
# Uses bs=256 on 1 GPU (same global batch as 2-GPU × bs=128) to isolate
# inter-GPU communication overhead as the only variable.
# After regenerating the visual token cache, increase to bs=512 (4× per-device).
#
# Usage: bash run_ablation.sh

set -e
cd /export/home/haoliang/projects/starVLA
export PATH="/export/home/haoliang/miniconda3/envs/starVLA/bin:$PATH"

config_yaml="examples/simBenchmarks/Robotwin/train_files/ablation_vggt_omega.yaml"

export WANDB_MODE=online
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

run_id="ablation_vggt_omega_rotate_qrcode_$(date +%Y%m%d_%H%M%S)"

accelerate launch \
  --config_file starVLA/config/single_gpu.yaml \
  starVLA/training/train_starvla.py \
  --config_yaml ${config_yaml} \
  --datasets.vla_data.data_mix robotwin_rotate_qrcode \
  --datasets.vla_data.per_device_batch_size 256 \
  --run_id ${run_id} \
  --wandb_project vggt-omega-ablation \
  --wandb_entity mango_juice
