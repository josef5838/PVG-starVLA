#!/bin/bash
# Training script for VGGTOmegaOFT on RoboTwin adjust_bottle task

set -e
cd /export/home/haoliang/projects/starVLA

export WANDB_MODE=offline   # logs stored locally; run `wandb sync <run_dir>/wandb` to upload later

config_yaml=examples/simBenchmarks/Robotwin/train_files/vggt_omega_robotwin.yaml
run_id=vggt_omega_oft_adjust_bottle

accelerate launch \
  --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml \
  --num_processes 2 \
  starVLA/training/train_starvla.py \
  --config_yaml ${config_yaml} \
  --framework.name VGGTOmegaOFT \
  --datasets.vla_data.data_mix robotwin_task1 \
  --datasets.vla_data.per_device_batch_size 8 \
  --trainer.max_train_steps 50000 \
  --trainer.save_interval 5000 \
  --trainer.logging_frequency 10 \
  --trainer.eval_interval 500 \
  --run_id ${run_id} \
  --wandb_project vggt-omega-action-pred \
  --wandb_entity josefshang \
  # --is_debug True
