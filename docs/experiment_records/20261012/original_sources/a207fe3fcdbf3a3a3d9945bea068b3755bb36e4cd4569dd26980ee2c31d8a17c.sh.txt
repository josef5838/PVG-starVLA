#!/usr/bin/env bash
set -euo pipefail

cd /export/home/haoliang/projects/PVG-starVLA-snapshot
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"

STARVLA_DISABLE_DEEPSPEED=1 CUDA_VISIBLE_DEVICES=0 \
  /export/home/haoliang/miniconda3/envs/starVLA/bin/python \
  starVLA/training/train_starvla.py \
  --config_yaml examples/simBenchmarks/Behavior/train_files/surflo_dit_turn_on_radio_2026.yaml \
  --run_id surflo_camera3_six_part_state_20k \
  --framework.surflo.append_per_view_camera_tokens true \
  --framework.state_conditioning parts \
  --framework.action_model.num_condition_tokens 138 \
  --trainer.max_train_steps 20000 \
  --trainer.save_interval 10000 \
  --trainer.eval_interval 1000 \
  > results/surflo_camera3_six_part_state_20k.log 2>&1
