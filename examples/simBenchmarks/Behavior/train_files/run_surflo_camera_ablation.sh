#!/usr/bin/env bash
set -euo pipefail

cd /export/home/haoliang/projects/PVG-starVLA-snapshot
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
config=examples/simBenchmarks/Behavior/train_files/surflo_dit_turn_on_radio_2026.yaml
python=/export/home/haoliang/miniconda3/envs/starVLA/bin/python

common=(
  --config_yaml "$config"
  --framework.surflo.append_per_view_camera_tokens true
  --trainer.max_train_steps 20000
  --trainer.save_interval 10000
  --trainer.eval_interval 1000
)

STARVLA_DISABLE_DEEPSPEED=1 CUDA_VISIBLE_DEVICES=0 "$python" \
  starVLA/training/train_starvla.py "${common[@]}" \
  --run_id surflo_camera3_additive_20k \
  --framework.action_model.num_condition_tokens 132 \
  --framework.state_conditioning add \
  > results/surflo_camera3_additive_20k.log 2>&1 &
additive_pid=$!

STARVLA_DISABLE_DEEPSPEED=1 CUDA_VISIBLE_DEVICES=1 "$python" \
  starVLA/training/train_starvla.py "${common[@]}" \
  --run_id surflo_camera3_state_token_20k \
  --framework.action_model.num_condition_tokens 133 \
  --framework.state_conditioning token \
  > results/surflo_camera3_state_token_20k.log 2>&1 &
state_token_pid=$!

status=0
wait "$additive_pid" || status=1
wait "$state_token_pid" || status=1
exit "$status"
