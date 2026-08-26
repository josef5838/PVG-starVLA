#!/usr/bin/env bash
set -euo pipefail

tasks_jsonl="${TASKS_JSONL:-/models/tasks.jsonl}"
host="${POLICY_HOST:-0.0.0.0}"
port="${POLICY_PORT:-8000}"
device="${POLICY_DEVICE:-cuda}"
dtype="${POLICY_DTYPE:-auto}"
execute_horizon="${EXECUTE_HORIZON:-8}"
export MODEL_ROOT="${MODEL_ROOT:-/models}"

checkpoint="${MODEL_CHECKPOINT:-}"
if [[ -z "${checkpoint}" ]]; then
  shopt -s nullglob
  candidates=(/models/run/final_model/*.pt /models/run/final_model/*.safetensors)
  shopt -u nullglob
  if [[ ${#candidates[@]} -ne 1 ]]; then
    echo "Expected exactly one checkpoint under /models/run/final_model" >&2
    exit 2
  fi
  checkpoint="${candidates[0]}"
fi

if [[ ! -f "${checkpoint}" ]]; then
  echo "Missing model checkpoint: ${checkpoint}" >&2
  exit 2
fi
if [[ ! -f "${tasks_jsonl}" ]]; then
  echo "Missing task descriptions: ${tasks_jsonl}" >&2
  exit 2
fi

exec python /opt/starvla/examples/simBenchmarks/Behavior/serve_vggt_omega_2026.py \
  --ckpt-path "${checkpoint}" \
  --tasks-jsonl "${tasks_jsonl}" \
  --host "${host}" \
  --port "${port}" \
  --device "${device}" \
  --dtype "${dtype}" \
  --execute-horizon "${execute_horizon}"
