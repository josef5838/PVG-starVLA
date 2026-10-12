#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd "$script_dir/../../../.." && pwd)
cd "$repo_root"
export PYTHONPATH="$repo_root${PYTHONPATH:+:$PYTHONPATH}"

: "${SURFLO_ROOT:?Set SURFLO_ROOT to the Surflo source checkout}"
: "${BEHAVIOR_DATA_ROOT:?Set BEHAVIOR_DATA_ROOT to the parent of behavior-1k-2026}"
[[ -f "$SURFLO_ROOT/checkpoints/surflo_v0.pt" ]] || { printf 'Missing Surflo release checkpoint\n' >&2; exit 1; }
[[ -f "$SURFLO_ROOT/configs/model/surflo.yaml" ]] || { printf 'Missing Surflo model config\n' >&2; exit 1; }
[[ -d "$BEHAVIOR_DATA_ROOT/behavior-1k-2026" ]] || { printf 'Missing BEHAVIOR dataset\n' >&2; exit 1; }

python_bin=${PYTHON_BIN:-python}
num_processes=${NUM_PROCESSES:-1}
[[ "$num_processes" =~ ^[1-9][0-9]*$ ]] || { printf 'NUM_PROCESSES must be positive\n' >&2; exit 1; }
arguments=(
  starVLA/training/train_starvla.py
  --config_yaml examples/simBenchmarks/Behavior/train_files/surflo_dit_turn_on_radio_2026.yaml
  --framework.surflo.source_root "$SURFLO_ROOT"
  --framework.surflo.checkpoint_path "$SURFLO_ROOT/checkpoints/surflo_v0.pt"
  --framework.surflo.model_config_path "$SURFLO_ROOT/configs/model/surflo.yaml"
  --datasets.vla_data.data_root_dir "$BEHAVIOR_DATA_ROOT"
  "$@"
)

if (( num_processes == 1 )); then
  exec env STARVLA_DISABLE_DEEPSPEED=1 "$python_bin" "${arguments[@]}"
fi
exec env STARVLA_DISABLE_DEEPSPEED=1 "${ACCELERATE_BIN:-accelerate}" launch \
  --num_processes "$num_processes" "${arguments[@]}"
