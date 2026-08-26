#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat >&2 <<'EOF'
Usage: train_method.sh --method METHOD --dataset DATASET --run-id ID [options]

Methods: vggt_surflo, pi05, fastwam, cosmos_policy
Options:
  --steps N          Training steps (default: 10; suitable only for a smoke test)
  --batch-size N     Global batch size (default: 4)
  --gpus IDS         CUDA_VISIBLE_DEVICES value (default: 0)
  --execute          Run the printed command. Without it this is a dry run.

For pi05 set OPENPI_ROOT and OPENPI_CONFIG. OPENPI_ASSET_ID defaults to
turning_on_radio. The OpenPI config must already point to --dataset.

For fastwam set FASTWAM_ROOT; it must contain the committed BEHAVIOR config.
FASTWAM_CONDA_ENV defaults to fastwam and FASTWAM_NPROC_PER_NODE defaults to 1.
EOF
}

method=""
dataset=""
run_id=""
steps=10
batch_size=4
gpus=0
execute=false

while [[ $# -gt 0 ]]; do
  case "$1" in
    --method) method="$2"; shift 2 ;;
    --dataset) dataset="$2"; shift 2 ;;
    --run-id) run_id="$2"; shift 2 ;;
    --steps) steps="$2"; shift 2 ;;
    --batch-size) batch_size="$2"; shift 2 ;;
    --gpus) gpus="$2"; shift 2 ;;
    --execute) execute=true; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage; exit 2 ;;
  esac
done

[[ -n "$method" && -n "$dataset" && -n "$run_id" ]] || { usage; exit 2; }
[[ -d "$dataset/meta" ]] || { echo "Not a prepared LeRobot dataset: $dataset" >&2; exit 2; }

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
render() { printf ' %q' "$@"; printf '\n'; }

case "$method" in
  vggt_surflo)
    dataset_parent="$(dirname "$(realpath "$dataset")")"
    command=(env "CUDA_VISIBLE_DEVICES=$gpus" "BEHAVIOR_DATA_PARENT=$dataset_parent" "RUN_ID=$run_id" "MAX_TRAIN_STEPS=$steps" "BATCH_SIZE=$batch_size" "WANDB_MODE=offline" bash "$repo_root/examples/simBenchmarks/Behavior/train_files/run_vggt_omega_behavior2026.sh")
    ;;
  pi05)
    : "${OPENPI_ROOT:?Set OPENPI_ROOT to the pinned OpenPI checkout}"
    : "${OPENPI_CONFIG:?Set OPENPI_CONFIG to the portable B1K config, e.g. pi05_b1k_radio}"
    asset_id="${OPENPI_ASSET_ID:-turning_on_radio}"
    stats_path="$OPENPI_ROOT/outputs/assets/$OPENPI_CONFIG/$asset_id/norm_stats.json"
    stats_command=(uv run --no-sync scripts/compute_norm_stats.py --config-name "$OPENPI_CONFIG")
    command=(env "CUDA_VISIBLE_DEVICES=$gpus" "XLA_PYTHON_CLIENT_MEM_FRACTION=${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.85}" uv run --no-sync scripts/b1k/train_b1k.py "$OPENPI_CONFIG" "--exp_name=$run_id" --overwrite "--batch_size=$batch_size" "--num_train_steps=$steps")
    ;;
  fastwam)
    : "${FASTWAM_ROOT:?Set FASTWAM_ROOT to the pinned FastWAM checkout}"
    fastwam_config="${FASTWAM_CONFIG:-behavior1k_radio_uncond_3cam_384_1e-4}"
    fastwam_env="${FASTWAM_CONDA_ENV:-fastwam}"
    fastwam_nproc="${FASTWAM_NPROC_PER_NODE:-1}"
    command=(env "CUDA_VISIBLE_DEVICES=$gpus" "FASTWAM_B1K_DATASET_ROOT=$dataset" "RUN_ID=$run_id" "WANDB_MODE=offline" conda run --no-capture-output -n "$fastwam_env" bash scripts/train_zero1.sh "$fastwam_nproc" "task=$fastwam_config" "batch_size=$batch_size" "max_steps=$steps")
    ;;
  cosmos_policy)
    echo "Cosmos Policy is intentionally blocked: complete benchmarks/behavior1k/adapters/cosmos_policy.md first." >&2
    exit 3
    ;;
  *) echo "Unknown method: $method" >&2; usage; exit 2 ;;
esac

echo "Method: $method"
echo "Dataset: $(realpath "$dataset")"
echo "Run ID: $run_id"
if [[ "$method" == pi05 && ! -f "$stats_path" ]]; then
  echo "Normalization statistics are absent; first command (from $OPENPI_ROOT):"
  (cd "$OPENPI_ROOT" && render "${stats_command[@]}")
fi
echo "Training command:"
render "${command[@]}"

if [[ "$execute" != true ]]; then
  echo "Dry run only. Re-run with --execute when the selected GPUs are idle."
  exit 0
fi

if [[ "$method" == pi05 ]]; then
  if [[ ! -f "$stats_path" ]]; then
    (cd "$OPENPI_ROOT" && "${stats_command[@]}")
  fi
  cd "$OPENPI_ROOT"
elif [[ "$method" == fastwam ]]; then
  cd "$FASTWAM_ROOT"
fi
exec "${command[@]}"
