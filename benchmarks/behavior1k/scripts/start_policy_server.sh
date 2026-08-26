#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat >&2 <<'EOF'
Usage: start_policy_server.sh --method METHOD --task TASK [--port PORT] [--execute]

Required environment variables:
  vggt_surflo: CKPT_PATH, B1K_TASKS_JSONL
  pi05:        OPENPI_ROOT, OPENPI_CONFIG, OPENPI_CKPT_PATH

Without --execute the script only renders the server command.
EOF
}

method=""
task=""
port=8000
execute=false
while [[ $# -gt 0 ]]; do
  case "$1" in
    --method) method="$2"; shift 2 ;;
    --task) task="$2"; shift 2 ;;
    --port) port="$2"; shift 2 ;;
    --execute) execute=true; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage; exit 2 ;;
  esac
done
[[ -n "$method" && -n "$task" ]] || { usage; exit 2; }

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
render() { printf ' %q' "$@"; printf '\n'; }

case "$method" in
  vggt_surflo)
    : "${CKPT_PATH:?Set CKPT_PATH to the transferred StarVLA checkpoint}"
    : "${B1K_TASKS_JSONL:?Set B1K_TASKS_JSONL to dataset meta/tasks.jsonl}"
    command=("${STARVLA_PYTHON:-python}" "$repo_root/examples/simBenchmarks/Behavior/serve_vggt_omega_2026.py" --ckpt-path "$CKPT_PATH" --tasks-jsonl "$B1K_TASKS_JSONL" --host 127.0.0.1 --port "$port" --execute-horizon "${EXECUTE_HORIZON:-16}" --dtype "${POLICY_DTYPE:-auto}")
    ;;
  pi05)
    : "${OPENPI_ROOT:?Set OPENPI_ROOT to the pinned OpenPI checkout}"
    : "${OPENPI_CONFIG:?Set OPENPI_CONFIG, e.g. pi05_b1k_radio}"
    : "${OPENPI_CKPT_PATH:?Set OPENPI_CKPT_PATH to the transferred OpenPI checkpoint directory}"
    command=(uv run --no-sync scripts/b1k/serve_b1k.py --robot b1k/R1Pro --task "b1k/$task" --repo_id "${OPENPI_ASSET_ID:-$task}" policy:checkpoint --policy.config "$OPENPI_CONFIG" --policy.dir "$OPENPI_CKPT_PATH" --port "$port" --action_horizon "${EXECUTE_HORIZON:-16}")
    ;;
  cosmos_policy)
    echo "Cosmos Policy server is not ready; see adapters/cosmos_policy.md." >&2
    exit 3
    ;;
  *) echo "Unknown method: $method" >&2; usage; exit 2 ;;
esac

echo "Policy server command:"
if [[ "$method" == pi05 ]]; then
  (cd "$OPENPI_ROOT" && render "${command[@]}")
else
  render "${command[@]}"
fi
if [[ "$execute" != true ]]; then
  echo "Dry run only. Re-run with --execute on the RTX machine."
  exit 0
fi
[[ "$method" == pi05 ]] && cd "$OPENPI_ROOT"
exec "${command[@]}"
