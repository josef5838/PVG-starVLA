#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat >&2 <<'EOF'
Usage: eval_method.sh --method METHOD --task TASK [options]

The method name is recorded in the output path; evaluation itself talks only to
the already-running local websocket policy server.

Options:
  --mode MODE                 train (development) or public_test (final)
  --instance-indices "IDS"    Space-separated instance positions (default: 0)
  --port PORT                 Policy server port (default: 8000)
  --output-root PATH          Root for raw evaluator outputs
  --execute                   Run the evaluator; otherwise print it

Required environment variables: BEHAVIOR_ROOT and BEHAVIOR_PYTHON.
EOF
}

method=""
task=""
mode=public_test
instance_indices=0
port=8000
output_root="${B1K_RESULTS_ROOT:-/data/b1k-eval}"
execute=false
while [[ $# -gt 0 ]]; do
  case "$1" in
    --method) method="$2"; shift 2 ;;
    --task) task="$2"; shift 2 ;;
    --mode) mode="$2"; shift 2 ;;
    --instance-indices) instance_indices="$2"; shift 2 ;;
    --port) port="$2"; shift 2 ;;
    --output-root) output_root="$2"; shift 2 ;;
    --execute) execute=true; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage; exit 2 ;;
  esac
done
[[ -n "$method" && -n "$task" ]] || { usage; exit 2; }
[[ "$mode" == train || "$mode" == public_test ]] || { echo "Invalid mode: $mode" >&2; exit 2; }
: "${BEHAVIOR_ROOT:?Set BEHAVIOR_ROOT to the v3.9.2 checkout}"
: "${BEHAVIOR_PYTHON:?Set BEHAVIOR_PYTHON to its behavior environment interpreter}"

destination="$output_root/$mode/$method/$task"
command=("$BEHAVIOR_PYTHON" -m omnigibson.eval.eval --task-name "$task" --robot-config omnigibson/eval/r1pro.yaml --mode "$mode" --instance-indices "$instance_indices" --host 127.0.0.1 --port "$port" --output-dir "$destination" --write-video)
printf 'Evaluation command:'
printf ' %q' "${command[@]}"
printf '\n'
if [[ "$execute" != true ]]; then
  echo "Dry run only. Start the local policy server and re-run with --execute."
  exit 0
fi
cd "$BEHAVIOR_ROOT/OmniGibson"
exec "${command[@]}"
