#!/usr/bin/env bash
# Compare checkpoints on the same simulator seeds; success is written by RoboTwin.
set -euo pipefail
if (( $# < 3 )); then
    echo "Usage: $0 <task> <seeds_file> <checkpoint> [checkpoint ...]" >&2
    exit 2
fi
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/../../../.." && pwd)"
cd "$repo_root"
task="$1"
seeds_file="$(realpath "$2")"
shift 2
server_gpu="${ROBOTWIN_SERVER_GPU:-0}"
eval_gpu="${ROBOTWIN_EVAL_GPU:-1}"
port="${ROBOTWIN_SERVER_PORT:-5694}"
python_bin="${STARVLA_PYTHON:-/export/home/haoliang/miniconda3/envs/starVLA/bin/python}"
server_pid=""
cleanup() {
    if [[ -n "$server_pid" ]]; then
        kill "$server_pid" 2>/dev/null || true
        wait "$server_pid" 2>/dev/null || true
        server_pid=""
    fi
}
trap cleanup EXIT INT TERM
for checkpoint in "$@"; do
    checkpoint="$(realpath "$checkpoint")"
    [[ -f "$checkpoint" ]] || { echo "Missing checkpoint: $checkpoint" >&2; exit 2; }
    run_name="$(basename "$(dirname "$(dirname "$checkpoint")")")"
    checkpoint_name="${run_name}_$(basename "$(dirname "$checkpoint")")_$(basename "$checkpoint" .pt)"
    log_dir="results/robotwin_surflo_eval/${task}/${checkpoint_name}"
    mkdir -p "$log_dir"
    "$python_bin" - "$port" <<'PYPORT'
import socket, sys
with socket.socket() as probe:
    probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    probe.bind(('127.0.0.1', int(sys.argv[1])))
PYPORT
    STARVLA_PYTHON="$python_bin" bash "$script_dir/run_policy_server.sh" "$checkpoint" "$server_gpu" "$port" > "$log_dir/server.log" 2>&1 &
    server_pid=$!
    "$python_bin" - "$port" "$server_pid" <<'PY'
import os, socket, sys, time
port, pid = map(int, sys.argv[1:])
deadline = time.monotonic() + 180
while time.monotonic() < deadline:
    os.kill(pid, 0)
    try:
        with socket.create_connection(('127.0.0.1', port), timeout=1):
            break
    except OSError:
        time.sleep(.5)
else:
    raise TimeoutError('Policy server did not become ready; inspect server.log')
PY
    bash "$script_dir/eval.sh" "$task" "${ROBOTWIN_TASK_CONFIG:-demo_clean}" "$checkpoint_name" 0 "$eval_gpu" "$checkpoint" "$port" 127.0.0.1 "$seeds_file" 2>&1 | tee "$log_dir/evaluation.log"
    cleanup
done
