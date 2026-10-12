#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 6 ]]; then
    echo "Usage: bash examples/simBenchmarks/Robotwin/eval_files/eval.sh <task_name> <task_config> <ckpt_setting> <seed> <gpu_id> <policy_ckpt_path> [policy_port] [policy_host]" >&2
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"

ROBOTWIN_PATH="${ROBOTWIN_PATH:-/export/home/haoliang/projects/RoboTwin_PVG}"
if [[ ! -d "${ROBOTWIN_PATH}" ]]; then
    echo "ROBOTWIN_PATH does not exist: ${ROBOTWIN_PATH}" >&2
    exit 1
fi

ROBOTWIN_EVAL_SCRIPT="${ROBOTWIN_PATH}/scripts/eval_policy_xpolicylab.py"

if [[ ! -f "${ROBOTWIN_EVAL_SCRIPT}" ]]; then
    echo "PVG XPolicyLab eval entry does not exist: ${ROBOTWIN_EVAL_SCRIPT}" >&2
    exit 1
fi

task_name="$1"
task_config="$2"
ckpt_setting="${3:-starvla_demo}"
seed="${4:-0}"
gpu_id="${5:-0}"
policy_ckpt_path="$6"
policy_port="${7:-${ROBOTWIN_POLICY_PORT:-5694}}"
policy_host="${8:-${ROBOTWIN_POLICY_HOST:-127.0.0.1}}"
seeds_file="${9:-}"
policy_name="${ROBOTWIN_POLICY_NAME:-starVLA}"
robotwin_python="${ROBOTWIN_PYTHON:-python}"
action_type="${ROBOTWIN_ACTION_TYPE:-joint}"

export PYTHONPATH="${ROBOTWIN_PATH}:${REPO_ROOT}:${PYTHONPATH:-}"
export PATH="$(dirname "${robotwin_python}"):${PATH}"

run_eval() {
    local eval_seed="$1"
    local -a args=(
        "${ROBOTWIN_EVAL_SCRIPT}"
        --task_name "${task_name}"
        --task_config "${task_config}"
        --policy_name "${policy_name}"
        --protocol starvla_ws
        --host "${policy_host}"
        --port "${policy_port}"
        --device_id "${gpu_id}"
        --seed "${eval_seed}"
        --additional_info "ckpt_name=${ckpt_setting},action_type=${action_type},policy_ckpt_path=${policy_ckpt_path},starvla_adapter_path=${SCRIPT_DIR}"
    )

    if [[ -n "${seeds_file}" ]]; then
        args+=(--test_num 1 --expert_check false)
    fi

    (
        cd "${ROBOTWIN_PATH}"
        PYTHONWARNINGS=ignore::UserWarning "${robotwin_python}" "${args[@]}"
    )
}

echo "[Robotwin/XPolicyLab] task=${task_name} config=${task_config} policy=${policy_name}"
echo "[Robotwin/XPolicyLab] server=${policy_host}:${policy_port} checkpoint=${policy_ckpt_path}"

if [[ -z "${seeds_file}" ]]; then
    run_eval "${seed}"
    exit 0
fi

if [[ ! -f "${seeds_file}" ]]; then
    echo "Seed file does not exist: ${seeds_file}" >&2
    exit 1
fi

mapfile -t fixed_seeds < <(tr -s "[:space:]" "\n" < "${seeds_file}" | sed "/^$/d")
if (( ${#fixed_seeds[@]} == 0 )); then
    echo "Seed file is empty: ${seeds_file}" >&2
    exit 1
fi

for fixed_seed in "${fixed_seeds[@]}"; do
    if [[ ! "${fixed_seed}" =~ ^[0-9]+$ ]]; then
        echo "Invalid seed in ${seeds_file}: ${fixed_seed}" >&2
        exit 1
    fi
    run_eval "${fixed_seed}"
done
