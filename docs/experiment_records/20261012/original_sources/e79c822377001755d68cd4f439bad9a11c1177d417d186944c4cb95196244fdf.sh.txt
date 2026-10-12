#!/usr/bin/env bash
# Collect RoboTwin_PVG trajectories and convert them with XPolicyLab.
set -euo pipefail

TASK=""
CONFIG="demo_clean"
NUM=50
GPU=0
ROBOTWIN_ENV="RoboTwin"
STARVLA_ENV="starVLA"
ROBOTWIN_PATH="${ROBOTWIN_PATH:-/export/home/haoliang/projects/RoboTwin_PVG}"
EMBODIMENT="${ROBOTWIN_EMBODIMENT:-aloha_agilex}"
REPO_ID=""
SKIP_COLLECT=false
SKIP_CONVERT=false

usage() {
    cat <<EOF
Usage: bash $(basename "$0") --task <task> [options]

Options:
  --config <name>          PVG task config (default: demo_clean)
  --num <episodes>         Episodes to collect/convert (default: 50)
  --gpu <id>               Simulator GPU (default: 0)
  --robotwin-env <name>    Conda environment for RoboTwin_PVG
  --starvla-env <name>     Conda environment with LeRobot v2.1
  --robotwin-path <path>   RoboTwin_PVG checkout
  --embodiment <name>      PVG data embodiment (default: aloha_agilex)
  --repo-id <id>           LeRobot dataset name
  --skip-collect           Reuse PVG-native trajectories already on disk
  --skip-convert           Stop after collection
  --skip-process           Accepted for compatibility; PVG data needs no GO1 process step
  --datasets-root <path>   Deprecated compatibility option; conversion uses HF_LEROBOT_HOME
  --modality <path>        Deprecated compatibility option
  --no-register            Deprecated compatibility option; this script never edits a registry
EOF
}

while (( $# > 0 )); do
    case "$1" in
        --task) TASK="$2"; shift 2 ;;
        --config) CONFIG="$2"; shift 2 ;;
        --num) NUM="$2"; shift 2 ;;
        --gpu) GPU="$2"; shift 2 ;;
        --robotwin-env) ROBOTWIN_ENV="$2"; shift 2 ;;
        --starvla-env) STARVLA_ENV="$2"; shift 2 ;;
        --robotwin-path) ROBOTWIN_PATH="$2"; shift 2 ;;
        --embodiment) EMBODIMENT="$2"; shift 2 ;;
        --repo-id) REPO_ID="$2"; shift 2 ;;
        --skip-collect) SKIP_COLLECT=true; shift ;;
        --skip-convert) SKIP_CONVERT=true; shift ;;
        --skip-process) echo "[gen-data] PVG-native data has no process step."; shift ;;
        --datasets-root|--modality) echo "[gen-data] $1 is deprecated; use HF_LEROBOT_HOME."; shift 2 ;;
        --no-register) echo "[gen-data] Registry mutation is intentionally disabled."; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown option: $1" >&2; usage >&2; exit 1 ;;
    esac
done

[[ -n "${TASK}" ]] || { echo "ERROR: --task is required" >&2; usage >&2; exit 1; }
[[ "${NUM}" =~ ^[1-9][0-9]*$ ]] || { echo "ERROR: --num must be a positive integer" >&2; exit 1; }
[[ -d "${ROBOTWIN_PATH}" ]] || { echo "ERROR: RoboTwin_PVG path does not exist: ${ROBOTWIN_PATH}" >&2; exit 1; }

TASK_CONFIG="${ROBOTWIN_PATH}/env_cfg/task_config/${CONFIG}.yml"
CONVERTER="${ROBOTWIN_PATH}/XPolicyLab/scripts/transform_lerobot_v21_format.py"
DATA_DIR="${ROBOTWIN_PATH}/data/${CONFIG}/${TASK}/${EMBODIMENT}/data"
REPO_ID="${REPO_ID:-robotwin_${TASK}_${CONFIG}_${EMBODIMENT}}"

[[ -f "${TASK_CONFIG}" ]] || { echo "ERROR: task config does not exist: ${TASK_CONFIG}" >&2; exit 1; }
[[ -f "${CONVERTER}" ]] || { echo "ERROR: XPolicyLab converter does not exist: ${CONVERTER}" >&2; exit 1; }

log() { echo "[gen-data] $*"; }

if ! ${SKIP_COLLECT}; then
    log "Collecting ${NUM} PVG-native episodes for ${TASK}."
    config_backup="$(mktemp "${TMPDIR:-/tmp}/robotwin_pvg_config.XXXXXX.yml")"
    cp "${TASK_CONFIG}" "${config_backup}"
    restore_config() {
        cp "${config_backup}" "${TASK_CONFIG}"
        rm -f "${config_backup}"
    }
    trap restore_config EXIT
    sed -i "s/^episode_num:.*/episode_num: ${NUM}/" "${TASK_CONFIG}"
    (cd "${ROBOTWIN_PATH}" && conda run --no-capture-output -n "${ROBOTWIN_ENV}" bash collect_data.sh "${TASK}" "${CONFIG}" "${GPU}")
    restore_config
    trap - EXIT
else
    log "Skipping collection and reusing ${DATA_DIR}."
fi

[[ -d "${DATA_DIR}" ]] || { echo "ERROR: PVG-native trajectory directory does not exist: ${DATA_DIR}" >&2; exit 1; }

if ${SKIP_CONVERT}; then
    log "Collection complete. Conversion skipped."
    exit 0
fi

log "Converting ${CONFIG}.${TASK}.${EMBODIMENT} to LeRobot repo ${REPO_ID}."
(
    cd "${ROBOTWIN_PATH}"
    conda run --no-capture-output -n "${STARVLA_ENV}" python "${CONVERTER}" \
        "${CONFIG}.${TASK}.${EMBODIMENT}" \
        --repo_id "${REPO_ID}" \
        --data_type RoboDojo \
        --max_episode "${NUM}"
)

LEROBOT_HOME="$(conda run --no-capture-output -n "${STARVLA_ENV}" python -c "from lerobot.datasets.lerobot_dataset import HF_LEROBOT_HOME; print(HF_LEROBOT_HOME)")"
DATASET_PATH="${LEROBOT_HOME}/${REPO_ID}"
[[ -d "${DATASET_PATH}" ]] || { echo "ERROR: converter did not create ${DATASET_PATH}" >&2; exit 1; }

log "DONE: ${DATASET_PATH}"
log "Use the XPolicyLab StarVLA adapter for training, or register this dataset explicitly in a compatible outer StarVLA config."
