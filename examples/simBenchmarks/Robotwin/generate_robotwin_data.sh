#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# RoboTwin → LeRobot data generation pipeline (end to end)
#
# Runs all four stages with the correct conda env per stage and relocates the
# output into the starVLA Datasets folder (the convert tool only ever writes to
# the LeRobot cache, which is the gotcha this script hides):
#
#   1. collect   [RoboTwin env]  sim rollouts      → RoboTwin/data/<task>/<config>
#   2. process   [starVLA env]   → ALOHA HDF5      → GO1/processed_data/<task>-<config>-<num>
#   3. convert   [starVLA env]   → LeRobot format  → $HF_LEROBOT_HOME/<task>
#   4. relocate  move cache → Datasets/<task>, drop in modality.json
#   5. register  add `robotwin_<task>` to the data-mix registry (idempotent)
#
# Usage:
#   bash generate_robotwin_data.sh --task rotate_qrcode
#   bash generate_robotwin_data.sh --task adjust_bottle --num 50 --gpu 0
#   bash generate_robotwin_data.sh --task X --skip-collect   # reuse existing sim data
#
# After it finishes, train with:
#   --datasets.vla_data.data_mix robotwin_<task>
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

# ── Defaults ────────────────────────────────────────────────────────────────
TASK=""
CONFIG="demo_clean"
NUM=50
GPU=0
ROBOTWIN_ENV="RoboTwin"
STARVLA_ENV="starVLA"
ROBOTWIN_PATH="/export/home/haoliang/projects/RoboTwin"
STARVLA_PATH="/export/home/haoliang/projects/starVLA"
DATASETS_ROOT=""   # default derived from STARVLA_PATH below
MODALITY_JSON=""   # default derived from STARVLA_PATH below
SKIP_COLLECT=false
SKIP_PROCESS=false
SKIP_CONVERT=false
NO_REGISTER=false

usage() {
    sed -n '3,21p' "$0" | sed 's/^# \{0,1\}//'
    exit "${1:-0}"
}

# ── Arg parsing ─────────────────────────────────────────────────────────────
while (( $# > 0 )); do
    case "$1" in
        --task)          TASK="$2"; shift 2 ;;
        --config)        CONFIG="$2"; shift 2 ;;
        --num)           NUM="$2"; shift 2 ;;
        --gpu)           GPU="$2"; shift 2 ;;
        --robotwin-env)  ROBOTWIN_ENV="$2"; shift 2 ;;
        --starvla-env)   STARVLA_ENV="$2"; shift 2 ;;
        --robotwin-path) ROBOTWIN_PATH="$2"; shift 2 ;;
        --starvla-path)  STARVLA_PATH="$2"; shift 2 ;;
        --datasets-root) DATASETS_ROOT="$2"; shift 2 ;;
        --modality)      MODALITY_JSON="$2"; shift 2 ;;
        --skip-collect)  SKIP_COLLECT=true; shift ;;
        --skip-process)  SKIP_PROCESS=true; shift ;;
        --skip-convert)  SKIP_CONVERT=true; shift ;;
        --no-register)   NO_REGISTER=true; shift ;;
        -h|--help)       usage 0 ;;
        *) echo "Unknown option: $1" >&2; usage 1 ;;
    esac
done

[[ -z "${TASK}" ]] && { echo "ERROR: --task is required" >&2; usage 1; }

DATASETS_ROOT="${DATASETS_ROOT:-${STARVLA_PATH}/playground/Datasets/RoboTwin}"
MODALITY_JSON="${MODALITY_JSON:-${STARVLA_PATH}/examples/simBenchmarks/Robotwin/train_files/modality.json}"
GO1_DIR="${ROBOTWIN_PATH}/policy/GO1"
PROCESSED_NAME="${TASK}-${CONFIG}-${NUM}"
MIXTURES_PY="${STARVLA_PATH}/starVLA/dataloader/gr00t_lerobot/mixtures.py"

log() { echo -e "\033[36m[gen-data]\033[0m $*"; }

log "task=${TASK}  config=${CONFIG}  num=${NUM}  gpu=${GPU}"
log "robotwin_env=${ROBOTWIN_ENV}  starvla_env=${STARVLA_ENV}"
log "datasets_root=${DATASETS_ROOT}"

# ── Stage 1: collect (RoboTwin env) ──────────────────────────────────────────
# collect_data.py reads episode_num from the task-config YAML, so we patch it to
# NUM for the duration of collection and restore it afterward (no permanent edit).
if ! ${SKIP_COLLECT}; then
    log "── Stage 1/5: collecting ${NUM} sim episodes ──"
    task_cfg_yml="${ROBOTWIN_PATH}/task_config/${CONFIG}.yml"
    [[ -f "${task_cfg_yml}" ]] || { echo "Missing task config: ${task_cfg_yml}" >&2; exit 1; }

    cfg_backup="$(mktemp)"
    cp "${task_cfg_yml}" "${cfg_backup}"
    restore_cfg() { cp "${cfg_backup}" "${task_cfg_yml}"; rm -f "${cfg_backup}"; }
    trap restore_cfg EXIT
    sed -i "s/^episode_num:.*/episode_num: ${NUM}/" "${task_cfg_yml}"

    ( cd "${ROBOTWIN_PATH}" && conda run --no-capture-output -n "${ROBOTWIN_ENV}" \
        bash collect_data.sh "${TASK}" "${CONFIG}" "${GPU}" )

    restore_cfg
    trap - EXIT
else
    log "── Stage 1/5: SKIPPED (--skip-collect) ──"
fi

# ── Stage 2: process → ALOHA HDF5 (starVLA env) ──────────────────────────────
if ! ${SKIP_PROCESS}; then
    log "── Stage 2/5: processing → ALOHA HDF5 ──"
    ( cd "${GO1_DIR}" && conda run --no-capture-output -n "${STARVLA_ENV}" \
        python scripts/process_data.py "${TASK}" "${CONFIG}" "${NUM}" )
else
    log "── Stage 2/5: SKIPPED (--skip-process) ──"
fi

# ── Stage 3: convert → LeRobot (starVLA env, writes to HF cache) ─────────────
if ! ${SKIP_CONVERT}; then
    log "── Stage 3/5: converting → LeRobot format ──"
    ( cd "${GO1_DIR}" && conda run --no-capture-output -n "${STARVLA_ENV}" \
        python scripts/convert_aloha_data_to_lerobot_robotwin.py \
            --raw_dir "processed_data/${PROCESSED_NAME}" \
            --repo_id "${TASK}" )
else
    log "── Stage 3/5: SKIPPED (--skip-convert) ──"
fi

# ── Stage 4: relocate cache → Datasets + modality.json ───────────────────────
log "── Stage 4/5: relocating to Datasets folder ──"
HF_HOME_LEROBOT="$(conda run --no-capture-output -n "${STARVLA_ENV}" python -c \
    'from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME; print(HF_LEROBOT_HOME)')"
SRC="${HF_HOME_LEROBOT}/${TASK}"
DST="${DATASETS_ROOT}/${TASK}"

[[ -d "${SRC}" ]] || { echo "ERROR: expected converted dataset at ${SRC}" >&2; exit 1; }
[[ -f "${MODALITY_JSON}" ]] || { echo "ERROR: modality.json not found: ${MODALITY_JSON}" >&2; exit 1; }

rm -rf "${DST}"
mkdir -p "${DST}/meta"
mv "${SRC}/data" "${DST}/data"
mv "${SRC}/meta/"* "${DST}/meta/"
cp "${MODALITY_JSON}" "${DST}/meta/modality.json"
rmdir "${SRC}/meta" "${SRC}" 2>/dev/null || true
log "dataset placed at ${DST}"

# ── Stage 5: register data mix (idempotent) ──────────────────────────────────
if ! ${NO_REGISTER}; then
    log "── Stage 5/5: registering data mix 'robotwin_${TASK}' ──"
    MIXTURES_PY="${MIXTURES_PY}" TASK="${TASK}" python3 - <<'PY'
import os, re
path = os.environ["MIXTURES_PY"]
task = os.environ["TASK"]
key = f"robotwin_{task}"
src = open(path).read()
if f'"{key}"' in src:
    print(f"  '{key}' already registered, skipping.")
else:
    block = f'    "{key}": [\n        ("{task}", 1.0, "robotwin"),\n    ],\n'
    src = src.replace("DATASET_NAMED_MIXTURES = {\n",
                      "DATASET_NAMED_MIXTURES = {\n" + block, 1)
    open(path, "w").write(src)
    print(f"  registered '{key}'.")
PY
else
    log "── Stage 5/5: SKIPPED (--no-register) ──"
fi

# ── Summary ──────────────────────────────────────────────────────────────────
log "DONE."
echo ""
echo "  Dataset : ${DST}"
echo "  Episodes: $(ls "${DST}/data/chunk-000/" 2>/dev/null | wc -l)"
echo "  Train with:  --datasets.vla_data.data_mix robotwin_${TASK}"
echo ""
echo "  (stats_gr00t.json & steps_data_index.pkl are auto-built on first training run)"
