#!/bin/bash
# Sequential RoboTwin eval for Phase 2 levels (2a-2d) on rotate_qrcode.
# Uses ABSOLUTE checkpoint paths (start_eval.sh cd's into eval_files/, so a
# relative -c path breaks its log redirection). Failure-isolated per level.
#
# Usage:  bash run_phase2_eval.sh [levels...]   (default: 2a 2b 2c 2d)

STARVLA=/export/home/haoliang/projects/starVLA
cd "${STARVLA}"

export ROBOTWIN_PATH=/export/home/haoliang/projects/RoboTwin
export ROBOTWIN_STARVLA_ENV=starVLA
export ROBOTWIN_ENV=RoboTwin

levels=("$@"); [ ${#levels[@]} -eq 0 ] && levels=(2a 2b 2c 2d)

TS=$(date +%Y%m%d_%H%M%S)
SUMMARY="${STARVLA}/results/Checkpoints/phase2_eval_summary_${TS}.log"
log() { echo "[$(date '+%H:%M:%S')] $*" | tee -a "${SUMMARY}"; }

log "Phase 2 eval START — levels: ${levels[*]}"
for lvl in "${levels[@]}"; do
    ckpt="${STARVLA}/results/Checkpoints/phase2_${lvl}_rotate_qrcode/final_model/pytorch_model.pt"
    if [ ! -f "${ckpt}" ]; then log "=== SKIP ${lvl} (no ckpt) ==="; continue; fi
    log "=== START ${lvl} ==="
    bash examples/simBenchmarks/Robotwin/eval_files/start_eval.sh \
        -m demo_clean -n "phase2_${lvl}" -c "${ckpt}" rotate_qrcode
    rc=$?
    # Pull the success-rate line from the freshest eval log for this level.
    el=$(ls -t results/Checkpoints/phase2_${lvl}_rotate_qrcode/final_model/robotwin_eval_logs/*/*_eval.log 2>/dev/null | head -1)
    sr=$(grep -aoE "Success rate:.*%|[0-9]+/[0-9]+ =>.*%" "${el}" 2>/dev/null | tail -1)
    log "=== DONE ${lvl} (rc=${rc})  ${sr:-<no success-rate line>} ==="
done
log "Phase 2 eval COMPLETE"
echo ""; echo "===== SUMMARY ====="; grep -E "START|DONE|SKIP|COMPLETE" "${SUMMARY}"
