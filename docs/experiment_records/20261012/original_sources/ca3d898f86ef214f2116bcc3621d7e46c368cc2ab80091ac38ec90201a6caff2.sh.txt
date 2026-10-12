#!/usr/bin/env bash
# Run inside a one-GPU allocation; executes both full-capacity variants in order.
set -euo pipefail
bundle="$(realpath "${1:?Usage: bash tools/run_omega512_cluster_smoke.sh BUNDLE [--trainer]}")"
repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo"
source "$bundle/configured/environment.sh"
python_bin="${STARVLA_PYTHON:-python}"
report="$bundle/smoke_$(date -u +%Y%m%d_%H%M%S)"
mkdir -p "$report"
nvidia-smi > "$report/gpu.txt"
"$python_bin" -c 'import torch; assert torch.cuda.is_available(); print(torch.__version__, torch.cuda.get_device_name(), torch.cuda.is_bf16_supported())' > "$report/runtime.txt"
"$python_bin" tools/omega512_cluster_bundle.py verify --bundle "$bundle" --configured > "$report/verify.json"
for test in test_omega512_policy.py test_omega512_migration.py test_robotwin_surflo_training.py test_training_resume.py test_config_overrides.py; do
    CUDA_VISIBLE_DEVICES='' "$python_bin" -m unittest discover -s tests -p "$test" -v > "$report/$test.log" 2>&1
done
for variant in dense state_query_local; do
    config="$bundle/configured/${variant}_smoke.yaml"
    "$python_bin" tools/preflight_omega512_policy.py --config "$config" --out "$report/${variant}_preflight.json" \
        --save-policy "$report/${variant}_policy" > "$report/${variant}_preflight.log" 2>&1
    checkpoint="$report/${variant}_policy/checkpoints/preflight_pytorch_model.pt"
    "$python_bin" tools/smoke_omega512_serving.py --checkpoint "$checkpoint" --config "$config" --out "$report/${variant}_serving.json" > "$report/${variant}_serving.log" 2>&1
    "$python_bin" tools/smoke_omega512_serving.py --checkpoint "$checkpoint" --config "$config" --out "$report/${variant}_bf16_serving.json" --use-bf16 > "$report/${variant}_bf16_serving.log" 2>&1
    if [[ "${2:-}" == '--trainer' ]]; then
        # The horizon stays at 16 updates across both executions: resume must
        # restore the same scheduler horizon, optimizer, RNG and data cursor.
        output="$("$python_bin" -c 'from omegaconf import OmegaConf; import sys; print(OmegaConf.load(sys.argv[1]).output_dir)' "$config")"
        if [[ -e "$output" ]]; then
            echo "Refusing to overwrite an existing smoke run: $output" >&2
            exit 1
        fi
        "$python_bin" starVLA/training/train_starvla.py --config_yaml "$config" > "$report/${variant}_trainer.log" 2>&1
        "$python_bin" starVLA/training/train_starvla.py --config_yaml "$config" \
            --trainer.resume_checkpoint "$output/checkpoints/steps_8" \
            --run_id "${variant}_smoke_resume_$(date -u +%Y%m%d_%H%M%S)" \
            --output_dir "${output}_resume" \
            --datasets.vla_data.statistics_cache_dir "${output}_resume/data_split" > "$report/${variant}_resume.log" 2>&1
        "$python_bin" - "$output" "${output}_resume" <<'PY'
import json, pathlib, sys, torch
a,b=map(pathlib.Path,sys.argv[1:])
for p in (a,b):
    assert json.loads((p/'final_model/training_state/complete.json').read_text())['completed_steps']==16
original=torch.load(a/'final_model/pytorch_model.pt',map_location='cpu',weights_only=True)
resumed=torch.load(b/'final_model/pytorch_model.pt',map_location='cpu',weights_only=True)
assert original.keys()==resumed.keys()
worst=max(float((original[k].float()-resumed[k].float()).abs().max()) for k in original)
print('Next-run final weight difference:',worst)
# Allow small GPU reduction differences; detect a lost optimizer/data/RNG cursor.
for key in original:
    torch.testing.assert_close(original[key],resumed[key],rtol=1e-4,atol=1e-5,msg=key)
PY
    fi
done
echo "Passed Omega cluster smoke checks: $report"
