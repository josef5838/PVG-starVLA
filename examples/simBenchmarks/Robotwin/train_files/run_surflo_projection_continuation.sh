#!/usr/bin/env bash
set -euo pipefail
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
cd "$repo_root"
python_bin="${PYTHON_BIN:-/export/home/haoliang/miniconda3/envs/starVLA/bin/python}"
exec "$python_bin" - "$@" <<'PY'
import datetime
import json
import os
from pathlib import Path
import subprocess
import sys

if len(sys.argv) != 2:
    raise SystemExit('Usage: run_surflo_projection_continuation.sh <source_comparison_manifest>')
root = Path.cwd()
source_path = Path(sys.argv[1]).resolve()
source = json.loads(source_path.read_text())
task = source['task']
gpus = [os.environ.get('LINEAR_GPU', '0'), os.environ.get('RESIDUAL_GPU', '1')]
if gpus[0] == gpus[1]:
    raise ValueError('The comparison requires two different GPUs')
steps = int(os.environ.get('TRAIN_STEPS', '10000'))
save_interval = int(os.environ.get('SAVE_INTERVAL', '5000'))
if steps < 1 or save_interval < 1:
    raise ValueError('Training steps and save interval must be positive')
checkpoints = {}
for variant in ('linear', 'residual'):
    run = next(run for run in source['runs'] if run['variant'] == variant)
    checkpoint = Path(run['checkpoint_dir']) / 'final_model/pytorch_model.pt'
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    checkpoints[variant] = checkpoint
stamp = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%d_%H%M%S')
directory = root / 'results/robotwin_projection_comparison' / f'{task}_continue_{stamp}'
directory.mkdir(parents=True, exist_ok=False)
manifest = {'task': task, 'instruction_override': task, 'seed': source['seed'],
            'source_manifest': str(source_path), 'effective_batch_per_variant': 8,
            'additional_optimizer_updates_per_variant': steps,
            'sample_budget_per_variant': steps * 8, 'save_interval': save_interval,
            'optimizer_state': 'fresh AdamW and cosine schedule; model weights continued', 'runs': []}
launcher = root / 'examples/simBenchmarks/Robotwin/train_files/run_surflo_body_parts_robotwin.sh'
for variant, gpu in zip(('linear', 'residual'), gpus):
    run_id = f'surflo_robotwin_{task}_{variant}_task_instruction_continue_{steps}steps_{stamp}'
    log = directory / f'{variant}.log'
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu, TASK=task, RUN_ID=run_id,
               PYTHONUNBUFFERED='1', OMP_NUM_THREADS='4', NO_ALBUMENTATIONS_UPDATE='1')
    command = ['bash', str(launcher), '--framework.visual_adapter_type', variant,
               '--seed', str(source['seed']),
               '--datasets.vla_data.instruction_override', task,
               '--trainer.pretrained_checkpoint', str(checkpoints[variant]),
               '--trainer.max_train_steps', str(steps),
               '--trainer.sample_budget', str(steps * 8),
               '--trainer.save_interval', str(save_interval)]
    with log.open('w') as stream:
        process = subprocess.Popen(command, cwd=root, env=env, stdin=subprocess.DEVNULL,
                                   stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
    manifest['runs'].append({'variant': variant, 'gpu': gpu, 'pid': process.pid,
                            'run_id': run_id, 'log': str(log), 'command': command,
                            'source_checkpoint': str(checkpoints[variant]),
                            'checkpoint_dir': str(root / 'results/Checkpoints' / run_id)})
path = directory / 'manifest.json'
path.write_text(json.dumps(manifest, indent=2) + '\n')
print(json.dumps({'manifest': str(path), 'runs': manifest['runs']}, indent=2))
PY
