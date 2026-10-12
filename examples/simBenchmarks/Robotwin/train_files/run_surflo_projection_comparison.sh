#!/usr/bin/env bash
set -euo pipefail
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
cd "$repo_root"
python_bin="${PYTHON_BIN:-/export/home/haoliang/miniconda3/envs/starVLA/bin/python}"
exec "$python_bin" - <<'PY'
import datetime
import json
import os
from pathlib import Path
import subprocess

root = Path.cwd()
task = os.environ.get('TASK', 'rotate_qrcode')
if task not in {'adjust_bottle', 'rotate_qrcode', 'stack_blocks_three'}:
    raise ValueError(f'Unsupported TASK: {task}')
gpus = [os.environ.get('LINEAR_GPU', '0'), os.environ.get('RESIDUAL_GPU', '1')]
if gpus[0] == gpus[1]:
    raise ValueError('The comparison requires two different GPUs')
stamp = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%d_%H%M%S')
steps = int(os.environ.get('TRAIN_STEPS', '2500'))
save_interval = int(os.environ.get('SAVE_INTERVAL', '500'))
if steps < 1 or save_interval < 1:
    raise ValueError('Training steps and save interval must be positive')
directory = root / 'results/robotwin_projection_comparison' / f'{task}_{stamp}'
directory.mkdir(parents=True, exist_ok=False)
manifest = {'task': task, 'seed': 42, 'sample_budget_per_variant': steps * 8,
            'effective_batch_per_variant': 8, 'optimizer_updates_per_variant': steps,
            'initialization': 'fresh policy layers; pretrained frozen Surflo visual backbone',
            'instruction_override': task, 'save_interval': save_interval,
            'training_resumable_checkpoints': True,
            'runs': []}
launcher = root / 'examples/simBenchmarks/Robotwin/train_files/run_surflo_body_parts_robotwin.sh'
for variant, gpu in zip(('linear', 'residual'), gpus):
    run_id = f'surflo_robotwin_{task}_{variant}_fresh_{steps}steps_{stamp}'
    log = directory / f'{variant}.log'
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu, TASK=task, RUN_ID=run_id,
               PYTHONUNBUFFERED='1', OMP_NUM_THREADS='4', NO_ALBUMENTATIONS_UPDATE='1')
    command = ['bash', str(launcher), '--framework.visual_adapter_type', variant,
               '--datasets.vla_data.instruction_override', task,
               '--trainer.max_train_steps', str(steps),
               '--trainer.sample_budget', str(steps * 8),
               '--trainer.save_interval', str(save_interval),
               '--trainer.save_training_state', 'true']
    with log.open('w') as stream:
        process = subprocess.Popen(command, cwd=root, env=env, stdin=subprocess.DEVNULL,
                                   stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
    manifest['runs'].append({'variant': variant, 'gpu': gpu, 'pid': process.pid,
                            'run_id': run_id, 'log': str(log), 'command': command,
                            'checkpoint_dir': str(root / 'results/Checkpoints' / run_id)})
path = directory / 'manifest.json'
path.write_text(json.dumps(manifest, indent=2))
print(json.dumps({'manifest': str(path), 'runs': manifest['runs']}, indent=2))
PY
