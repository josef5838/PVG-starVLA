# VGGT-Omega on BEHAVIOR-1K 2026

This path uses offline artifact transfer between machines:

1. Collect, replay, and validate demonstrations on the RTX machine.
2. Transfer the prepared LeRobot v3 dataset to the H100 machine.
3. Train on the H100 machine.
4. Transfer the complete StarVLA run directory back to the RTX machine.
5. Run policy inference and OmniGibson evaluation on the RTX machine over localhost.

The adapter follows the official R1Pro contract: head and two wrist RGB cameras,
61-dimensional proprioception, 23-dimensional actions, and a 16-step action
horizon. The policy uses only challenge-allowed onboard RGB and proprioception.

## 1. RTX environment

Use BEHAVIOR-1K `v3.9.2` for challenge replay and evaluation. Keep a separate
checkout of current `main` if you need the latest JoyLo collection fixes.

```bash
git clone -b v3.9.2 https://github.com/StanfordVL/BEHAVIOR-1K.git BEHAVIOR-1K-eval
cd BEHAVIOR-1K-eval
./setup.sh --new-env --omnigibson --bddl --joylo --dataset --eval
conda activate behavior
vulkaninfo --summary
```

The directories `behavior-1k-assets`, `omnigibson-robot-assets`, and
`2026-challenge-task-instances` must exist under `BEHAVIOR-1K-eval/datasets`.

## 2. Collect a five-demo smoke dataset

Start with `turning_on_radio` (task ID 0). Record each accepted demonstration
to a separate HDF5 file. The official replay tool infers the task ID from
`demo_id // 10000`, so preserve this naming scheme.

```bash
export B1K_ROOT=/path/to/BEHAVIOR-1K-eval
export SAMPLE_ROOT=/data/behavior-smoke
mkdir -p "$SAMPLE_ROOT/2026-challenge-rawdata/task-0000"
cd "$B1K_ROOT"
conda activate behavior

# Repeat with demo IDs 0, 1, 2, 3, and 4.
python joylo/scripts/launch_og.py \
  --task-name turning_on_radio \
  --recording-path "$SAMPLE_ROOT/2026-challenge-rawdata/task-0000/episode_00000000.hdf5"
```

In another terminal:

```bash
cd "$B1K_ROOT"
conda activate behavior
python joylo/scripts/run_joylo.py
```

Replay every candidate with QA and discard failed, truncated, or visibly
incorrect demonstrations:

```bash
python joylo/scripts/replay_data.py \
  "$SAMPLE_ROOT/2026-challenge-rawdata/task-0000/episode_00000000.hdf5" \
  --task turning_on_radio --qa
```

Convert the first accepted recording to LeRobot v3, then append the rest:

```bash
python OmniGibson/scripts/learning/replay_obs.py \
  --data_folder "$SAMPLE_ROOT" --demo_id 0 --output_format lerobot \
  --lerobot_root_dir "$SAMPLE_ROOT/lerobot" \
  --lerobot_repo_id behavior-1k-2026

for demo_id in 1 2 3 4; do
  python OmniGibson/scripts/learning/replay_obs.py \
    --data_folder "$SAMPLE_ROOT" --demo_id "$demo_id" --output_format lerobot \
    --lerobot_root_dir "$SAMPLE_ROOT/lerobot" \
    --lerobot_repo_id behavior-1k-2026 --resume_lerobot
done
```

Download the official natural-language task descriptions. StarVLA uses these
instead of training on terse identifiers such as `turning_on_radio`:

```bash
huggingface-cli download behavior-1k/2026-challenge-demos \
  --repo-type dataset \
  --local-dir "$SAMPLE_ROOT/metadata" \
  --include meta/tasks.jsonl
```

Prepare and checksum the generated dataset from the StarVLA checkout:

```bash
export STARVLA_ROOT=/path/to/starVLA
export DATASET="$SAMPLE_ROOT/lerobot/behavior-1k-2026"
cd "$STARVLA_ROOT"
conda activate starVLA
python examples/simBenchmarks/Behavior/prepare_behavior2026_dataset.py \
  --dataset-root "$DATASET" \
  --tasks-jsonl "$SAMPLE_ROOT/metadata/meta/tasks.jsonl" \
  --prune-unused-videos
```

The preparation command validates all required shapes, installs
`meta/modality.json`, converts official statistics into StarVLA's cache format,
and writes `transfer.sha256`. Do not precompute VGGT tokens: one RGB timestep
expands to thousands of 2048-dimensional tokens, making a full cache much
larger than the source video dataset. `--prune-unused-videos` removes replayed
depth and external-camera videos only after QA; the immutable raw HDF5 remains
the recovery source.

### Download one official task instead of the full dataset

For a compact smoke-training dataset, download only the corrected `turning_on_radio`
episodes (task ID 0), its Parquet frames, and its three allowed RGB streams. This
does not download depth or external-camera videos. The `v3.0` dataset tag currently
resolves to the corrected dataset revision required by BEHAVIOR-1K `v3.9.2`.

```bash
cd /path/to/starVLA
conda activate starVLA
export SPARSE_SOURCE=/data/starvla-datasets/behavior-1k-2026-turning_on_radio-source
export DATASET=/data/starvla-datasets/behavior-1k-2026

# This downloads metadata first and prints the exact file count, without data/videos.
python examples/simBenchmarks/Behavior/download_behavior2026_task.py \
  --output-root "$SPARSE_SOURCE" --task turning_on_radio --dry-run

# Fetch only task 0, then materialize a valid one-task training dataset with hardlinks.
python examples/simBenchmarks/Behavior/download_behavior2026_task.py \
  --output-root "$SPARSE_SOURCE" --task turning_on_radio
python examples/simBenchmarks/Behavior/prepare_behavior2026_task_subset.py \
  --source-root "$SPARSE_SOURCE" --output-root "$DATASET" \
  --task turning_on_radio --file-mode hardlink
python examples/simBenchmarks/Behavior/prepare_behavior2026_dataset.py \
  --dataset-root "$DATASET"
```

The source and final dataset must be on the same filesystem for `--file-mode
hardlink`, so the final step does not duplicate the downloaded Parquet or RGB files.
Any checkpoint trained before the v3.9.2 velocity correction should be retrained
from this refreshed dataset; `meta/stats.json` and StarVLA's normalization cache are
both updated.

## 3. Stage the dataset and train on H100

Use removable storage or an object-store/staging volume that each machine accesses
separately. The RTX and H100 machines do not need routes, credentials, or live
connections to each other. In these examples, `/mnt/offline-transfer` is the
mounted transfer medium.

```bash
# RTX machine
export TRANSFER_MEDIA=/mnt/offline-transfer
mkdir -p "$TRANSFER_MEDIA/behavior-1k-2026"
rsync -a --info=progress2 "$DATASET/" \
  "$TRANSFER_MEDIA/behavior-1k-2026/"

# H100 machine, after attaching or downloading the staged artifact
export TRANSFER_MEDIA=/mnt/offline-transfer
mkdir -p /data/starvla-datasets/behavior-1k-2026
rsync -a --info=progress2 "$TRANSFER_MEDIA/behavior-1k-2026/" \
  /data/starvla-datasets/behavior-1k-2026/
cd /data/starvla-datasets/behavior-1k-2026
sha256sum -c transfer.sha256
```

The H100 directory layout must be:

```text
/data/starvla-datasets/
└── behavior-1k-2026/
    ├── data/
    ├── meta/
    ├── videos/
    └── transfer.sha256
```

Run a short overfit/smoke job before a long training run:

```bash
cd /path/to/starVLA
conda activate starVLA
export BEHAVIOR_DATA_PARENT=/data/starvla-datasets
export RUN_ID=behavior2026_radio_smoke
export MAX_TRAIN_STEPS=500
export BATCH_SIZE=4
export WANDB_MODE=offline
bash examples/simBenchmarks/Behavior/train_files/run_vggt_omega_behavior2026.sh
```

Check that the first batch has 3 images, state shape `(1, 61)`, action shape
`(16, 23)`, finite loss, and a decreasing short-run loss. Then increase the
step count. Batch size 4 is a conservative H100 starting point; increase it
only after measuring peak memory.

For SurfloDiT training on another machine, clone the StarVLA development fork
and the matching Surflo fork. Transfer `behavior-1k-2026` and the released
`surflo_v0.pt` separately; neither belongs in Git. The launcher overrides the
machine-specific paths in the base YAML:

```bash
cd /path/to/PVG-starVLA
export SURFLO_ROOT=/path/to/Surflo
export BEHAVIOR_DATA_ROOT=/path/to/Datasets  # contains behavior-1k-2026/
export PYTHON_BIN=/path/to/starVLA-env/bin/python
CUDA_VISIBLE_DEVICES=0 bash examples/simBenchmarks/Behavior/train_files/run_surflo_dit_turn_on_radio_2026.sh \
  --run_id surflo_camera3_six_part_state_20k \
  --framework.surflo.append_per_view_camera_tokens true \
  --framework.state_conditioning parts \
  --framework.action_model.num_condition_tokens 138 \
  --trainer.max_train_steps 20000 \
  --trainer.save_interval 10000
```

Set `NUM_PROCESSES` and make `accelerate` available in the same environment
for multi-GPU training. Use a distinct `run_id` for every experiment.

## 4. Return the checkpoint bundle

For SurfloDiT checkpoints, use [SURFLO_BUNDLE.md](SURFLO_BUNDLE.md) instead of
the VGGT-Omega run-directory copy below. It handles frozen versus trained
Surflo backbones and bundles the architecture code needed on RTX.

Transfer the whole run directory, not only the `.pt` file. Loading requires
`config.yaml` and `dataset_statistics.json` two levels above the checkpoint.

```bash
# H100 machine
export RUN_DIR=/path/to/starVLA/results/Checkpoints/behavior2026_radio_smoke
export TRANSFER_MEDIA=/mnt/offline-transfer
cd "$RUN_DIR"
find . -type f ! -name checkpoint.sha256 -print0 | sort -z | \
  xargs -0 sha256sum > checkpoint.sha256
mkdir -p "$TRANSFER_MEDIA/behavior2026_radio_smoke"
rsync -a --info=progress2 "$RUN_DIR/" \
  "$TRANSFER_MEDIA/behavior2026_radio_smoke/"

# RTX machine, after attaching or downloading the staged artifact
export TRANSFER_MEDIA=/mnt/offline-transfer
mkdir -p /data/starvla-runs/behavior2026_radio_smoke
rsync -a --info=progress2 "$TRANSFER_MEDIA/behavior2026_radio_smoke/" \
  /data/starvla-runs/behavior2026_radio_smoke/
cd /data/starvla-runs/behavior2026_radio_smoke
sha256sum -c checkpoint.sha256
```

The RTX StarVLA environment must also have the VGGT-Omega code and the CLIP
tokenizer/config cached. Model weights themselves are present in the StarVLA
checkpoint.

## 5. Evaluate locally on RTX

Terminal 1, in the StarVLA environment:

```bash
cd /path/to/starVLA
conda activate starVLA
CUDA_VISIBLE_DEVICES=0 python examples/simBenchmarks/Behavior/serve_vggt_omega_2026.py \
  --ckpt-path /data/starvla-runs/behavior2026_radio_smoke/checkpoints/steps_500_pytorch_model.pt \
  --tasks-jsonl "$DATASET/meta/tasks.jsonl" \
  --host 127.0.0.1 --port 8000 --execute-horizon 8 --bf16
```

Terminal 2, in the pinned BEHAVIOR environment:

```bash
cd "$B1K_ROOT/OmniGibson"
conda activate behavior
CUDA_VISIBLE_DEVICES=0 python -m omnigibson.eval.eval \
  --task-name turning_on_radio \
  --robot-config omnigibson/eval/r1pro.yaml \
  --mode train --instance-indices 0 \
  --host 127.0.0.1 --port 8000 \
  --max-steps 500 --output-dir /data/b1k-eval/radio-smoke --write-video
```

After this succeeds, evaluate public-test instances, then expand collection to
more demonstrations and tasks. Keep raw HDF5 recordings immutable; regenerate
the LeRobot dataset and transfer manifest whenever accepted episodes change.
