# Surflo body-part conditioning on RoboTwin

This adapts `surflo_camera3_six_part_state_20k`'s architecture to Agilex:
three simultaneous cameras at 518×518; frozen official Surflo EMA backbone;
128 surface tokens, one pooled camera token, three per-view camera tokens,
and four separate body-part tokens (136 total); DiT-L; 16-step chunks;
20% state-conditioning dropout.

The 14-D normalized packed state is `[left joints (6), right joints (6),
left gripper (1), right gripper (1)]`. Part tokens are ordered left arm,
left gripper, right arm, right gripper. There are no base or torso groups.
Actions use the same packed joint/gripper order. The dedicated
`robotwin_surflo` data config selects only the current camera timestep.
Joint values use dataset min/max statistics; grippers use binary transforms.

Run from the repository root:

```bash
bash examples/simBenchmarks/Robotwin/train_files/run_surflo_body_parts_robotwin.sh
TASK=adjust_bottle bash examples/simBenchmarks/Robotwin/train_files/run_surflo_body_parts_robotwin.sh
TASK=stack_blocks_three bash examples/simBenchmarks/Robotwin/train_files/run_surflo_body_parts_robotwin.sh
```

Defaults: GPU 0, offline W&B, batch size 4 and two accumulation batches,
for an effective batch of 8. The budget is 20,000 training samples: 2,500
optimizer updates on one GPU. Changing batch size or process count adjusts
updates to preserve the sample budget; smaller explicit step limits are
allowed for smoke tests. The budget must be divisible by the global batch.
Training drops incomplete batches, and accumulation spans epoch boundaries.
Set
`CUDA_VISIBLE_DEVICES`, `PYTHON_BIN`, or `RUN_ID` as needed; extra arguments
are forwarded to the trainer, for example `--trainer.max_train_steps 2`.
The YAML contains this machine's Surflo and dataset paths.

This trains new robot-specific projections and a new action head using the
pretrained Surflo backbone. It does not load the incompatible 61-D/23-D
BEHAVIOR policy checkpoint. The policy currently has no language conditioning,
so these launch commands train one task at a time. Saved models and RoboTwin
normalization statistics are written under `results/Checkpoints/<RUN_ID>`.

A deterministic split holds out 10% of whole episodes (seed 42). For
`rotate_qrcode` this is 90 training episodes and 10 validation episodes.
Normalization statistics are computed only from training episodes and reused
for validation and deployment. Episode IDs and isolated statistics/index
caches are under `<RUN_ID>/data_split`; source dataset caches are not changed.

Validation runs every 250 optimizer updates using 20 DDIM steps on a fixed,
deterministically sampled subset of held-out data (up to 32 batches). It
reports actual squared action error as `validation_action_mse`, restores
training mode afterward, and preserves the training RNG state. The lowest
validation MSE checkpoint is saved in `best_model/pytorch_model.pt` with
`validation.json`. Periodic checkpoints are saved every 500 updates.
Validation MSE is not a simulator success rate.

The RoboTwin adapter now reads resolution, state inclusion, diffusion steps,
and execution horizon from server metadata. It packs raw simulator state as
`[left joints, right joints, left gripper, right gripper]`; the server applies
the checkpoint's training transforms once. Images use the same PIL resizing
as the loader. The model predicts 16 actions and executes 8 before replanning.
Serving in bfloat16 preserves float32 backbone weights for VGGT's camera head,
with visual computation under autocast and the action head in bfloat16.
Reset clears cached actions. Output joints/grippers are reordered for RoboTwin.

Serve a checkpoint:

```bash
bash examples/simBenchmarks/Robotwin/eval_files/run_policy_server.sh \
  results/Checkpoints/RUN_ID/best_model/pytorch_model.pt 0 5694
```

Compare best and periodic checkpoints on the same fixed simulator seeds:

```bash
ROBOTWIN_PYTHON=/export/home/haoliang/miniconda3/envs/RoboTwin/bin/python \
  bash examples/simBenchmarks/Robotwin/eval_files/eval_surflo_checkpoints.sh \
  rotate_qrcode /path/to/seeds.txt \
  results/Checkpoints/RUN_ID/best_model/pytorch_model.pt \
  results/Checkpoints/RUN_ID/checkpoints/steps_2500_pytorch_model.pt
```

The comparison launcher starts/stops a policy server for each checkpoint,
uses `demo_clean` by default, and records server/evaluation logs under
`results/robotwin_surflo_eval`. Set `ROBOTWIN_PATH`, `ROBOTWIN_TASK_CONFIG`,
`ROBOTWIN_SERVER_GPU`, and `ROBOTWIN_EVAL_GPU` for the simulator machine.
Seeds are whitespace-separated integers. The simulator writes success rates
in its result files; choose the final checkpoint by those rollout results.
A working SAPIEN rendering environment is required for simulator evaluation.

## Linear versus residual projection comparison

```bash
bash examples/simBenchmarks/Robotwin/train_files/run_surflo_projection_comparison.sh
```

This starts two independent, detached single-GPU runs: linear on GPU 0 and
residual on GPU 1. Set `TASK`, `LINEAR_GPU`, and `RESIDUAL_GPU` to override
these defaults. Both use the same seed, episode split, effective batch 8,
and default 20,000-sample budget (2,500 updates each). Instructions are fixed
to the literal task name for training and validation. Set `TRAIN_STEPS=10000`
and `SAVE_INTERVAL=5000` for fresh 10,000-step runs. A manifest with PIDs, commands,
checkpoint directories, and log paths is written under
`results/robotwin_projection_comparison/<task>_<timestamp>`.

The `framework.visual_adapter_type` option is `linear` by default. `residual`
adds `x + Linear(GELU(Linear(LayerNorm(x))))` after the existing 512→1024
visual projection. Both MLP layers are 1024→1024; the final layer's weights
and bias are initialized to zero. This adds 2,101,248 trainable parameters,
applies to surface and camera tokens, and leaves state tokens unchanged.
Shared model parameters have identical initial weights across variants;
the adapter is constructed after shared layers and model construction is
seeded before initialization. The adapter learning rate is 0.0003.

To continue both final checkpoints for another 10,000 optimizer updates:

```bash
bash examples/simBenchmarks/Robotwin/train_files/run_surflo_projection_continuation.sh \
  results/robotwin_projection_comparison/COMPARISON/manifest.json
```

This starts linear on GPU 0 and residual on GPU 1 in new run directories.
Each run loads its variant's final model weights, uses the literal task name
as `datasets.vla_data.instruction_override` in both training and validation,
and saves periodic checkpoints at continuation steps 5,000 and 10,000.
The effective batch remains 8, with 80,000 additional samples per variant.
The old checkpoints contain model weights only, so AdamW and the cosine
schedule start fresh. Step numbers in the new run count additional updates.
Set `TRAIN_STEPS`, `SAVE_INTERVAL`, `LINEAR_GPU`, or `RESIDUAL_GPU` to override.
SurfloDiT conditions on images and state; it currently does not encode text.

## Resumable training checkpoints

`trainer.save_training_state: true` saves model weights, AdamW state, the LR
scheduler, random states, optimizer step count, validation selection metric,
and the consumed data cursor using Accelerate. Periodic states are stored in
`checkpoints/steps_5000/` and `checkpoints/steps_10000/`; the adjacent `.pt`
files are inference exports. Best and final models also have a
`training_state/` directory. `complete.json` marks a fully written state.
RoboTwin samples are deterministic by epoch and index; the loader restarts
workers each epoch and skips already consumed batches on resume.

Resume the same run, keeping its original training configuration and GPU:

```bash
TASK=rotate_qrcode RUN_ID=YOUR_RUN_ID CUDA_VISIBLE_DEVICES=0 \
bash examples/simBenchmarks/Robotwin/train_files/run_surflo_body_parts_robotwin.sh \
  --config_yaml results/Checkpoints/YOUR_RUN_ID/config.full.yaml \
  --trainer.is_resume true
```

This selects the latest complete periodic training state. Alternatively pass
`--trainer.resume_checkpoint /absolute/path/to/checkpoints/steps_5000` to
choose a particular state. Restore the directory, rather than the inference
`.pt` file, to continue optimizer and scheduler state. Saved step numbers
remain absolute within the run. A regression check verifies that restoring
a stochastic training step reproduces the next update, LR, data batch, and RNG.
