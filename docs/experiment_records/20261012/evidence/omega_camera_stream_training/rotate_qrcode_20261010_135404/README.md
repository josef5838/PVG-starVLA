# rotate_qrcode camera-stream experiment

Launched 2026-10-10. Direct408 and fused32 train on separate H100 NVL GPUs; each GPU automatically starts its control after its primary run completes. Status below is the launch snapshot; use the status command for live progress.

| Variant | GPU | Launch status | Configuration |
|---|---|---|---|
| direct | 0 | running | [direct.yaml](direct.yaml) |
| fused | 1 | running | [fused.yaml](fused.yaml) |
| state_only | 0 | queued | [state_only.yaml](state_only.yaml) |
| current_only | 1 | queued | [current_only.yaml](current_only.yaml) |

All four variants: seed42, 20,000 optimizer updates, effective batch8, 160,000 sampled observation anchors, action horizon16. Primary observation histories contain8 consecutive frames per camera. Three independent frozen Ω calls produce408 camera/register tokens; fused conditioning compresses them to32. Learned chronological embeddings are present; temporal RoPE is disabled.

The state_only control provides current robot state/task with constant visual memory and no Ω feature-extraction calls. The current_only control repeats the current image across8 slots per camera, preserving408-token conditioning while removing history.

## Data and registered evaluation

80 training /10 validation /10 test episodes. Training-only normalization. Validation uses16 fixed anchors per episode and selects checkpoints by normalized right-arm joint/gripper MSE over valid future commands; full-action MSE is also logged. No test policy predictions are used for selection.

All100 demonstrations are right-arm trajectories. Sign models0/1/2/3 appear29/36/30/5 times; the random test split has no model3 episodes, so model3 results come from the separately registered fresh cohort. Initial orientation offsets span1.38°–39.67°. All100 source mappings and initial scenes were audited. Fifty stable unseen right-arm scenes were registered before training, without expert/policy-outcome filtering.

The paired rollout suite contains10 training,10 validation,10 test,50 fresh scenes. Evaluate horizons8 and4 separately with the same selected weights. Record official success, object orientation error, target-XY placement error, contact/lift/rotation/placement/release diagnostics and full videos. Source RGB has a slight blue tint relative to current rendering; record RGB discrepancies and verify grayscale/state/model/registered-pose agreement. See the protocol amendment and [source comparison](source_seed9_comparison.png).

## Verification and storage

24 tests passed. All four production-batch preflights passed with identical common initialization, image and target hashes. Full trainer smoke runs exercised optimization, validation, checkpoint selection and compact saving; a resume smoke restored complete state and continued training successfully. Disposable smoke model/optimizer files were removed; logs and [integration audit](trainer_integration_audit.json) remain.

Keep permanent best/final/update10000 weights in each run directory. Save the latest complete optimizer/scheduler/RNG/cursor state every1000 updates under `/tmp/omega_rotate_qrcode_training_states/20261010_135404/<variant>/latest`, linked from the run's `latest_training_state`. This temporary state can be cleared on reboot; it is separate from permanent policy weights. Existing experiments were not removed.

[Protocol](../../../docs/omega_rotate_qrcode_protocol_20261010.md) · [Manifest](manifest.json) · [Geometry registration](geometry_registration.json) · [Split audit](split_validation_audit.json)

## Commands

From `/export/home/haoliang/projects/PVG-starVLA-snapshot`:

```bash
/export/home/haoliang/miniconda3/envs/starVLA/bin/python tools/omega_camera_stream_status.py --manifest /export/home/haoliang/projects/PVG-starVLA-snapshot/results/omega_camera_stream_training/rotate_qrcode_20261010_135404/manifest.json
```

After all runs finish, prepare and launch evaluation (a separate stage):

```bash
/export/home/haoliang/miniconda3/envs/starVLA/bin/python tools/prepare_rotate_qrcode_evaluation.py --training-manifest /export/home/haoliang/projects/PVG-starVLA-snapshot/results/omega_camera_stream_training/rotate_qrcode_20261010_135404/manifest.json
/export/home/haoliang/miniconda3/envs/starVLA/bin/python tools/run_omega_stream_evaluation.py --manifest <evaluation-manifest.json> --sequential-per-gpu
```

An ordinary visual encoder baseline and additional training seeds remain follow-up studies. These four variants test fusion/compression, dependence on images, and observation history; they do not alone prove geometry-specific superiority over ordinary visual representations.
