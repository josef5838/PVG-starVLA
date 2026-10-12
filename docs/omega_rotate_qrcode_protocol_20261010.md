# rotate_qrcode: camera histories and action conditioning

Registered 2026-10-10, before training. Primary comparison: direct408 versus fused32. Controls: state/task only and current-image-only direct408. All models train from scratch with seed 42 and matched common initial parameters, training sample order, action diffusion objective, optimizer, and sample budget. This single seed does not establish benchmark-wide significance.

## Data and split

Use the local RoboTwin `rotate_qrcode` dataset: 100 episodes, 8341 frames, nominal 15 Hz, three embedded 640×480 RGB cameras. Head, left wrist, right wrist order is fixed. Audit actual parquet joint commands and raw HDF5 mappings. All existing demonstrations use the right arm; left joints and left gripper are constant. Evaluate right-arm scenes as the primary supported distribution. Left-arm performance is not an in-distribution claim.

Use a deterministic episode split with NumPy default_rng(42): first ten shuffled IDs are validation, next ten test, remaining eighty training. Save identities before any optimization. Compute normalization statistics from training episodes only, including validation/test loaders. Test episodes never select checkpoints or stopping time. Task text is overridden by the learned `rotate_qrcode` task embedding; this is not a language-generalization experiment.

An anchor t supplies camera-major chronological histories t−7,…,t, stride one, and current robot state. Clamp missing past indices to frame zero and future targets to the final episode command; never cross episode boundaries. Converted state[t] equals original command[t], while converted action[t] equals original command[t+1]; verify every episode against source data before launch. Predict sixteen 14-D absolute joint/gripper commands. Decoder packing is left6, right6, left gripper, right gripper. Padding remains in the training objective; selection metrics exclude nonexistent future targets.

## Registered variants

| Variant | Observation content | Action visual memory |
|---|---|---|
| direct | Three independent eight-image Ω calls | 408 camera/register tokens |
| fused | Identical histories/features | Two width-512 transformer layers and 32 learned readout queries |
| state_only | Current robot state and task; Ω is bypassed | 408 constant zero memory tokens, matching direct decoder shape |
| current_only | Latest image repeated across eight slots in each physical camera's call | 408 tokens, matching direct token count and positional embeddings |

The current_only control is trained using repeated images, rather than applying repetition only at test time. It preserves token budget and embeddings while removing historical observations. The state_only model retains decoder cross-attention modules but their memory carries no image information. Image-dependent projection/embedding parameters are frozen, and preflight must verify zero Ω forward calls and exact image invariance. Ω may be loaded for common initialization/serialization but never extracts baseline features.

Ω is frozen at the verified official checkpoint, final feature layer 23, official balanced-512 preprocessing. Each image contributes one camera token and sixteen scene/register tokens. Patch tokens participate inside Ω but are discarded from action conditioning. Both primary models use the existing 2048→512 projection, learned camera/time/token-type embeddings, 512→1024 output projection, four state-part tokens and one task token. No temporal RoPE is added. Decoder remains width 1024, depth 24, 16 heads, six visual cross-attention layers. The fused comparison jointly changes fusion and compression; further query-count/no-compression ablations are follow-ups.

## Optimization and selection

Each model receives 20,000 optimizer updates, microbatch four, gradient accumulation two, effective batch eight, 160,000 sampled anchors. AdamW learning rate 3e-4, betas (0.9,0.95), weight decay 1e-8, warmup 100, cosine floor 1e-5, gradient clipping 1.0, bf16. No state or condition dropout. Add Gaussian noise to the sixteen-action target at a random diffusion timestep and minimize noise-prediction MSE. Preserve initialization and RNG across conditioning variants.

Every 250 updates, run DDIM20 on sixteen deterministic evenly spaced anchors from each of the ten validation trajectories (160 total), same sample order and seed 42. Select best checkpoint by normalized action MSE restricted to valid future commands and right-arm dimensions [6,7,8,9,10,11,13]. Report full 14-D padded action MSE separately. All losses remain full 14-D denoising loss. Do not inspect test action predictions during selection. Retain weights at update 10000, validation-selected best, and final 20000.

Two GPU queue workers start direct on GPU0 and fused on GPU1. After their respective primary runs complete, GPU0 trains state_only and GPU1 current_only. This avoids contention during the main comparison. Status/queue logs remain local, W&B offline. Match the first sixteen production microbatches across all variants and record sample, input, target, and common-parameter hashes during preflight.

## Storage and resumability

Project storage is constrained. Preserve best/final/10000-update weights in the project. Every 1000 updates, save complete model/optimizer/scheduler/RNG/cursor state to the run-specific `/tmp/omega_rotate_qrcode_training_states/<study>/<variant>/latest`; expose a symlink `latest_training_state` from each run. Write a staging state and completion marker before replacing the previous state. Retain only the latest complete state. Resume by setting trainer.resume_checkpoint to that directory. Best weights do not have their own optimizer snapshot. The temporary filesystem can be cleared on reboot; copy a needed resume state elsewhere before such maintenance. Existing study artifacts are untouched.

## Geometry registration and evaluation

Before launch, verify all 100 source identities with initial simulator RGB, state, and model-ID checks. Record initial object XYZ, quaternion, orientation offset from the task's nominal initial orientation, initial error to the success orientation, sign model, and train/validation/test membership. Report pose ranges, orientation bins (0–15°, 15–30°, >=30°) and model coverage by split. These are initial-state labels only; do not assign regenerated later-frame simulator poses to original demonstration images.

Prelaunch audit amendment: source seed9 has RGB MAE 6.027 despite identical regenerated geometry/state/model in a fresh environment. Inspection shows a blue-to-brown photometric tint change between recorded and current rendered images. Report RGB errors and flags above6; use grayscale MAE <=6 for appearance identity, alongside exact state/model checks and registered pose verification. Preserve the seed9 comparison image and fresh-instance diagnostic. This is a renderer distribution difference and must be disclosed in downstream evaluation. It is not evidence of policy error and no policy outcomes informed the amendment.

Register 50 fresh stable right-arm scenes by increasing simulator seeds starting at 100000, excluding source seeds. Reject only unstable setup or left-arm scenes, log all rejections, and never filter using expert/policy success. This is an explicitly right-arm-conditional fresh-scene diagnostic, not the full official benchmark. Freeze scenes before policy training. The paired suite includes ten uniformly selected train trajectories, all ten validation trajectories, all ten test trajectories, and all fifty fresh scenes. Group success by initial orientation, position and model.

After training, use best checkpoints for primary closed-loop comparison. Predict sixteen actions with DDIM20/seed4000. Execute eight, then replan; update eight-observation camera histories on every control observation. Evaluate execution horizon four separately on the exact same scenes and weights. Use the unmodified task success criterion and official step limit. The success implementation checks quaternion components, low object height, and both open grippers, without an explicit target-XY check. Record target-XY error separately.

Record head videos, state/action traces, object position/quaternion, sign orientation error in degrees (quaternion geodesic), contact, lift, placement and release. Secondary phases use two-finger contact, lift >=4cm, orientation error <15° while raised, low-height placement within15°, and both open grippers. These diagnostics do not replace official success or prove stable grasp. Verify scene pose/model/state identity against registration and fully decode all videos. Replay recorded expert commands for controller controls.

Offline evaluation uses equal-trajectory anchor sampling on all train/validation/test episodes, paired DDIM seeds, active-arm joint RMSE in radians, active-gripper accuracy, valid-future MSE and first/executed-action errors. Preserve final and update10000 checkpoints for budget comparisons. Test set is distinct from validation selection; fresh scenes additionally test unseen initial configurations.

Ordinary visual-encoder comparisons, systematic out-of-distribution camera/pose interventions, more fusion queries, backbone tuning, and multiple training seeds are later experiments. The present four variants test conditioning design, dependence on visual observations, and temporal history; they alone do not establish geometry-specific superiority over ordinary visual representations.

## Launch and evaluation tools

Prepare with `tools/launch_rotate_qrcode_streams.py`. Run `tools/register_rotate_qrcode_geometry.py --mapping-only` using the ML environment, then the same script without the flag using the simulator environment and matching renderer/CUDA/Warp runtime. Run `tools/preflight_omega_camera_streams.py` for all four configs. Launch with `tools/launch_rotate_qrcode_streams.py --launch-manifest <manifest.json>`. Inspect `tools/omega_camera_stream_status.py --manifest <manifest.json>`.

After completion, `tools/prepare_rotate_qrcode_evaluation.py --training-manifest <manifest.json>` prepares checkpoint hashes and paired evaluation descriptors. `tools/evaluate_omega_camera_stream_actions.py` and `tools/rollout_omega_camera_streams.py` support this task and both execution horizons. Training launches now; post-training simulator evaluation is a separate stage.

Run `tools/run_omega_stream_evaluation.py --manifest <evaluation-manifest.json> --sequential-per-gpu` for the prepared four-variant evaluation. Each GPU evaluates its assigned variants in sequence, avoiding websocket-port collisions. This runs horizons8/4, recorded expert/controller controls for direct, offline best/final/update10000 evaluation, and full video decoding.
