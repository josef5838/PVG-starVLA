# Omega512 state-conditioning investigation

Updated 2026-10-05T12:14:30.702022+00:00. Measurements below are completed diagnostics. Training statuses are a timestamped observation; the manifest is the live queue record.

## What was implemented

Explicit conditioning switches (`state_conditioning=full/none/decoder_only/readout_only`, `visual_conditioning=true/false`) mask entire projected state tokens/query offsets or the entire visual memory. They preserve model dimensions, parameter names, initialization order, and checkpoint compatibility. The default full model behavior is preserved. State-only masks query residuals as well, avoiding indirect state leakage through visual memory. An optional local diffusion generator makes rollout noise deterministic by chunk index without changing default sampling.

## Completed intervention results

Three held-out episodes (18, 21, 27), two frames each, and three fixed diffusion seeds were tested per checkpoint. The values are averages of chunk-level right-joint RMSE in radians. There are three independent episodes, not eighteen independent trajectories.

| Input intervention | Dense | Query/local |
|---|---:|---:|
| Original | 0.309 | 0.318 |
| All explicit state masked | 0.509 | 0.589 |
| Visual memory masked | 0.359 | 0.302 |
| Query state masked; decoder state retained | — | 0.265 |

![Conditioning intervention measurements](conditioning_interventions.png)

Removing all explicit state worsens held-out expert-action predictions for both checkpoints. This establishes functional dependence on state, but does not establish that including state during training caused failure. Masking only query-state offsets improves query/local mean joint error across all three tested diffusion seeds. That result identifies a candidate harmful route, with a small sample and a masking combination the original training did not explicitly use. Visual-memory masking is also a stress test outside the normal full-conditioning training distribution.

The simulator generated five physically consistent scenes: baseline, object X -4 cm, X +4 cm, Y +3 cm, and world-Z rotation +0.25 rad. All three cameras are rerendered; object identity, instruction and commanded initial robot state are fixed. The simulator expert succeeds in **5/5** scenes. Against the expert first-chunk end-effector response to the three translations, dense predictions oppose that response in **9/9** comparisons and query/local in **5/9**, across three diffusion seeds. This is stronger evidence of incorrect visual targeting than a black-image test, but is limited to one object/initial robot pose.

![Rendered object interventions, ordered base/X-/X+/Y+/yaw+](controlled_scenes/target_interventions.png)

Static FK compares sixteen-command predicted chunks with simulator-generated expert chunks. The expert reference uses the current simulator's recording cadence; it is not asserted to reproduce the historical training cadence. FK evaluates commanded geometry, not tracking dynamics, grasp success, or end-effector contact. These diagnostics do not establish seen-scene rollout success.

## Correction to the state description and data coverage

RoboTwin `get_left_arm_jointState/get_right_arm_jointState` read joint **drive targets**, whereas the `*_real_jointState` functions read physical joint positions. The observation vector and converted demonstration states use the drive-target path. The converter pairs state at t with command at t+1. Calling these measured physical joint states was inaccurate. Command history can be predictive of the next command even without successful contact.

Both the converted data audit and original source `scene_info.json` contain **100 right-arm episodes, zero left-arm episodes**. This is a limitation of the source pool, rather than an identified conversion drop of left-arm episodes. The causal controls preserve this pool; evaluation uses known right-side seeds 200000/300000. Full-task training will require new left-arm demonstrations and regenerated normalization statistics.

## Fresh matched training controls

Dense architecture is held fixed. The existing seed-42 dense training is the full-conditioning reference. Two fresh controls train without explicit state or without visual information. Neither starts from an old action checkpoint. Three seed-43 configs reproduce full, vision-only, and state-only conditioning for an independent training repetition.

All controls use the original 90/10 episode split (validation seed 42), cameras/native preprocessing, task instruction, optimizer, batch/accumulation, and frozen Omega512 reference. Each uses **10000 updates / 80000 samples**, with **5000/10000 resumable saves**. The masks consume the same training-dropout RNG draws; they do not change module initialization. Vision-only still sees robot pose in RGB.

| Run | Status | GPU | PID | Logged step |
|---|---|---:|---:|---:|
| vision_only_seed42 | training | 0 | 324643 | 567 |
| state_only_seed42 | training | 1 | 324649 | 575 |
| full_seed43 | prepared | None | None | None |
| vision_only_seed43 | prepared | None | None | None |
| state_only_seed43 | prepared | None | None | None |

The local queue automatically evaluates each selected finished checkpoint on the same two right-side scenes, saves videos, and reevaluates the original dense reference with the same diffusion seed 4000 + chunk index. Original full/query evaluation was 0/3 and used an uncontrolled sampling seed, so it is not substituted for this matched evaluation. Inference uses 20 DDIM steps and executes eight actions before replanning. Two right-side scenes are an initial smoke test, not a comprehensive success-rate estimate.

Five new runs would require about 100 GiB with the current complete checkpoint policy. Home had about 44 GiB free, so the first two controls are selected for durable home storage; seed-43 configs are prepared pending a storage choice. The queue reserves checkpoint space and will not silently fill the disk. No GAAS job was submitted.

GPU preflights for both selected conditions pass: correct native images/task, two real gradient updates, expected zero gradients on masked routes, finite DDIM predictions, frozen Omega gradients, and strict policy reload. **27 focused tests passed in separate processes**, including masking, fixed-noise behavior, client ordering/chunk seeds, configuration compatibility, and resume equivalence. Combining those suites in one process exposed their incompatible Accelerate singleton initialization; isolated suite runs passed.

## How to interpret the pending comparison

Evidence for harmful explicit-state reliance would be improved visual response **and** improved matched rollout performance after retraining the vision-only model, repeated across training seeds. A low-error state-only control would quantify how much the expert data can be predicted through the shortcut. All-state masking at inference worsening error alone does not establish overreliance. If vision-only training also fails, prioritize visual target supervision, adapter/expert replay validation, and grasp-specific data rather than treating state removal as a solution.

The candidate remedies remain conditional on this comparison: test stronger state dropout, a vision-only object-goal prediction branch with simulator labels, and recovery demonstrations from displaced objects/imperfect approaches. Those follow-up interventions were not mixed into the causal controls.

## Artifacts and reproduction

- [Dense probe with actions and FK](dense_probe.json), [query/local probe](state_query_local_probe.json), [scene/expert verification](controlled_scenes/scenes.json).
- [Live training manifest](training_controls/manifest.json), [summary snapshot](summary.json), [vision-only GPU preflight](vision_only_preflight.json), [state-only GPU preflight](state_only_preflight.json).
- Diagnostics: `tools/robotwin_omega_intervention_scenes.py` (RoboTwin environment), `tools/omega512_causal_probe.py` (starVLA environment). Run scene capture to a fresh output directory, probe each checkpoint against those scenes, then FK scoring via `--predictions`.
- Refresh this report: `python tools/summarize_omega512_causal.py --directory /export/home/haoliang/projects/PVG-starVLA-snapshot/results/robotwin_omega512_causal/20261005`.
- Queue PID is 324639; logs are in `training_controls/`. `tools/omega512_training_status.py --manifest training_controls/manifest.json` reports cursors and complete periodic states. Extra configs are not automatically submitted to GAAS.
