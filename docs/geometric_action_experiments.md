# Geometric action experiments

## Current GAAS progress — 2026-10-13

The [GAAS experiment record](omega_gaas_experiments.md) documents the single-call
baseline A, implemented current-patch highway C, batch benchmark, passed CPU/GPU
preflights, and their completed paired 10K milestone rollouts. Latest evidence
observed at **2026-10-13 18:36:37 SGT** is committed under
[gaas_20261013_183637](experiment_records/gaas_20261013_183637/README.md).
The earlier [12:43 SGT partial snapshot](experiment_records/gaas_20261013/README.md)
remains available.

| GAAS experiment | PBS training job | Progress / 20K updates | Best active-arm validation MSE |
| --- | --- | ---: | ---: |
| A: joint24 single-call direct registers | 84822.gaas, running | 19,380 | 0.0318774 at 18.75K |
| C: A plus gated current-patch highway | 84945.gaas, running | 17,840 | 0.0305380 at 17.5K |

Both retain effective batch 8, LR 3e-4 and a 160K observation budget with matched
data, initialization and schedules. At a matched 17.75K updates, validation MSE
is 0.0329414 for A and 0.0313200 for C, about 4.9% lower for C. Training remains
in progress; offline MSE is not task success.

Their 10K-checkpoint evaluation jobs 85451/85452.gaas both completed with exit
status 0 and all videos decoded. Both policies scored **0/3 test and 0/5 train**
on the eight preregistered, verified scenes. A/C grasp-contact flags were 1/8
and 5/8; lift >=4 cm flags were 2/8 for both. These secondary flags do not prove
stable grasps or successful completion. Later training checkpoints have not
been evaluated.

Full eight-scene trace comparisons show different initial predictions despite
matched initial state/noise, consistent with visual sensitivity. C's later
trajectories are more similar across scenes, but that does not measure useful
visual grounding or establish a state shortcut. The GAAS record documents
the measurements, unresolved hypotheses and proposed image/geometry,
highway-disabled and execution-horizon diagnostics. No new diagnostic suite
or training variant was launched in this documentation update.

## Original workstation snapshot — 2026-10-12

This ledger records the retained evidence in `/export/home/haoliang/projects/PVG-starVLA-snapshot` as of **2026-10-12**. The machine-readable [inventory](experiment_records/20261012/inventory.json) lists every retained training directory and result family; the [training table](experiment_records/20261012/training_runs.csv) lists individual runs. The [artifact index](experiment_records/20261012/artifact_index.json) records original local paths, file sizes, and hashes for copied evidence. Registered original source archives are preserved by content hash under `original_sources/`, with original path mappings in the inventory. Large weights, videos, meshes, datasets, optimizer states, and per-step traces remain in their original local directories.

Coverage includes 28 training directories, 23 artifact families, and the separate RoboTwin simulator result files. Smoke tests, failed/abandoned preparations, retries, and unlaunched configurations remain visible. This is an inventory of surviving records, not a claim to recover deleted experiments or unrelated users' jobs. Historical manifests and copied reports retain their original timestamps and may contain stale PIDs/statuses; the inventory reconciles current trainer evidence where available. An old final checkpoint does not prove that a later continuation reached its requested budget.

## Original-workstation camera-stream study: rotate_qrcode

Study: `results/omega_camera_stream_training/rotate_qrcode_20261010_135404`. See the [protocol](omega_rotate_qrcode_protocol_20261010.md), [registered manifest](experiment_records/20261012/evidence/omega_camera_stream_training/rotate_qrcode_20261010_135404/manifest.json), and inventory for the exact observation time and update counts.

| Variant | Budget | State at 2026-10-12 04:46 UTC | Best validation MSE | Best update |
|---|---:|---|---:|---:|
| Direct 408 tokens | 20,000 updates | Running, 18,890 updates | 0.0240906 | 18,250 |
| Fused 32 tokens | 20,000 updates | Running, 19,800 updates | 0.0497199 | 18,000 |
| State-only | 20,000 updates planned | **Cancelled before starting by user** | — | — |
| Current-image-only | 20,000 updates planned | **Cancelled before starting by user** | — | — |

All four configurations were prepared and passed their input/gradient preflights. Only the two primary policies were trained. On October 12, both queue workers were terminated individually and both controls were marked cancelled; the primary processes were preserved. Their trainers run independently now. They do not have an active queue worker to update the historical manifest when they finish, so consult `stream_training_status.json` and `tools/omega_camera_stream_status.py`.

Both policies use three separate frozen Omega512 calls, each receiving an eight-frame history from one physical camera. The action policy retains one camera token and sixteen scene/register tokens per image: `3 × 8 × 17 = 408`. Direct supplies these tokens to the action decoder; fused uses two 512-wide transformer layers and 32 learned query outputs. Learned camera/time/type embeddings are enabled; temporal RoPE is disabled. Both use robot state, a task token, a 24-layer 1024-wide action DiT, 16 future absolute commands, batch four with accumulation two, and matched common initialization/data order.

The dataset has 100 right-arm demonstrations and 8,341 observations, split 80/10/10 by episode with seed 42. Statistics use training episodes only. Validation uses 160 fixed anchors from ten validation trajectories, DDIM20, normalized right-arm joints plus right gripper, and valid future commands only. The ten test trajectories have not been used for checkpoint selection. **No rotate_qrcode camera-stream offline test evaluation or simulator rollout has been completed.** The registered future suite contains matched train/validation/test scenes, fifty fresh stable right-arm scenes, execution horizons eight/four, orientation and grasp/lift/place/release diagnostics. This suite is prepared, not an achieved benchmark result.

The original run began October 10 at 14:18:52 UTC. Both trainers and queue workers disappeared around October 11 at 08:56 UTC without a traceback; the cause is unknown. Both complete 10,000-update checkpoints survived, including optimizer/scheduler/RNG/data cursor. Training resumed October 11 at 10:13 UTC, with state restoration logged at 10:14:33. Fused repeated approximately 240 unsaved updates. Original configs/logs and a [recovery record](experiment_records/20261012/evidence/omega_camera_stream_training/rotate_qrcode_20261010_135404/recovery_20261011_101334.json) are retained.

At the 03:59 UTC October 12 check, elapsed wall time was about 37h40m, including approximately 36h23m active runtime per model and a 1h17m interruption. The policies run concurrently. Runtime includes repeated online frozen-backbone inference, diffusion validation every 250 updates, and checkpoint writes. Both GPUs are shared with other jobs. A frozen-token cache has been proposed but **has not been implemented or used in these runs**; 6,674 unique training anchors would require roughly 10.4 GiB for BF16 raw `408 × 2048` token arrays.

## Camera-stream policy training: adjust_bottle

Study: `results/omega_camera_stream_training/adjust_bottle_20261009_112722`. Both variants completed 10,000 updates / 80,000 sampled anchors, using 45 training and five validation episodes. Direct took approximately 17h57m and fused 17h46m. These validation metrics include the full action vector and padding, unlike the newer rotate_qrcode active-arm metric; compare variants within a study rather than absolute scores across tasks.

| Variant | Best training-time validation MSE | Best update | Best checkpoint offline train MSE | Offline validation MSE |
|---|---:|---:|---:|---:|
| Direct | 0.01650 | 9,750 | 0.01381 | 0.02389 |
| Fused | 0.03297 | 9,000 | 0.02224 | 0.03651 |

Offline evaluation covers all 45 training trajectories at eight anchors each and all five validation trajectories at sixteen anchors each. Direct had lower valid-future error on 41/45 train and 4/5 validation trajectories. The validation episodes were used for model selection and are not an independent test set.

| Best checkpoint | Five selected training scenes | Five validation scenes | Five fresh scenes |
|---|---:|---:|---:|
| Direct | 4/5 | 2/5 | 2/5 |
| Fused | 3/5 | 2/5 | 2/5 |

Raw recorded-expert replay and normalization-roundtrip expert replay each passed 10/10 matched train/validation scenes. All fifty demonstrations used the right arm; constant left-arm normalization restricts left-arm commands. Both fresh-seed policies scored 0/2 on left-arm scenes and 2/3 on right-arm scenes. This small evaluation supports learning with camera/register tokens, but does not establish a geometric-feature advantage over state or ordinary RGB baselines.

Full numeric evidence: [evaluation summary](experiment_records/20261012/evidence/omega_camera_stream_training/adjust_bottle_20261009_112722/evaluation_20261010_112911/summary.json). Original videos, traces, galleries, and plots remain in the local evaluation directory. The [evaluation protocol](omega_camera_stream_evaluation_protocol_20261010.md) describes scene verification, action order, normalization, and paired noise.

## Eight-image reconstruction intuition checks

Independent histories were tested on RoboTwin `rotate_qrcode`, `adjust_bottle`, and `stack_blocks_three`, and four chunks of BEHAVIOR `turning_on_radio` episode 197. Each physical-camera stream of eight images enters **one Omega call**. Displayed all-frame clouds superpose the eight outputs of that single call; they are not eight separate model invocations or a motion-compensated static object surface. Independent calls have independent coordinate gauges and scales.

RoboTwin fixed-head relative rotation discrepancies were below approximately 0.52° in the selected chunks. A moving-left-wrist `stack_blocks_three` example, frames 170–177, showed recorded relative rotation of 29.26° versus Omega's 1.29°, with maximum discrepancy 28.08°. These recorded pose labels were not independently replay-verified; visible image motion alone does not validate the 29° interpretation. Depth ground truth was unavailable.

The BEHAVIOR ablation supplies all 24 matched images to one Omega call in chronological timestamp order. Results are mixed: on frames 934–941, the left-wrist maximum rotation discrepancy improved from 11.84° to 6.27°, while the right wrist worsened from 21.17° to 34.80°. Cross-camera discrepancies were also substantial. A common output coordinate system did not guarantee correct camera registration. No action training or success-rate claim follows from these reconstruction checks.

Evidence: [RoboTwin independent streams](experiment_records/20261012/evidence/omega_camera_streams/20261009_chunk8_stride1/manifest.json), [moving-left-wrist case](experiment_records/20261012/evidence/omega_camera_streams/20261009_chunk8_left_motion/manifest.json), [BEHAVIOR independent streams](experiment_records/20261012/evidence/omega_camera_streams/20261009_behavior_radio_chunk8_stride1/manifest.json), and [joint-24 ablation](experiment_records/20261012/evidence/omega_camera_streams/20261009_behavior_radio_joint24_stride1/manifest.json). Per-camera metrics are preserved alongside the manifests.

## Earlier Omega512 action experiments

| Experiment | Training evidence | Execution outcome / limitation |
|---|---|---|
| Fresh dense vs state-query/local, Oct 5 | 10K updates each; best full-action validation MSE 0.02450 / 0.02362 | Both 0/3 fresh scenes and 0/3 verified training scenes; expert replay 3/3 |
| Conditioning interventions and controls, Oct 5–6 | Fresh vision-only and state-only, 10K each; best MSE 0.06282 / 0.02834 | Full, vision-only, state-only each 0/2 matched right-side scenes; seed-43 repeats prepared but unlaunched |
| Single-trajectory overfit, Oct 6 | Episode 0, 82 observations, 10K updates; final trajectory MSE 0.007498, best 0.007122 at 9K | Same scene: execute-eight succeeds in 52 steps; execute-one/two fail, one rollout each; chunk-index noise changes with horizon |
| Eight-trajectory dense fit, Oct 7–8 | 20K updates on 668 observations; full training MSE 0.01930 at 10K → 0.00826 at 20K | Original 24-rollout suites: 1/24 → 0/24; earlier input-size mismatch limits interpretation |
| Corrected-size extra evaluation, Oct 8 | Same 10K/20K checkpoints, 640×480 policy input, three verified scenes, diffusion seed 5000 | Both 0/3; lower training error did not establish successful control |
| Visual-only localization probe, Oct 7–8 | Simulator-synchronized image/geometry labels, trajectory-disjoint split, frozen Omega features | Test mean object error 2.20 cm vs train-mean baseline 7.55 cm; matched-head test-image shuffle 5.92 cm; trajectory-shuffled head 5.74 cm |

The localization probe's matched object-minus-gripper error was 1.84 cm, versus 7.46 cm for the train-mean baseline. This supports usable localization information in the selected features and regenerated scenes, not reliable grasping or a causal explanation of policy failure. Label-collection alignment amendments are retained.

Conditioning interventions found that masking all explicit state worsened sampled expert-action error; masking only query-state offsets improved the small query/local probe. Controlled object translations produced incorrect predicted end-effector responses despite expert success in 5/5 scenes. These diagnostics are not equivalent to training a masked policy and do not isolate a state shortcut as the cause.

RoboTwin observations and converted states contain **joint drive targets**, not measured physical qpos. Converted examples pair observation at time t with command at t+1. All hundred source rotate_qrcode demonstrations use the right arm. Train-only normalization, action order, fixed task identity, and native-versus-training image dimensions were audited and corrected in the tooling.

Evidence: [minimum-case results](experiment_records/20261012/evidence/robotwin_omega512_minimum_case/20261006_075346/completed_results.md), [causal-control manifest](experiment_records/20261012/evidence/robotwin_omega512_causal/20261005/training_controls/manifest.json), [eight-trajectory protocol](experiment_records/20261012/evidence/omega512_protocol/20261007_185108/manifest.json), [completed localization probe](experiment_records/20261012/evidence/omega512_protocol/20261007_185108/visual_probe/probe_results.json). Final-block adaptation configurations were prepared but not launched. GAAS payload/CPU validation was prepared separately; no GAAS GPU training result is claimed.

## Earlier Surflo policies, diagnostics, and reconstruction

| Family | Recorded experiments | Current evidence |
|---|---|---|
| BEHAVIOR radio | Three-view Surflo action policy, continuation, 20% state dropout; additive/state-token/six-part-state variants | Three camera/state variants have 20K terminal checkpoints. Dropout run has a 40K terminal checkpoint. The original radio directory retains final weights but only 39K periodic evidence against a later 60K configuration; continuation completion is unverified. No task-success claim is made from denoising or training-batch MSE. |
| RoboTwin projection comparison, Oct 1–2 | Linear/residual, initial 20K sampled anchors (2,500 updates), abandoned continuation attempts, fresh 10K-update variants, smoke/budget checks | Fresh linear/residual each 0/5 selected fresh scenes. Verified training scenes: linear 1/3, residual 0/3. Linear's success was a push into the accepted pose with open grippers, not a demonstrated grasp/lift. |
| Surflo compressor adaptation, Oct 2–3 | Train compressor alone vs compressor plus state conditioning, 10K each | Best validation MSE 0.02770 / 0.02606; both 0/5 selected fresh scenes. |
| Controller and visual diagnostics | Matched expert normalization replay, held-out tracking, scene swap/blank images, geometry probes and controls | Expert replays succeed; transform errors are tiny. Visual sensitivity is present but does not prove useful target alignment. Unmatched historical scenes are not controller-failure evidence. |
| Surflo reconstruction reproduction/repair | Robot and garden references; plain/guided modes, camera/precision/normal/time/source-geometry controls, extraction diagnostics | Reproduction and failure artifacts retained in `robotwin_surflo_reconstruction`; inspect per-case evidence rather than infer a single global accuracy score. |
| Four-model reconstruction comparison, Oct 5 | Surflo, VGGT, Omega256 Text, Omega512 on robot and BEHAVIOR examples | Same saved RGB inputs; different representations/extraction procedures and arbitrary predicted gauges. No action-prediction conclusion. |
| Registered four-model RoboTwin study, Oct 7 | Three tasks × episodes 7/23 × initial/interaction = 12 cases | All twelve cases recorded. Renderer fixes rerendered cached geometry without changing raw reconstruction; reconstruction failures remain failures. |

For the twelve-case reconstruction study, mean wrist fitted-view RGB MAE was 0.3883 for guided Surflo, 0.0209 for VGGT, 0.0212 for Omega256, and 0.0287 for Omega512; wrist coverage was 60.4%, 99.8%, 99.5%, and 99.8%. Source-colored depth meshes and Surflo's learned shared surface are different representations. These numbers measure fitted-view consistency, not comparable novel-view quality or ground-truth surface accuracy.

The inventory preserves individual configurations, checkpoint paths, smoke-test budgets, retry evidence, original quantitative reports, and transfer/storage-cleanup manifests. See [reconstruction summary](experiment_records/20261012/evidence/robotwin_reconstruction_study/20261007/summary.json), [training-scene results](experiment_records/20261012/evidence/robotwin_training_scene_rollouts/summary.json), and the full inventory for entries not singled out in this narrative.

## Refresh and reproduce

```bash
python tools/omega_camera_stream_status.py \
  --manifest results/omega_camera_stream_training/rotate_qrcode_20261010_135404/manifest.json

python tools/snapshot_machine_experiments.py \
  --output docs/experiment_records/YYYYMMDD \
  --simulator-results /export/home/haoliang/projects/RoboTwin_PVG/eval_result
```

Run the snapshot tool from the StarVLA environment. A snapshot is a dated observation of live files, not a synchronized checkpoint transaction. Use completed checkpoint markers before copying/resuming state. Retained source hashes and original configs describe the actual experiments; subsequent formatting/maintenance commits do not change already-running processes.

Machine setup: H100 NVL GPUs, StarVLA environment at `/export/home/haoliang/miniconda3/envs/starVLA`, simulator environment at `.../envs/RoboTwin`, Omega source at `/export/home/haoliang/projects/vggt-omega-action-pred`, and RoboTwin simulator at `/export/home/haoliang/projects/RoboTwin_PVG`. Rendering/training use isolated NVIDIA libraries matching kernel 595.71.05; simulation uses isolated Warp 1.9.1. The verified Omega512 checkpoint SHA256 is `c02da418b18bb01d0392598d3f6147366bcde1bb70fd08a5e3bf7925b0667934`. Permanent best/final/10K camera-stream weights are in project `results/Checkpoints`; latest full rotate_qrcode resume states are in `/tmp/omega_rotate_qrcode_training_states/20261010_135404/`.
