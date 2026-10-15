# GAAS Omega experiments and current progress

Observed **2026-10-13 18:36:37 SGT / 10:36:37 UTC**. This record covers the GAAS H200
experiments; the original workstation's October 12 results remain a separate
dated section in [the experiment ledger](geometric_action_experiments.md).
Machine-readable evidence: [summary](experiment_records/gaas_20261013_183637/summary.json)
and [artifact index](experiment_records/gaas_20261013_183637/artifact_index.json).
The 10K-checkpoint rollout results are complete; training observations are
provisional and are not a synchronized optimizer-state snapshot. The earlier
[12:43 SGT snapshot](experiment_records/gaas_20261013/README.md) is retained.
Model weights, datasets, videos and action traces
remain at the local paths recorded in the manifests.

## Training A and C

| Experiment | PBS job / state | Completed updates | Sampled observations | Best active-arm validation MSE | Best update |
| --- | --- | ---: | ---: | ---: | ---: |
| A: single-call joint24, direct registers | 84822.gaas / running | 19,380 / 20,000 | 155,040 / 160,000 | 0.0318774 | 18,750 |
| C: A plus current-patch highway | 84945.gaas / running | 17,840 / 20,000 | 142,720 / 160,000 | 0.0305380 | 17,500 |

Both use all 24 images in one frozen Omega512 call: eight chronological
observations from head/left/right cameras, time-major input order, independent
batch elements. A supplies 408 camera/register tokens to the action decoder.
C preserves that path and adds the latest three-camera patches: 3,108 tokens
at a 28x37 patch grid per image, explicit XY coordinates, shared camera/current
time embeddings, a separate spatial type embedding, and six zero-initialized
gated attention residuals after decoder blocks 4/8/12/16/20/24. Spatial KV is
computed once per diffusion sample and reused. Learned time embeddings remain
enabled for both policies; there is no temporal representation ablation here.

Matched settings: seed 42 and fresh initialization, train-only normalization,
the same 80/10/10 episode split of 100 right-arm demonstrations / 8,341 frames,
microbatch 4 with accumulation 2 (effective batch 8), 20,000 updates / 160,000
sampled observations, LR 3e-4, AdamW betas 0.9/0.95, epsilon 1e-8, optimizer
weight decay 1e-8, clip 1, warmup 100 and cosine floor 1e-5. Action horizon is
16; robot state/action dimension is 14. Validation uses the same 160 anchors,
batch 4, DDIM20, seed 42, active right-arm indices plus right gripper, and valid
future commands only, every 250 updates. Full state is saved every 1,000
updates, with policy milestones at 10,000 and best/final policy weights.

At the latest shared validation budget, **17,750 updates / 142,000 observations**,
A's active-arm MSE is **0.0329414** and C's is **0.0313200**, approximately 4.9%
lower for C. C is lower at each of the five most recent shared checkpoints,
16,750 through 17,750, but earlier checkpoints fluctuate. Neither the
unmatched best values in the table nor this small
single-seed difference establish a reliable architecture advantage or task
success. Full curves are retained as [A CSV](experiment_records/gaas_20261013_183637/training/A/validation_history.csv)
and [C CSV](experiment_records/gaas_20261013_183637/training/C/validation_history.csv).

A has 331,455,502 trainable parameters; C has 358,505,492, including 27,049,990
added highway parameters. Their common initial-weight hash, initial training
sample order, image/action hashes and normalization statistics match exactly.
C's GPU preflight verified nonvacuous zero-gate baseline equivalence, finite
loss/actions, gates opening and nonzero spatial/position/KV gradients, with no
frozen Omega gradients. Its four disposable updates were discarded before fresh
production training. Both CPU and real-data GPU preflights passed. Targeted CPU
checks passed in groups of 24 for training/highway and 13 for rollout contracts;
the earlier original-workstation suite is recorded separately.

Source snapshots: A `59930247f7ae82a61dfbf4b9581ab7a86399617f`, C
`a83d68f47ad4e2c39868b8d8b6f3c545ebda2994`. Running jobs use immutable source
snapshots; later repository commits do not alter their implementation.
Protocols: [single-call baseline](omega_joint24_gaas_protocol_20261012.md),
[highway C](omega_spatial_highway_gaas_protocol_20261012.md),
[corrected highway proposal](highway_experiment_proposal.md).

## Batch and efficiency measurements

Both training jobs use H200 GPUs with about 140 GiB device memory. Peak PyTorch
training allocations are **16.57 GiB for A** and **16.98 GiB for C**; CUDA
allocator reservations and scheduler process-memory peaks may be larger.
PBS utilization telemetry is retained in each evidence directory's
`pbs_status.json`; its sampled utilization fluctuates and is not a controlled
throughput measurement.

A's disposable batch benchmark tested larger physical/effective batches:

| Microbatch x accumulation | Effective batch | Seconds/update | Observations/sec | Peak allocation, GiB |
| --- | ---: | ---: | ---: | ---: |
| 4 x 2 | 8 | 4.627 | 1.729 | 16.57 |
| 8 x 1 | 8 | 4.576 | 1.748 | 22.64 |
| 16 x 1 | 16 | 9.153 | 1.748 | 37.24 |

Larger batches fit for A, but offered only about 1.1% higher throughput, below
the preregistered >5% selection threshold. Production kept effective batch 8
and the original 20K update/event schedule. C uses the same fixed batch;
larger C batches have not been benchmarked. Increasing accumulation alone does
not increase the number of examples processed in one GPU pass.

C's single disposable paired DDIM20 inference probe measured 0.722 s with the
register path and 0.737 s with its highway active. These single timings are
diagnostic, not a stable performance benchmark. Peak memory and update timing
records are in the retained [A batch benchmark](experiment_records/gaas_20261013/training/A/batch_tuning.json)
and [C GPU preflight](experiment_records/gaas_20261013/training/C/preflight.json).

## Matched 10,000-step rollout evaluation

The exact 10,000-step milestone checkpoints (80,000 observations each) have
completed evaluation while training continues. A's evaluation is **85451.gaas**
and C's is **85452.gaas**, on separate GPUs in `hpc-gaas-g04`. Both jobs finished
with PBS exit status 0; worker status is completed and all videos passed full
decoding. These outcomes describe the 10K policies, not later training weights.
The [registered evaluation manifest](experiment_records/gaas_20261013_183637/evaluation/manifest.json)
contains checkpoint/config/normalization/reference hashes and full paths.

Both policies use the same scenes selected before observing outcomes:

| Split | Dataset episodes | Simulator seeds |
| --- | --- | --- |
| Test: 3 requested | 2, 4, 24 | 91, 142, 62 |
| Train: 5 requested | 0, 11, 22, 34, 44 | 131, 181, 9, 256, 240 |

All eight initial scenes passed real-GPU reset verification for each policy:
robot state and object position error were zero, maximum orientation error was
about 2.4e-6 degrees, and maximum three-camera RGB/grayscale MAE was
1.2704/1.1839 on the 0–255 scale. Object model and right-arm identity matched.
These checks verify the selected initial scenes on the copied GAAS simulator;
they do not make all subsequent dynamics identical to the source workstation.

Rollouts use native 640x480 cameras with preserved field of view, stride-one
eight-image histories captured every control observation and initially
left-padded, literal `rotate_qrcode`, bf16 policy inference with float32 Omega
weights, DDIM20, execution horizon 8, and diffusion seed 4000 + chunk index.
Each episode resets history and diffusion sequence. Videos and action/geometry
traces are recorded; all sixteen videos have now passed full decoding.
The rollout source snapshot is `cf32b3592a24da834ca35e3bb623a050e218e18d`.

**Completed official outcomes:**

| Policy | Test successes | Train successes | Total completed | Grasp-contact flag | Lift >=4 cm flag |
| --- | ---: | ---: | ---: | ---: | ---: |
| A, 10K milestone | 0 / 3 | 0 / 5 | 8 / 8 | 1 / 8 | 2 / 8 |
| C, 10K milestone | 0 / 3 | 0 / 5 | 8 / 8 | 5 / 8 | 2 / 8 |

All episodes reached the 400-control-step limit. The secondary failure-phase
labels are:

| Split | Episode / seed | A failure phase | C failure phase |
| --- | --- | --- | --- |
| Test | 2 / 91 | grasp | grasp |
| Test | 4 / 142 | grasp | lift |
| Test | 24 / 62 | grasp | rotation |
| Train | 0 / 131 | grasp | lift |
| Train | 11 / 181 | grasp | lift |
| Train | 22 / 9 | lift | grasp |
| Train | 34 / 256 | grasp | lift |
| Train | 44 / 240 | grasp | grasp |

Contact and lift flags are supporting diagnostics, not proof of a stable grasp
or task completion. An object can rise after a collision without passing the
grasp check. C's more frequent contact flags do not establish a reliable
architecture advantage in this eight-scene, single-seed evaluation.
See [A complete results](experiment_records/gaas_20261013_183637/evaluation/A/rollout_results.json),
[C complete results](experiment_records/gaas_20261013_183637/evaluation/C/rollout_results.json),
and the [per-episode CSV](experiment_records/gaas_20261013_183637/evaluation/episode_outcomes.csv).
The machine-readable summary retains the historical field name
`partial_outcomes`; its counts are final for this completed evaluation.

Rollout protocol: [matched 10K train/test evaluation](omega_paired_10k_gaas_rollouts_20261013.md).
Large rollout outputs live under
`/projects_vol/gp_chuanxia.zheng/n2506021c/PVG-starVLA/evaluations/rotate_qrcode_AC_10000_20261013/`.
C's checkpoint and full-state output is also on shared project storage. A's
latest complete optimizer state remains node-local until its runner copies it
to durable output on ordinary/catchable exit; its milestone/best weights are
already durable. Complete marker provenance is recorded where locally visible.

## Repeated-motion and visual-conditioning investigation

Inspection of videos suggested similar motions despite different object scenes.
The [trace analysis](experiment_records/gaas_20261013_183637/diagnostics/visual_sensitivity_trace_analysis.json)
uses all eight completed scenes per policy. It records each trace's path and
SHA256 and compares the six active right-arm joints, excluding grippers and the
idle arm. Each value below is the mean pairwise joint RMSE over 28 scene pairs:

| Control steps, zero-based | A, radians | C, radians |
| --- | ---: | ---: |
| First command: 0 | 0.033313 | 0.025540 |
| First chunk: 0-7 | 0.045776 | 0.039411 |
| Entire rollout: 0-399 | 0.415069 | 0.110298 |
| Last 80 commands: 320-399 | 0.478767 | 0.107942 |

Initial command states, task, diffusion seed and history padding are identical
across scenes; the first chunk varies with the images. This is evidence
consistent with visual sensitivity, not a proof of correct visual grounding.
Repeated identical-input/noise-control inference has not been run. Later
cross-scene comparisons also differ in robot state and interaction history;
C's smaller trajectory differences do not establish less reliance on vision.
The [contact sheet](experiment_records/gaas_20261013_183637/diagnostics/test_rollout_contact_sheet.jpg)
shows video frames 0/80/160/320 for three test scenes per policy. Video-frame
indices are not control-step indices, and these are early excerpts, not final
rollout poses.

Potential causes remain hypotheses: command-state shortcuts with zero state
dropout, distribution shift after a missed grasp, incorrect mapping from frozen
visual features to robot motion, and executing eight commands before replanning.
RoboTwin state contains joint drive targets rather than measured physical qpos.
The input path inspected updates history at every control step, resets per
episode and re-encodes images for each predicted chunk; no permanently cached
image bug was found in that inspection.

C's six gates at the **17,840-step training snapshot** are approximately
[-0.001076, 0.000110, 0.000180, -0.000598, -0.000268, -0.002634]. This is not a
measurement of the evaluated 10K gates. Nonzero gates show an available spatial
route but gate magnitude alone cannot establish its actual residual contribution
or useful object-dependent control. No image intervention, highway-disabled
ablation, execution-horizon comparison or new state-dropout training has been
launched in this update.

## Next work

1. Hold robot state and diffusion noise fixed while changing object position in
   matched images. Measure whether predicted gripper motion responds correctly
   to object displacement, not just whether action vectors change. Repeat
   identical inputs to establish inference reproducibility.
2. Measure C with its highway disabled on matched expert observations; compare
   action error and spatial-branch residual magnitudes. On a small matched
   scene set, compare execution horizons 1 and 8 with a specified noise pairing.
3. Finish both 20K training runs, confirm their complete full-state markers, and
   compare validation at matched observation counts. Decide final/best-checkpoint
   rollouts from the registered protocol, preserving the same scene/input/noise
   contract. No additional suite has been launched in this documentation update.
4. Use those diagnostics to choose state-dropout training, explicit spatial
   supervision or recovery-data collection. Consider the registered
   register-capacity control Creg before attributing an advantage to spatial
   information. Temporal B0/Bfixed and larger-history D/E remain unlaunched.
   Refresh the original machine's three-call results separately when available.

Capture a future dated observation without overwriting this one:

```bash
python tools/snapshot_omega_gaas_experiments.py \
  --out docs/experiment_records/gaas_YYYYMMDD_HHMMSS
```

The artifact index hashes copied source bytes and lists original paths. Exported
validation CSVs and the summary are derived from the captured metrics/results.

## A final 20K conditioning evaluation, submitted 2026-10-13 20:52 SGT

A finished exactly 20,000 updates / 160,000 sampled observations. Its final
weights are now registered for a four-condition inference intervention:
full inputs, vision without explicit state, explicit state without vision,
and task only. All conditions retain the task token and share weights,
normalization, eight registered scenes and diffusion seeds. This is 32
rollouts: three test and five train per condition. PBS jobs 85749–85752.gaas
were running at first check; all four CPU masking preflights passed.
GPU/scene checks and rollout results remain pending in this submission record.
**No C conditioning evaluation was queued.**

See [A 20K conditioning protocol and job registration](omega_A20k_conditioning_gaas_protocol_20261013.md).
Inference masking tests dependence on the available routes and introduces
inputs unseen during training; it cannot alone prove harmful overreliance.
The previous dated 10K evidence and training snapshot above remain unchanged.

## A 20K diagnostics update, 2026-10-14

All four conditioning workers 85749–85752.gaas finished with PBS exit 0,
eight rollouts each and full video validation. All conditions scored 0/3 test
and 0/5 train. Full/vision-only/state-only/task-only recorded grasp contact
in 1/4/0/4 of eight scenes and lift >=4 cm in 2/3/0/0. These secondary flags
can result from collisions and are not stable-grasp or task success evidence.
Mean first-chunk active-joint RMSE versus full was 0.886162 rad without explicit
state, 0.196809 rad without vision, and 0.573187 rad with task only. This
supports stronger explicit-state dependence and a nonzero visual effect,
without proving harmful overreliance under inference masking.

A controlled object-position diagnostic was submitted as **86207.gaas** at
12:25 SGT. It uses the same final A weights, eight registered scenes, fixed
state/task/noise, 40 rerendered scenes, 360 action predictions and 72 repeat
checks, with successful expert-pair response summaries and a state-only
negative control. It was queued at first check; results are pending.
**No C evaluation or new training run was queued.**
See [controlled object-position protocol](omega_A20k_object_position_gaas_protocol_20261014.md)
and its archived prior four-condition comparison for complete evidence.

## Fresh highway C state-dropout training, 2026-10-14 18:25 SGT

At the user's request, submitted **86363.gaas** for a fresh seed-42 highway C
training run with 20% complete explicit-state-token dropout, 10,000 updates
and 80,000 sampled observations. The existing highway, frozen Omega, dataset,
split, normalization, effective batch8, optimizer and validation settings are
preserved. The original 20K cosine LR horizon is retained while optimization
stops at 10K, matching the LR trajectory of C's original 10K milestone.
Validation/deployment uses full state. No trained checkpoint is resumed.

The job was running at first check, with five CPU tests and all dataset checksum
verification passed; real-batch GPU preflight was in progress. Production
starts only after initialization/data/statistics, mask semantics and highway
gradient checks pass. No new evaluation was submitted with this training run.
See [registered protocol/config/job](omega_highway_state_dropout20_10k_gaas_protocol_20261014.md).

The C dropout worker's GPU preflight subsequently passed on H200: p=.2 route
semantics, original initialization/data/normalization parity, zero-gate
baseline equivalence, frozen Omega and useful spatial gradients were verified.
Fresh production training has started. The archived submission record includes
the real-batch preflight and actual production configuration.

## C state-dropout training completed; matched evaluation, 2026-10-15

Training 86363.gaas completed successfully at 08:47 SGT: 10,000 updates,
80,000 observations, final validation active-action MSE 0.067631 and best
0.055829 at 8,250 (original C at 10K: 0.049457). Checkpoints and full latest
training state are saved. Validation fit worsened; rollout and grounding
behavior still require direct evaluation.

At the user's request, submitted **86685.gaas** at 11:54 SGT for the same
three test and five train scenes used for A/C at 10K. Use the dropout model's
10K milestone, full state and vision, DDIM20, identical seeds, native cameras,
history8 and execute8. First scheduler check: running on hpc-gaas-g10.
Scene/GPU preflights and eight video/action-trace outcomes remain pending.
See [matched evaluation protocol and receipt](omega_C_dropout20_10k_gaas_rollouts_20261015.md).
