# GAAS Omega experiments and current progress

Observed **2026-10-13 12:43 SGT / 04:43 UTC**. This record covers the GAAS H200
experiments; the original workstation's October 12 results remain a separate
dated section in [the experiment ledger](geometric_action_experiments.md).
Machine-readable evidence: [summary](experiment_records/gaas_20261013/summary.json)
and [artifact index](experiment_records/gaas_20261013/artifact_index.json).
These are observations of live files, not final results or a synchronized
optimizer-state snapshot. Model weights, datasets, videos and action traces
remain at the local paths recorded in the manifests.

## Training A and C

| Experiment | PBS job / state | Completed updates | Sampled observations | Best active-arm validation MSE | Best update |
| --- | --- | ---: | ---: | ---: | ---: |
| A: single-call joint24, direct registers | 84822.gaas / running | 15,260 / 20,000 | 122,080 / 160,000 | 0.0364182 | 15,000 |
| C: A plus current-patch highway | 84945.gaas / running | 13,740 / 20,000 | 109,920 / 160,000 | 0.0387621 | 13,500 |

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

At the latest shared validation budget, **13,500 updates / 108,000 observations**,
A's active-arm MSE is **0.0416403** and C's is **0.0387621**, approximately 6.9%
lower for C. Nearby checkpoints fluctuate: C was worse at 12,750 and better at
13,000/13,250. Neither the unmatched best values in the table nor this small
single-seed difference establish a reliable architecture advantage or task
success. Full curves are retained as [A CSV](experiment_records/gaas_20261013/training/A/validation_history.csv)
and [C CSV](experiment_records/gaas_20261013/training/C/validation_history.csv).

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

The exact 10,000-step milestone checkpoints (80,000 observations each) are being
evaluated while training continues. A's evaluation is **85451.gaas** and C's is
**85452.gaas**, on separate GPUs in `hpc-gaas-g04`. Both jobs are running.
The [registered evaluation manifest](experiment_records/gaas_20261013/evaluation/manifest.json)
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
traces are recorded; full video decoding occurs after all eight episodes finish.
The rollout source snapshot is `cf32b3592a24da834ca35e3bb623a050e218e18d`.

**Partial outcomes at the observation time:**

| Policy | Completed test outcomes | Completed train episodes | Total completed |
| --- | --- | ---: | ---: |
| A, 10K milestone | 0 successes / 2 completed; episode 24 still pending | 0 / 5 | 2 / 8 |
| C, 10K milestone | 0 successes / 2 completed; episode 24 still pending | 0 / 5 | 2 / 8 |

Episodes 2 and 4 reached the 400-step limit for both policies. A's secondary
diagnostic labels both failures as grasp; C's are grasp and lift respectively.
Contact flags are supporting diagnostics, not proof of a stable grasp. The
remaining test/train episodes are unfinished and must not be counted as failed.
The worker's `completed_episodes` field is only updated after the full rollout
command returns; this snapshot counts the persisted episode results directly.
No completed video-decoding result or completed success-rate comparison is
claimed yet. See [A partial results](experiment_records/gaas_20261013/evaluation/A/rollout_results.json)
and [C partial results](experiment_records/gaas_20261013/evaluation/C/rollout_results.json).

Rollout protocol: [matched 10K train/test evaluation](omega_paired_10k_gaas_rollouts_20261013.md).
Large rollout outputs live under
`/projects_vol/gp_chuanxia.zheng/n2506021c/PVG-starVLA/evaluations/rotate_qrcode_AC_10000_20261013/`.
C's checkpoint and full-state output is also on shared project storage. A's
latest complete optimizer state remains node-local until its runner copies it
to durable output on ordinary/catchable exit; its milestone/best weights are
already durable. Complete marker provenance is recorded where locally visible.

## Next work

1. Finish the registered 3-test/5-train milestone rollouts and decode all videos.
   Report official success and grasp/lift/rotation/placement/release diagnostics
   separately for test and training scenes; inspect traces/videos for failures.
2. Finish both 20K training runs, confirm their complete full-state markers, and
   compare validation at matched observation counts. Decide final/best-checkpoint
   rollouts from the registered protocol, preserving the same scene/input/noise
   contract. No additional suite has been launched in this documentation update.
3. Before attributing an advantage to spatial information, measure trained C
   with its highway disabled and consider the registered register-capacity
   control Creg. Temporal B0/Bfixed and larger-history D/E remain unlaunched.
   Refresh the original machine's three-call results separately when available.

Capture a future dated observation without overwriting this one:

```bash
python tools/snapshot_omega_gaas_experiments.py \
  --out docs/experiment_records/gaas_YYYYMMDD_HHMMSS
```

The artifact index hashes copied source bytes and lists original paths. Exported
validation CSVs and the summary are derived from the captured metrics/results.
