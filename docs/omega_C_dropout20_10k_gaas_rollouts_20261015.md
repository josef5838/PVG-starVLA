# Highway C with 20% state dropout: matched 10K rollouts

Evaluate the 10,000-update milestone from training job 86363.gaas, with the
same 80,000-observation budget as the previous A/C 10K evaluation. Training
completed successfully on 2026-10-15 at 08:47 SGT. Validation active-action
MSE was 0.067631 at 10K (original C: 0.049457); its best was 0.055829 at
8,250. These errors do not establish rollout success or useful visual grounding.
Select the 10K milestone for the matched-budget comparison, rather than the
best-validation checkpoint.

Reuse the previous eight registered scenes unchanged: test episodes 2/4/24,
seeds 91/142/62; train episodes 0/11/22/34/44, seeds 131/181/9/256/240.
Use full explicit state and vision, with no stochastic state dropout during
inference. Keep native 640x480 cameras, camera order head/left/right,
eight consecutive history frames with initial left padding, the rotate_qrcode
instruction, bf16 inference with float32 frozen Omega, DDIM20, diffusion seed
4000 plus chunk index reset per episode, and eight commands per replan.

Preparation verifies completed training, split membership, unchanged highway
architecture aside from training dropout, exact normalization and retained
scene-reference/config hashes. The worker verifies checkpoint/config/statistics
hashes and all eight initial scene states, object poses/models and camera images
using the prior tolerances. A real-checkpoint GPU preflight verifies full-state
and vision metadata, nonzero state/task tokens, one Omega call per encoding,
finite action predictions and repeated seeded prediction agreement. Deployment
uses eval mode and its conditioning path does not apply training dropout.

Run one GPU job through the existing paired-rollout worker. Record eight videos,
action/geometry traces, separate train/test task success and supporting phase
diagnostics, and fully decode all videos before marking completion. Archive
immutable source hashes, the manifest and submission receipt. This is a small
paired diagnostic suite; useful grounding requires the subsequent object-position
intervention, and eight scenes do not support a precise generalization estimate.

Study: `results/omega_camera_stream_evaluation/rotate_qrcode_C_dropout20_10000_20261015/`.

Large outputs:
`/projects_vol/gp_chuanxia.zheng/n2506021c/PVG-starVLA/evaluations/rotate_qrcode_C_dropout20_10000_20261015/`.

Preparation: `tools/prepare_omega_highway_dropout_rollouts_gaas.py`.
Worker: `tools/run_omega_paired_rollouts_gaas.py`.
PBS: `examples/simBenchmarks/Robotwin/eval_files/omega_paired_10k_gaas_rollout.pbs`.

## Submission

Submitted **86685.gaas** at **2026-10-15 11:54 SGT**. The first scheduler
check found it running on **hpc-gaas-g10**. Scene and GPU preflights and
rollout outcomes are pending in this submission record. No additional
checkpoint or conditioning variants were queued.

Immutable source: `a12c0c3f750a9aac83f67516088022ebce3e2edf`
(2,029 tracked files copied and hashed).

See [manifest](experiment_records/gaas_C_dropout20_10k_rollouts_20261015/manifest.json),
[submission receipt](experiment_records/gaas_C_dropout20_10k_rollouts_20261015/pbs_submission_C_dropout20.json),
and [training validation summary](experiment_records/gaas_C_dropout20_10k_rollouts_20261015/training_validation_summary.json).
