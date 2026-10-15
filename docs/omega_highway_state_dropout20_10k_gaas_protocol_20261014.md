# Fresh highway C with 20% state dropout, 10K updates

Train the existing current-frame patch highway architecture on rotate_qrcode
from its original seed-42 initialization for exactly 10,000 optimizer updates.
This is a fresh training run, with no warm start from the trained C checkpoint.
The reference is C's existing 10K milestone from PBS 84945.gaas.

Copy C's actual production configuration. Change the explicit-state dropout
probability from 0 to 0.2, stop at 10K rather than 20K, and use fresh output,
statistics-cache and durable optimizer-state paths. Preserve the effective
batch of 8 (microbatch 4 × accumulation 2), for 80,000 sampled observations.
An exact flattened configuration diff is registered in the study manifest.
All dataset file checksums are verified again on the GPU worker before model
preflight. The same 80/10/10 episode split, normalization and validation anchors
are retained. No new rollout evaluation is submitted with this training job.

The 20% probability is sampled independently per training example. When an
example is dropped, all four projected state tokens and their part embeddings
are zeroed together. The task token, register memory, spatial highway and robot
appearance in the images remain available. This changes explicit conditioning;
it does not erase robot information from the cameras. Validation and deployment
use full state without stochastic dropout.

Preserve C's original **20K cosine LR curve**, while stopping optimization at
10K, so that the comparison against C's 10K milestone has matching LR values
at every update. Set `trainer.lr_schedule_steps=20000` separately from the
10K stop/sample budget. Base and trainable module LR remain 3e-4, warmup100,
cosine floor1e-5, AdamW betas (.9,.95), epsilon1e-8, weight decay1e-8 and
gradient clipping1. The new scheduler override defaults to the existing
behavior when absent. This run stops before reaching the cosine floor.

Architecture is unchanged: one frozen Omega512 joint24 call, 408 restored
register tokens, 3,108 current-camera spatial patch tokens, learned time/view/
type/XY embeddings and six zero-gated spatial attention branches after decoder
blocks 4/8/12/16/20/24. Decoder width1024, depth24, heads16, action horizon16,
absolute joint targets, bf16 inference/training and float32 frozen Omega weights.
There are 358,505,492 trainable parameters. Condition dropout remains zero.

Keep DDIM20 validation every 250 updates, active-joint/gripper metrics on valid
future targets, latest full optimizer/RNG/cursor state every 1,000 updates on
shared project storage, best/final weights and the 10K milestone weights.

Five targeted CPU tests cover highway equivalence/order/gradients/KV reuse,
complete state masking and retained task/vision, masked-state gradient isolation,
full-state evaluation and exact first-10K LR parity in both scheduler builders.
They passed locally before submission. The worker repeats them before GPU
preflight. The real-batch GPU preflight verifies original common initialization,
images/actions/sample keys/statistics, real highway dimensions, zero-gate output
equivalence, complete state-mask semantics and finite spatial gradients through
four disposable updates at p=.2. Fresh production discards all probe weights
and initializes again from seed42.

One PBS GPU worker requests 12 CPUs, 128 GB host memory (cluster-adjusted),
and 72 hours maximum walltime. Source files and the config are hashed before
submission. Study:
`/home/n2506021c/PVG-starVLA-geometric-review/results/omega_camera_stream_training/rotate_qrcode_highway_C_state_dropout20_10k_gaas_20261014/`.
The exact output directory is recorded in its manifest, under
`/projects_vol/gp_chuanxia.zheng/n2506021c/PVG-starVLA/checkpoints/`.

Preparation: `tools/prepare_omega_highway_dropout_gaas.py`.
Worker: `tools/run_omega_joint24_gaas.py`.
GPU preflight: `tools/preflight_omega_spatial_highway.py`.
PBS: `examples/simBenchmarks/Robotwin/train_files/omega_highway_dropout20_10k_gaas_train.pbs`.

## Job registration

Submitted **86363.gaas** at **2026-10-14 18:25:59 SGT**. The first check found
it running on hpc-gaas-g04. All five CPU tests passed on that worker, all
registered dataset checksums matched, and the GPU preflight was in progress.
Training results are pending in this submission record.
Immutable source: `fd53980caccea42a120ef92823b964df4a4147aa`
(2,018 tracked files copied and hashed).

See [manifest](experiment_records/gaas_C_dropout20_10k_20261014/manifest.json),
[configuration](experiment_records/gaas_C_dropout20_10k_20261014/C_dropout20.yaml),
and [submission receipt](experiment_records/gaas_C_dropout20_10k_20261014/pbs_submission.json).

Checkpoint/output directory:
`/projects_vol/gp_chuanxia.zheng/n2506021c/PVG-starVLA/checkpoints/omega_joint24_rotate_qrcode_highway_state_dropout20_10000steps_gaas_20261014_102247/`.

The real-batch GPU preflight subsequently **passed** on NVIDIA H200, including
p=.2 dropout-route checks, original common initialization and data/statistics
parity, zero-gate baseline equivalence, frozen-backbone checks and working
spatial/gate gradients. Four disposable probe updates took 4.62–4.76 seconds
each, with 16.98 GiB peak allocation. Production initialized fresh and entered
training; early updates in the production log were finite. The preflight's
artificial gate=.1 sensitivity check is discarded and does not warm-start
production. See [GPU preflight](experiment_records/gaas_C_dropout20_10k_20261014/preflight.json)
and [actual production config](experiment_records/gaas_C_dropout20_10k_20261014/production.yaml).

## Completed training, 2026-10-15

PBS 86363.gaas completed with exit 0 at 08:47 SGT after 14h21m. It finished
10,000 updates and 80,000 sampled observations, retaining best/final/10K
weights and latest optimizer/RNG/cursor state. Final validation active-action
MSE was 0.067631; best was 0.055829 at step 8,250. Original C at the same
10K budget had 0.049457. These metrics measure expert-action fit and do not
establish visual grounding or closed-loop success.

The user subsequently requested [matched 10K rollout evaluation](omega_C_dropout20_10k_gaas_rollouts_20261015.md),
submitted as 86685.gaas at 11:54 SGT, using the same three test and five train
scenes and full state at inference. Rollout results remain pending.
