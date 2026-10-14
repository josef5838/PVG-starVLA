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
