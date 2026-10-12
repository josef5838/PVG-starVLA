# Experiment C: latest-patch highway on rotate_qrcode

C starts fresh from seed 42 alongside A (PBS 84822.gaas). Its config is copied
from A's actual production config, preserving all training, data, normalization,
validation and checkpoint schedules: microbatch 4, accumulation 2, effective
batch 8, 20,000 updates / 160,000 observations, AdamW LR 3e-4, warmup 100,
cosine floor 1e-5, validation every 250 updates, state saves every 1,000.
Learned time embeddings remain unchanged. The backbone remains frozen.

C preserves the 408-register memory and adds the latest three cameras' patches
from the same joint 24-image Omega call. At production resolution this adds
3,108 spatial tokens, each with an explicit XY embedding, existing camera and
current-time embeddings, and a separate patch-type embedding. Spatial attention
runs after the existing register attention at blocks 4, 8, 12, 16, 20 and 24.
Each spatial residual has a zero-initialized scalar gate and nonzero attention
output projection. Spatial KV is prepared once per diffusion sample and reused.
Additional modules use a forked RNG; all common initial weights and subsequent
training RNG state remain identical to A. There is no trained-checkpoint resume.

CPU tests require nonvacuous zero-gate baseline equivalence, latest-patch ordering,
batch isolation, useful gate/spatial gradients, frozen backbone, cached KV and
complete image-route masking. The queued GPU preflight also verifies baseline
image/action/sample/initial-weight hashes and exact normalization statistics,
real patch dimensions, finite loss/actions, and gradients opening through four
disposable updates. Four updates allow both the initially zero diffusion output
head and the highway gate to open; the preflight's weights are then discarded.
Production runs only after every check passes, initializing again from seed 42.

Policy weights and latest complete optimizer/RNG/cursor state use writable shared
project storage under /projects_vol/gp_chuanxia.zheng/n2506021c/PVG-starVLA/checkpoints.
This storage-only difference avoids exceeding home quota and survives node loss.
An immutable source snapshot plus source/config hashes are retained in the study.
No independent batch or LR tuning is performed for C. Report measured throughput,
validation and eventual rollout outcomes before claiming an architecture benefit.
