# Single-call direct rotate_qrcode comparison on GAAS

This fresh run compares joint Omega contextualization against the retained
three-call direct policy from `rotate_qrcode_20261010_135404`. Both use the same
24 images: eight chronological observations from each of head/left/right
cameras. Joint input order is timestamp first, then head/left/right. Only one
Omega aggregator call is made per batch element's 24-image sequence. Batch
elements never attend to one another. Camera/register outputs are restored to
the original camera-first, timestamp-second order before the same embeddings
and direct 408-token action conditioning. The reference-camera embedding and
cross-image attention inside Omega change with grouping, as intended.

The original direct config, source checkpoint hash, seed 42, 80/10/10 episode
split, train-only normalization, 160 fixed validation anchors, 24-layer action
DiT, action horizon 16, all optimizer hyperparameters and zero dropout are
retained. The local transfer dataset's 106 files are verified against the
source bundle manifest. Native-history image, normalized action and sample-ID
hashes plus normalization statistics must match the source preflight exactly.
Common initial trainable weights must also match before GPU training starts.

The user authorized a different GPU and a larger effective batch if it improves
utilization, then requested an aligned total training budget. Benchmark three
disposable production-size configurations on the allocated GPU:

| Microbatch | Accumulation | Effective batch | Fresh updates | Observation budget |
| --- | --- | --- | --- | --- |
| 4 | 2 | 8 | 20,000 | 160,000 |
| 8 | 1 | 8 | 20,000 | 160,000 |
| 16 | 1 | 16 | 10,000 | 160,000 |

Each benchmark includes one warmup update and two timed full optimizer updates.
Choose a larger setting only with more than 5% higher observations/second and
at least 15% free allocated-memory headroom; stop increasing after OOM. Preserve
effective batch eight when throughput differences fall below this threshold.
Record all timings, GPU identity, memory and the selected production config.
These brief timings inform batch selection, not a final speed benchmark.

Keep learning rate 3e-4, AdamW betas/epsilon/weight decay, and gradient clip 1.0.
Warmup always covers 800 observations; cosine decay spans the full 160K budget
with floor 1e-5. Validation occurs every 2,000 observations, periodic resumable
state every 8,000 observations, and milestone weights at 80,000 observations.
Validation batches stay four regardless of training batch, preserving the
reference's diffusion-noise grouping and all 160 selected anchors. Best/final
weights are also retained. A two-update GPU preflight requires finite loss,
gradients, sampled actions, a frozen backbone, one 24-image call and 408 memory
tokens before fresh production initialization.

If effective batch sixteen is selected, total observations and schedule progress
are aligned, but optimizer update count, batch stochasticity and Adam moment
updates differ. GPU/software versions also differ. Report these limitations
when interpreting architecture results. This run does not by itself establish
a fully controlled architecture advantage or closed-loop success.

Latest complete training state uses node-local `/tmp` like the source experiment,
avoiding staging several large optimizer snapshots on the nearly full home
filesystem. On ordinary job exit or catchable termination the latest completed
state is copied to durable output; abrupt node loss can lose that local state.
Best/milestone/final policy weights stay on home storage. Only one fresh training
job is queued; no cancelled controls are relaunched and no rollouts are queued.

The PBS job first benchmarks, then checks GPU/input parity, then trains fresh.
Source is snapshotted before submission so subsequent workspace edits do not
change the queued job. See its `manifest.json`, `batch_tuning.json`,
`production.yaml`, `preflight.json`, training log and `stream_training_status.json`.
