# Experiment Proposal: Temporal Labels and Spatial Highway for VGGT-Ω Action DiT

Revised 2026-10-12 after auditing the implemented camera-stream policy.

Implementation and current GAAS results are recorded in
[the GAAS experiment ledger](omega_gaas_experiments.md), observed 2026-10-13.
C has passed its CPU/GPU checks and is training; its registered 10K milestone
rollouts are complete: both scored 0/3 test and 0/5 train at the 10K milestone.
Training remains in progress at the latest dated observation. B0/Bfixed, Creg
and D/E remain future experiments; visual-grounding diagnostics are proposed
in the GAAS record before launching further full training jobs.

## 1. Scope and verified baseline

The first experiments target RoboTwin `rotate_qrcode` using
`OmegaCameraStreamDiT` with `streams.encoding=joint24` and
`streams.conditioning=direct`. This is the single-call counterpart of the
three-call direct policy in `rotate_qrcode_20261010_135404`.

The implementation being developed is in
`/home/n2506021c/PVG-starVLA-geometric-review`, on branch
`exp/rotate-qrcode-joint24-gaas-20261012`. Its initial joint-call implementation
and queue infrastructure are recorded in commit `5993024`. The older main
workspace contains other architectures and must not be used as an interchangeable
baseline. See the [joint-call protocol](../../PVG-starVLA-geometric-review/docs/omega_joint24_gaas_protocol_20261012.md).

Verified camera-stream baseline:

- Input: eight past/current observations from three physical cameras, 24 images
  total. Histories are chronological, oldest to newest, with initial padding by
  repeating the earliest available observation. All images are at or before the
  decision time. The dataset is recorded at 15 Hz.
- Joint Omega input order: timestamp first, then head/left/right camera. Each
  batch element is an independent 24-image sequence.
- Backbone: official Omega512, entirely frozen, layer index 23 features.
  Checkpoint SHA256:
  `c02da418b18bb01d0392598d3f6147366bcde1bb70fd08a5e3bf7925b0667934`.
- Decoder memory: one camera token and sixteen scene/register tokens per image,
  totaling `3 × 8 × 17 = 408` tokens. Direct conditioning retains all 408.
- Existing learned labels: camera identity, historical timestep and token type
  (camera versus scene). These are already present and are part of the baseline.
- State: 14-dimensional normalized joint/gripper command-state vector. The
  recorded state describes drive targets; it is not measured physical qpos.
- Task: existing single-task embedding for `rotate_qrcode`.
- Action: sixteen future absolute 14-dimensional joint/gripper commands.
  Eight observation frames and the sixteen-action prediction horizon are separate
  quantities. Rollouts initially execute eight actions per predicted chunk.
- Decoder: 24-layer, width-1024, 16-head action DiT. Four state-part tokens and
  one task token form the condition prefix; each future action is one token.
  Register cross-attention is applied after every fourth block, at blocks
  4, 8, 12, 16, 20 and 24 (one-based indexing).
- Objective: the existing diffusion noise-prediction loss, with 100 diffusion
  steps and DDIM20 evaluation. This implementation does not use flow matching.

The 61-dimensional proprioception and 23-dimensional actions belong to the
BEHAVIOR task setup. Its older `VGGTOmegaOFT` framework already pools over
camera/register/patch features from multiple layers. A highway there would
bypass learned pooling, and needs a separate protocol preserving that model's
state, text and action contracts. It is not the initial experiment here.

## 2. Motivation and limits of the references

[LagerNVS](https://arxiv.org/html/2603.20176v1) motivates retaining direct access
to dense image features instead of forcing all information through compact
latents. Its highway representation preserves per-source-image features; our
proposed gated action-decoder branch is an adaptation of that idea. Its
novel-view synthesis results do not establish robotic action improvements.
LagerNVS also reports a substantial benefit from end-to-end encoder fine-tuning.
We retain a frozen Omega initially to isolate decoder access to features.

The [Omega robotics experiment](https://arxiv.org/html/2605.15195v1) augments
OpenVLA-OFT's original input tokens with frozen scene registers. It does not
establish that registers alone preserve all manipulation-relevant visual detail.

The hypotheses are:

1. Explicit observation-age labels help the decoder use chronology. Learned and
   fixed relative-time representations are alternative encodings of known times.
2. Direct patch access supplies useful local information beyond register memory.
3. Additional historical patch access helps with occlusion or motion beyond
   access to current patches.

Improved action error is intermediate evidence; paired closed-loop task success
is the primary practical outcome. A highway advantage is not assumed.

## 3. Temporal labels

Preserve the existing register computation for the first highway comparison:

`features[c,t,j] = visual_proj(LN(raw[c,t,j])) + camera_embedding[c] + time_embedding[t] + type_embedding[j]`

Then apply the existing `memory_proj`, output normalization and camera-major
flattening. The intermediate width is 512; decoder memory width is 1024.

Known observation indices identify the time; learned embeddings determine how
those indices are represented. An eight-entry lookup is appropriate for the
fixed history window but does not enforce a distance structure or extrapolate
naturally to unseen ages. A separate fixed sinusoidal/Fourier relative-time
ablation should encode elapsed times `(-7,...,0)/15` seconds at this sampling
rate, retaining the other labels and projection paths.

Omega's pretrained spatial RoPE and first-image/reference-token distinction do
not supply chronological timestamps or physical camera identity. Labels added
after encoding guide the action decoder; they do not modify the frozen Omega's
internal temporal processing. Camera identity is a categorical label, not a
camera extrinsic transform or a guarantee of geometric alignment.

## 4. Spatial highway

### Token extraction and ordering

The cached layer-23 output has shape `[B,24,N_total,2048]`, where the feature
width concatenates local and inter-image features. Per image:

- Index `0`: camera token.
- Indices `1:17`: sixteen scene/register tokens.
- Indices `17:`: image patch tokens.

Reshape joint outputs to `[B,8,3,N_total,2048]` in timestamp/camera order.
The existing register path restores `[B,3,8,17,2048]` before flattening.
Select current patches as `features[:,7,:,17:,:]` in the timestamp-first layout.
Do not apply a latest-timestamp slice to a camera-first tensor without adapting
its axes. Assert `patch_start == 17` and all expected dimensions at runtime.

Extract selected patches before the current helper discards them. Both memories
come from one Omega encoding, with no second backbone call. All 24 images still
participate in that encoding even when only current patches are retained.
Thus current patch features already contain historical context; this experiment
changes the decoder's direct spatial access.

At the current processed image size `448 × 592`, patch size 16 gives
`28 × 37 = 1,036` patches per image. Read the actual grid from preprocessing:

| Spatial selection | Patch tokens | Register tokens retained |
| --- | ---: | ---: |
| Current timestamp, three cameras | 3,108 | 408 |
| Current plus oldest timestamp | 6,216 | 408 |
| All eight timestamps | 24,864 | 408 |

Keep spatial patches as individual tokens. Add explicit normalized 2D grid
coordinates and camera identity. Add observation-age labels to every selected
historical frame. Project to width 512 for compatible camera/time labels, then
to decoder width 1024; the new spatial projection must preserve the existing
register computation. Per-camera grid coordinates describe image locations,
not a shared metric 3D frame.

### Decoder integration

Retain the original block and register-attention updates. After each existing
visual-attention location, add a separate read-only spatial branch:

`Q = existing_block_and_register_attention(Q, state, task, diffusion_time, register_memory)`

`Q = Q + gate[layer] * spatial_attention[layer](LN(Q), spatial_KV[layer])`

Initially use the same six locations, every fourth block. An every-second-block
variant is a later capacity/latency experiment. The spatial branch does not
update visual memory. Preserve existing diffusion/state/task conditioning.

Compute spatial feature projections once per observation encoding, and each
layer's keys/values once before the denoising loop. Reuse these caches across
all DDIM steps. The short condition/action sequence queries the visual memories;
avoid self-attention over a concatenation of all spatial tokens and actions.

Use a learnable gate per branch, initialized to zero, with nonzero attention
output initialization. This preserves the initial baseline function while
allowing gate gradients; branch weights can receive gradients after the gate
opens. Do not initialize both gate and attention output to zero. Disabling the
highway must skip its update and recover the baseline with matched inputs/noise.
Initialize new modules in a separate RNG context so common parameters and the
fresh-production RNG state remain matched.

## 5. Experiment matrix and priority

| ID | Register memory | Additional decoder memory | Purpose |
| --- | --- | --- | --- |
| A | Existing learned time/camera/type labels | None | Joint-call direct baseline |
| B0 | A with time labels disabled only | None | Test whether time labels help |
| Bfixed | A with learned time labels replaced by fixed relative-time encoding | None | Compare time representations |
| C | Exactly A | Current three-camera patches | First highway experiment |
| Creg | Exactly A | Extra branch reads register features | Control for added decoder capacity |
| D | Exactly A | Current plus oldest three-camera patches | Incremental historical spatial access |
| E | Exactly A | All 24 images' patches | Optional full spatial access |

First comparison: **A versus C**. Hold the learned time representation fixed.
Do not add another temporal/camera embedding to the baseline. Keep all encoder
freezing settings fixed; LoRA or backbone fine-tuning is a separate later study.

If C is useful, run Creg before attributing gains specifically to spatial
information, then D. Creg should retain the same added attention and projection
parameter budget where possible; report its different memory length and compute.
Temporal B0/Bfixed experiments are separate from the highway comparison. Defer
E until selected-frame variants justify its memory and latency cost.

## 6. First training job and matched budget

The first new training job is **`rotate_qrcode_joint24_direct_current_patch_highway`**
(experiment C), compared against A. Train fresh from the same seed and official
Omega weights, with shared baseline parameter initialization; do not warm-start
C from a trained A checkpoint for this comparison.

A is already running as GAAS PBS job `84822.gaas`. As checked on 2026-10-12,
its GPU preflight passed and its batch benchmark selected microbatch four with
accumulation two on an H200. Larger microbatches offered less than the registered
5% throughput improvement. These are observed launch facts, not final training
or task-success results. Use its final `production.yaml` as C's configuration
reference and record a field-by-field diff.

Primary comparison settings:

- Dataset: the same 100 right-arm trajectories and 8,341 observations.
- Identical saved 80/10/10 episode split, seed 42, and train-only normalization.
- Microbatch four, accumulation two, effective batch eight, 20,000 updates:
  **160,000 sampled training observations per model**. Keep C's effective batch
  and update count matched to A; benchmark C's fit before production launch.
- AdamW LR 3e-4, betas (0.9,0.95), epsilon 1e-8, weight decay 1e-8,
  gradient clipping 1.0, and all conditioning dropout zero.
- Warmup 100 updates; cosine decay over 20,000 updates to floor 1e-5.
- Validation every 250 updates using the same 160 fixed validation anchors,
  batch four, DDIM20 and seed 42. Checkpoint selection uses active right-arm
  joints/gripper, indices `[6,7,8,9,10,11,13]`, and valid future targets only.
  Keep the training loss and padding convention unchanged.
- Save resumable state every 1,000 updates, milestone policy weights at 10,000,
  and best/final weights. Preserve complete optimizer/scheduler/RNG/cursor state
  using the registered storage convention and record node-local storage limits.

If a later comparison uses a different effective batch, preregister both models
at that setting. Match 160,000 observations and scale warmup, validation,
checkpoint and milestone intervals by samples; document the remaining difference
in optimizer dynamics. Do not independently tune A and C to different batches
and call the result an architecture-only comparison. Record actual GPU and
software differences if hardware cannot be matched.

## 7. Diagnostics and rollout evaluation

### Chronology and permutation controls

A pure joint permutation of memory tokens and their attached labels should
leave decoder cross-attention invariant within numerical tolerance. Treat this
as an implementation control, not evidence that chronology is ignored.

Test chronology by changing feature-to-time-label assignments while retaining
current observations and camera synchronization. For a decoder-only diagnostic,
reuse fixed Omega features and assign incorrect ages to historical timestamps.
Separately test synchronized historical image shuffling before Omega encoding.
Keep its first reference image fixed or report the reference-image change as an
additional intervention. Keep current images fixed. A current-repeat comparison
can test dependence on historical visual content independently of label order.

### Visual and spatial dependence

Hold robot state, task and diffusion noise fixed while testing matched images,
mismatched images, removal of all visual routes, and disabling the spatial
branch alone. The all-visual mask must cover both register and highway paths.
Record gate values, attention-residual magnitudes and action sensitivity, but
interpret them alongside task success rather than as causal evidence by themselves.

### Evaluation contract

Use paired offline test anchors and paired closed-loop scenes/diffusion seeds.
Start with execute horizon eight and DDIM20, on right-arm scenes within the
training support. Verify scene identity, local simulator feasibility, and raw
and normalization-roundtrip expert replay before interpreting policy failures.
Transfer original raw replay data if required by those controls; Git evidence
alone does not include all simulator/HDF5 inputs.

Preregister a larger fresh stable right-arm cohort before inspecting policy
outcomes. Report grasp/lift/rotate/place/release phases, active-arm action error,
paired task-success uncertainty, and failures by scene/object subgroup.
Rollout success is the primary endpoint; lower offline MSE alone is insufficient.

Measure encoder latency, visual projection/KV preparation, denoising latency,
end-to-end inference time, peak GPU memory, training observations/second and
trainable parameter count. Profile steady-state rollouts separately from the
brief batch-selection benchmark. Report equal observation budgets and actual
compute costs; an added highway increases compute per observation.

## 8. Minimal implementation and preflight requirements

The smallest change testing C is to retain current patches from the existing
Omega result and add the optional spatial cross-attention branch. No new
observation-fusion transformer or encoder training strategy is required.

Before full training:

1. Verify extraction, timestamp/camera/grid ordering and initial-history padding
   on real samples; no observation may come from after the decision time.
2. Verify one 24-image Omega call, 408 register tokens and 3,108 spatial tokens
   at the current resolution, with batch-element isolation.
3. Verify shared initial baseline weights/RNG and unchanged data, action,
   normalization and validation-anchor hashes. Validate the config diff.
4. Verify highway-disabled output and zero-gate initial output match A under
   paired noise. Test nonzero gate gradients; after the gate opens, test spatial
   projection and attention gradients. Require no frozen Omega gradients.
5. Run four disposable production-size optimizer updates with finite loss,
   gradients and sampled actions. This allows the initially zero diffusion
   output head and then the spatial gates to open before checking spatial
   projection gradients. Record memory and paired DDIM20 inference timings;
   repeat timings later for a stable benchmark.
6. Discard disposable weights and launch C fresh. Record source/config/backbone
   hashes and full training state. The existing A job must retain its source
   snapshot and baseline configuration.

The frozen-encoder path may detach raw features. Supporting trainable Omega
adapters requires a separate implementation change: the current stream helper
uses `no_grad`, detaches features and rejects trainable Omega blocks. Do not
claim adapter support until that gradient path is implemented and tested.

Deliverables are the audited baseline, optional highway module/config, shape
and gradient checks, deterministic sensitivity/efficiency evaluation, and paired
A/C training/rollout results. Temporal representation switches and D/E follow
as separate controlled experiments.

## 9. C submission record — 2026-10-12

C was implemented and queued as **PBS 84945.gaas**, with an immutable source
snapshot at commit `a83d68f47ad4e2c39868b8d8b6f3c545ebda2994` in the geometric
review worktree. All 24 targeted CPU tests passed. The config audit against A's
actual production config permits only the optional highway and output/storage
paths; effective batch 8, 20,000 updates, LR 3e-4 and all data/schedule settings
remain matched. C has 358,505,492 trainable parameters, including 27,049,990 added
highway parameters. GPU preflight must pass before fresh production training.

Study/config/logs:
`/home/n2506021c/PVG-starVLA-geometric-review/results/omega_camera_stream_training/rotate_qrcode_highway_C_gaas_20261012/`.
Policy weights and complete resumable state use the writable shared project
checkpoint directory recorded in its manifest. A remains PBS 84822.gaas with
its existing source snapshot and configuration.
