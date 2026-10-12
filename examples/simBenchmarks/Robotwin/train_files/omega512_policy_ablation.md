# Omega512 action policy experiments

`Omega512DiT` reads dense final-layer patch features from the official
`vggt_omega_1b_512.pt` checkpoint. Loading verifies its configured SHA256 and
strictly loads the full model. The camera/depth heads are retained for reference
loading but are not evaluated by the action policy. No mesh or reconstruction
loss is used.

The baseline keeps Omega frozen in evaluation mode. Its 2048-dimensional patch
features pass through LayerNorm and a trainable 1024-dimensional linear
projection, with learned camera identity and normalized 2D position embeddings.
Three 4:3 cameras use the official balanced preprocessing at resolution budget
512: actual images are 448x592, with 28x37 patches per camera (3108 tokens).
The dataset's `preserve_native_images=true` prevents resizing raw 640x480
training frames before this shared preprocessing. Deployment advertises this
flag to the RoboTwin client, which also bypasses its preliminary image resize.

The decoder retains the previous 24-layer, 1024-wide DiT's short self-attention
sequence. Four arm/gripper state tokens and one explicitly encoded task token
condition a 16-step, 14-D absolute joint-position action chunk. Every fourth
decoder block adds visual cross-attention (six layers total). Visual keys and
values are computed once per observation and reused throughout DDIM sampling;
visual patches do not enter quadratic joint self-attention. Training uses the
existing 100-step diffusion schedule, noise-prediction MSE, and 20-step DDIM
validation. There is no classifier-free guidance at deployment. The policy
applies 10% whole-condition dropout and 20% state dropout during training.

The first two experiments differ only in `framework.readout.type` and artifact
paths. Common policy weights and subsequent random noise are seed-matched:

| Variant | Visual memory | Trainable modules |
|---|---|---|
| `dense` | All 3108 patch tokens | Projection, positions/views, task/state embeddings, action decoder |
| `state_query_local` | 64 queries per camera + 16x16 local grid per camera (960 tokens) | Same modules + two query-attention/FFN layers and state-to-query MLP |

Each camera's queries attend to that camera's patches; those Omega patches
already include interactions across views. The state-to-query offset starts at
zero. A shared dropout mask covers both state-to-query conditioning and the
decoder state tokens, including their biases. The local path pools each
camera's projected spatial grid independently and bypasses learned query
compression. This grid is a coarse local representation; it does not guarantee
preservation of every original patch detail.

Task conditioning uses a learned task-ID embedding, suitable for these fixed
tasks. An unknown or missing instruction is rejected. The dataset supplies
`rotate_qrcode`; deployment exposes its saved `instruction_override` so the
RoboTwin client supplies the same name rather than a randomly worded instruction.
This is not a general text encoder or a language-aligned Omega checkpoint.

`framework.omega.train_last_blocks=N` implements the third experiment: adapt
the final N within-view and across-view Omega blocks, leaving patch embedding,
earlier blocks, and geometry heads frozen. Adapted blocks use a separate 1e-5
learning rate. The launcher writes `adapt_last_blocks.yaml` (N=2, dense readout)
as a template but launches only the first two frozen-backbone runs. Select the
readout based on their results before running this adaptation experiment; the
template initializes a fresh action policy unless an explicit resume is supplied.

The runs inherit the previous fresh linear Surflo experiment's data split,
normalization, seed, action representation, batch, optimizer/scheduler, and
validation budget: 10000 updates, effective batch 8 (4 x accumulation 2), 80000
samples, saves at 5000 and 10000, validation every 250 updates, 32 validation
batches, 20 DDIM steps, execution horizon 8. Policy learning rates are 3e-4.

Checkpoint state includes all learned policy/adapted Omega weights, optimizer,
scheduler, RNG, and training cursor. Frozen Omega tensors are referenced by
path and SHA256 instead of copied into every checkpoint. Keep that reference
checkpoint available when evaluating or resuming. Missing policy/adapted
weights are rejected; only the verified immutable reference tensors can be
omitted. `framework.omega.weights_mode=embedded` can save all backbone tensors
when a standalone weight artifact is required.

Prepare with the standalone config (no previous Surflo checkpoint is needed):

```bash
export OMEGA_ROOT=/path/to/vggt-omega
export ROBOTWIN_DATA_ROOT=/path/to/RoboTwin/task_datasets
python \
  tools/launch_robotwin_omega512_ablations.py --prepare-only

# Inspect/run preflight, then launch the prepared manifest:
python \
  tools/launch_robotwin_omega512_ablations.py --launch-manifest PATH/manifest.json
```

The manifest records configs, hashes, GPU assignments, PIDs, logs, and source
snapshots. `tools/preflight_omega512_policy.py` verifies real-data gradients,
finite DDIM samples, and image dependence at fixed state/noise. Preflight
weights are discarded. `tests/test_omega512_policy.py` additionally verifies
official preprocessing identity, cached visual attention, state dropout,
matched initialization, adapted-weight saving, native-image transport, and
the next stochastic update after a real Accelerate save/load.

For GAAS, use the fresh configs and PBS templates described in
`docs/omega512_cluster_handoff.md`. They are prepared independently of the
two production jobs running locally. No GAAS jobs have been submitted.
