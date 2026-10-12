# Independent camera histories for action conditioning

Registered 2026-10-09. Initial task: RoboTwin `adjust_bottle`.

## Motivation and limits of the reconstruction checks

Process each physical camera's past images independently with a shared frozen VGGT-Ω, then learn to combine its camera and scene/register tokens for action prediction. Temporal overlap within a camera may provide more useful input than three weakly overlapping simultaneous views. This is a representation hypothesis; accurate reconstruction is neither demonstrated nor required by the action objective.

The completed RoboTwin and BEHAVIOR checks used one joint eight-image call per camera. They exposed substantial wrist-camera pose errors. The BEHAVIOR ablation with all 24 images in one call had mixed temporal-pose results and substantial cross-camera registration errors. A common predicted coordinate frame does not guarantee accurate reconstruction. These selected checks do not establish which tokens or architecture improve closed-loop control.

Independent calls express geometry in separate reference-camera frames with nonmetric scales. Explicit point-cloud merging needs alignment. Learned feature fusion can operate directly on the common encoder's latent vectors, while camera identity, time, and proprioception help relate them to robot actions. Scene tokens are associated with each input image but contextualized by the full input set; camera tokens are latent features, not camera matrices.

Primary references: [Ω architecture and register applications](https://arxiv.org/html/2605.15195v1), [learned query extraction in BLIP-2](https://proceedings.mlr.press/v202/li23q.html), [temporal RoPE mechanism](https://arxiv.org/abs/2104.09864). Ω's published robotics experiment adds scene tokens to an existing visual policy; it does not establish that visual tokens can be removed without a performance cost.

## Initial registered comparison

| Variant | Ω inputs | Action memory |
|---|---|---|
| Direct 408 | Three independent eight-image camera histories | All 408 projected camera/register tokens |
| Fused 32 | Identical histories and frozen features | Two fusion self-attention layers, then 32 learned cross-attention queries |

The intervention includes both learned fusion and compression. Any observed difference cannot yet be attributed separately to self-attention versus token count. Direct action-decoder cross-attention itself can combine observations; it is a substantive baseline.

Each image contributes one camera and sixteen scene tokens, each 2048-D in the released features. Three calls × eight images × seventeen tokens = 408. Both variants discard patch tokens only from the action-conditioning output. Ω still processes image patches internally. Feature extraction calls the released Ω aggregator and slices the final cached layer's first seventeen tokens, which are the same features returned by the full model; camera/depth prediction heads are skipped during action training.

Both variants use LayerNorm and a 2048→512 projection, additive camera-ID embeddings (head/left/right), learned chronological time embeddings (oldest through current), and camera/scene token-type embeddings. The direct variant flattens all camera/time/token axes. The fused variant uses a two-layer pre-LayerNorm transformer, width 512, eight heads, FFN width 2048, GELU, and zero dropout. Thirty-two learned queries cross-attend to the fused sequence, followed by a residual FFN and LayerNorm. Both variants use the same 512→1024 projection and final LayerNorm into the action decoder.

**No temporal RoPE is used**, either in the new conditioning module or Ω's cross-image attention. Ω retains its pretrained spatial RoPE. Learned time embeddings provide basic temporal labels in both variants. Proprioception and task conditioning enter the decoder identically and do not enter the fusion module. Ω remains frozen and in evaluation mode.

## Data and causal contract

- Dataset: `/export/home/haoliang/projects/starVLA/playground/Datasets/RoboTwin/adjust_bottle`; 50 episodes, 3249 recorded frames, 15 Hz.
- Three cameras: `cam_high`, `cam_left_wrist`, `cam_right_wrist`, in that order.
- For observation index t, each camera provides t−7,…,t, oldest first. The first image of each independent Ω call is its reference. History before episode start repeats the first recorded image; no future images and no cross-episode history are permitted. Actual timestamp span for a full window is 7/15 seconds.
- Preserve the embedded 640×480 RGB images and use the official balanced 512-budget preprocessing. These converted images differ in resolution from the 320×240 source images used in the earlier reconstruction check. Record the resulting Ω tensor shape in preflight and runtime status.
- Fifty episodes are split into 45 training and five validation episodes, deterministically with seed 42. Save IDs before launch. Both variants use the identical split. Normalization statistics are computed using training episodes only and then reused for validation; dataset-wide preexisting statistics must not replace split statistics.
- Use the existing dataset's absolute 14-D joint/gripper action representation and action-window indices 0,…,15. The converted action field is the supervision source; do not reinterpret it as measured next-step state. The source-to-command temporal meaning is a conversion dependency, not revalidated by this experiment.
- Robot state is the current normalized 14-D vector. The task is represented by the existing single-task `adjust_bottle` embedding, not a language model. This initial experiment does not evaluate language generalization.

## Matched training settings

Train both variants fresh with seed 42, the same common initial weights, and the existing Omega cross-attention action diffusion decoder: width 1024, 24 layers, 16 heads, visual cross-attention every four layers, action horizon sixteen, four state-part tokens plus one task token. The fused branch is initialized inside a separate RNG context so it does not perturb common decoder weights or diffusion RNG.

Initial budget: **10,000 optimizer updates per variant**, effective batch eight (microbatch four, accumulation two), totaling 80,000 sampled observations per run. One independent process runs on each available GPU. AdamW, learning rate 3e-4 for all trainable components, betas (0.9,0.95), eps 1e-8, weight decay 1e-8, gradient clipping 1.0, warmup 100 updates, cosine schedule with minimum LR 1e-5. All conditioning dropout is zero in this first comparison to isolate the architecture. Mixed precision is bfloat16; Ω is pretrained and immutable.

Record trainable parameter counts, peak GPU memory, throughput, losses, and conditioning shapes. Save model and complete optimizer/scheduler/RNG/cursor state every 1000 updates and at completion. Frozen Ω weights are referenced by their verified SHA256 rather than duplicated in each checkpoint. Training logs and W&B artifacts stay local/offline. Record the first sixteen microbatches' episode/frame identities to verify matched sample order.

## Validation and interpretation

Every 250 updates, evaluate the same deterministic validation sampling sequence: up to 32 microbatches of four held-out-episode observations, twenty DDIM steps, seed 42, normalized action MSE. This is a fixed sampled validation subset, not an exhaustive sweep of every held-out frame. Restore training modes and RNG after evaluation. Select best weights using this validation error only; preserve regular milestones for later comparison.

Offline action error is an intermediate measure. The decisive later evaluation is matched closed-loop RoboTwin task success using unseen simulator seeds, the same start scenes, diffusion seeds, execution horizon eight, and twenty DDIM steps. Also report approach/grasp/contact/lift failures. A client must supply the same chronological eight-step history at each decision; feeding only a current image would invalidate evaluation. Closed-loop rollouts are a later stage, not automatically launched by this training protocol.

Higher action success would support the conditioning strategy under these settings. Lower reconstruction error is not a necessary success criterion. A lower held-out action MSE alone does not establish better task success. This single task and single training seed cannot support benchmark-wide conclusions.

## Required checks before launch

1. Verify native images, history order and current-frame inclusion against raw loaded frames.
2. Verify three separate Ω calls per batch, each containing eight images per sample, with no attention between different physical cameras or batch elements inside Ω.
3. Verify 408 raw tokens, direct memory length 408 and fused memory length 32; no patch tokens in action memory.
4. Run two disposable training updates at the production microbatch size. Require finite loss, gradients in projection/time embeddings and fused queries, no Ω gradients, and finite sampled actions.
5. Change a historical wrist image while holding state, current RGB, noisy actions and diffusion timestep fixed. Verify action predictions can respond to history.
6. Verify identical common initial-weight, input-image and target-action hashes between preflights. Verify identical configs except conditioning variant and output paths.
7. Verify policy checkpoint serialization excludes frozen Ω and round-trips correctly. Discard preflight updates and launch fresh models.

## Planned follow-ups

Hold the action decoder, split and observation budget fixed while testing scene-only versus scene-plus-camera tokens; joint-24 versus independent-three-call encoding; fused 32 versus 64 queries; fusion without compression; learned time embeddings versus temporal RoPE; and latest-image tokens versus all-history tokens. Include an ordinary temporal visual encoder and a retained visual-policy branch to distinguish Ω-specific value from temporal context and information loss from compression. Repeat promising results across training seeds and additional tasks before drawing broader conclusions.

## Commands and artifacts

Prepare configs and a manifest:

```bash
/export/home/haoliang/miniconda3/envs/starVLA/bin/python tools/launch_omega_camera_streams.py
```

Run `tools/preflight_omega_camera_streams.py --config <variant.yaml> --out <variant_preflight.json>` with the variant's GPU selected. After both pass, launch with `tools/launch_omega_camera_streams.py --launch-manifest <manifest.json>`. The launcher verifies parity and archives source snapshots/hashes before starting either process. Each run has `stream_training_status.json`, `stream_training_metrics.jsonl`, `initial_sample_order.jsonl`, split/statistics artifacts, resumable checkpoints, and saved full configuration. The exact prepared paths and launched PIDs are recorded in the manifest under `results/omega_camera_stream_training/`.
