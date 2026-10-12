# Omega512 experiments: local machine and GAAS

Updated 2026-10-08 UTC. **GAAS is prepared for independent fresh experiments;
no GAAS training or GPU smoke job has been submitted.** The original dense and
query/local jobs completed 10000 steps and each scored 0/3 on the initial local
evaluation, with videos. Their latest complete 10000-step training states remain
available; intermediate 5000-step copies were pruned during the October 6 cleanup.
The initially copied local policy/optimizer snapshots were removed from the
GAAS payload. Code, task data and the public Omega reference are retained.

## Local jobs and next steps

October 8: the eight-trajectory policy completed 20K updates and its scheduled evaluations. The 10K checkpoint scored 1/24 successes; 20K scored 0/24 despite lower recorded-observation action error. Additional three-scene video evaluations of both checkpoints run sequentially on GPU 0, queue PID 633749, matching the 640×480 training input dimensions (older simulator inputs were 320×240). The localization probe resumed collection from 13 verified caches on GPU 0, PID 634427; an initial wrist-pixel check was replaced by audited wrist differences plus a new 2mm object-position check, while model/state/head RGB checks remain. GPU 1 is occupied by another user's processes. Details and limits: [October 8 report](../results/omega512_protocol/20261007_185108/progress_results_20261008.md). The October 7 launch allocation below is historical.

Current local run, October 7: [protocol experiments 1 and 2](../results/omega512_protocol/20261007_185108/README.md) are running. GPU 0/PID 506587 trains a fresh dense policy on eight varied right-arm trajectories for 20,000 updates, batch eight, original normalization/settings, resumable saves every 5,000. It records fixed 128-observation probes every 500 updates and full-trajectory probes at saves. Matched-scene videos are queued at updates 10,000 and 20,000 for all eight scenes and diffusion seeds 4000/4001/4002. GPU 1/PID 506588 runs synchronized simulator label collection, frozen Omega feature extraction, and visual-only object/EE/displacement probes with 24/8/8 trajectory splits and image-shuffle controls. Regenerated probe images and poses are paired directly; the [data amendment](../results/omega512_protocol/20261007_185108/probe_data_amendment.json) documents why later-frame labels could not be recovered from the original recordings. Both preflights and the save/load next-update test passed. These are local experiments; no GAAS jobs were submitted.

Exact configs, selected source episodes, seeds, coordinate definitions, source hashes and launch commands: [manifest](../results/omega512_protocol/20261007_185108/manifest.json). Live status: `python tools/omega512_protocol_status.py --manifest results/omega512_protocol/20261007_185108/manifest.json`. Training log: `reference_training.log`; probe log: `visual_probe.log` in that run directory. The following October 6 results and launch identifiers are historical.

Storage update, 2026-10-06: cleanup reclaimed about 221 GiB and left about
223 GiB free on home. The two causal controls and their automatic evaluations
have completed; their PIDs below are historical. Extra seed configs remain
prepared. Latest complete
`checkpoints/steps_10000` states and best-model states were retained. Duplicate
`final_model/training_state` copies were removed after byte-for-byte comparison;
`final_model/TRAINING_STATE_LOCATION.txt` names the retained resume directory.
The full audit is `results/storage_cleanup/20261006/report.md`.

The causal conditioning investigation at
`results/robotwin_omega512_causal/20261005/report.md` has completed its selected
training controls. Vision-only, state-only, and the dense reference all scored
0/2 on matched right-side evaluation scenes at 10,000 updates. Its record is
`results/robotwin_omega512_causal/20261005/training_controls/manifest.json`.
Three seed-43 repetition configs are prepared but unlaunched. They are
not submitted on GAAS. Data coverage remains right-arm only for this causal
comparison; full-task data collection and visual-goal supervision are follow-ups.

The minimum-case investigation launched at 07:53 UTC on October 6 has completed:
`results/robotwin_omega512_minimum_case/20261006_075346/manifest.json`.
GPU 1 trains a fresh dense policy on converted episode 0 only (source episode
51, simulator seed 131), with the original 90-episode normalization statistics,
frozen Omega, unchanged conditioning dropout and optimizer, 10,000 updates,
and complete resumable saves at 5,000 and 10,000. A fixed-noise action probe
enumerates all 82 training observations every 250 updates; it is explicitly
training-trajectory error, not held-out validation. After training, the queue
evaluates its checkpoint on the matching scene with execution horizons 1/2/8.
GPU 0 replays three recorded training trajectories through the production
normalization transform and ModelClient, then evaluates each original dense
and query/local 10,000 checkpoint on the same verified scenes (seeds 131,
162, 91). Failed normalized expert replays receive raw-action comparators.
Videos and action/state/end-effector/object/controller-duration traces are saved.
Training PID 352854, diagnostic worker PID 352855, queue PID 352849 are launch
identifiers; all workers exited successfully and both GPUs were idle at completion on October 6.
The 10,000-step overfit policy reached full-trajectory action MSE 0.007498
(best probe 0.007122 at update 9,000). Expert replay scored 3/3; both original
policies scored 0/3 on the verified training scenes. The overfit checkpoint
succeeded on its exact training scene with execution horizon 8, and failed
with horizons 1 and 2 (one rollout each). Chunk noise is seeded by chunk index,
so changing horizon also changes its schedule; repeatability is not established.
Videos, traces and complete resumable 5,000/10,000 states are saved. Full results:
`results/robotwin_omega512_minimum_case/20261006_075346/completed_results.md`.

State correction: the current RoboTwin observation path uses commanded joint
drive targets, not measured physical qpos. They enter four action tokens, and
also the query offsets in query/local. Explicit route masks were added without
changing the default checkpoint contract. Read the causal report before
interpreting state masking or low average validation MSE as evidence of success.

The following original training table is historical.

Repository: `/export/home/haoliang/projects/PVG-starVLA-snapshot`.
Python: `/export/home/haoliang/miniconda3/envs/starVLA/bin/python`.
Experiment: `results/robotwin_omega512_ablation/rotate_qrcode_20261005_053805`.

| GPU | Variant | Parent PID | At 06:50:56 UTC | Last complete best state then |
| --- | --- | --- | --- | --- |
| 0, H100 NVL | dense | 284321 | 5250 / 10000 | 5000, validation MSE 0.0423424 |
| 1, H100 NVL | state_query_local | 284322 | 5233 / 10000 | 4500, validation MSE 0.0464712 |

This table is a timestamped observation, not current live progress. Refresh:

```bash
python tools/omega512_training_status.py \
  --manifest results/robotwin_omega512_ablation/rotate_qrcode_20261005_053805/manifest.json
```

Each has four data workers; those workers are part of the two jobs. No
simulation evaluation, policy server or reconstruction job was running when
migration preparation began. The third final-block adaptation experiment has
not been launched. Logs are `dense.log` and `state_query_local.log` in that
experiment directory; checkpoint paths are recorded in `manifest.json`.

Next on this machine: compare recorded-expert replay, original-policy seen-scene
success, and single-trajectory overfit with execution-horizon controls. These
separate data/controller problems from basic imitation and generalization.
Noise loss and normalized-action MSE do not establish rollout success.

Checkpoints remain training-resumable locally. Complete state directories
contain model, optimizer, scheduler, RNG and trainer/data cursor plus
`complete.json`, written last. Periodic saves occur at 5000/10000, with
additional best/final states. Best states are overwritten; never copy one
while its completion marker is absent or changes.

## Fixed experiment contract

- Task: `rotate_qrcode`, encoded as a learned task ID at train and test.
- Cameras: head, left wrist, right wrist; simultaneous native RGB, no history.
  Frames are 640x480 (width x height). Official balanced Omega512 preprocessing
  produces 448x592 tensors and 28x37 patches per view.
- Backbone: verified official frozen Omega512, final frame/inter-frame feature
  concatenation (2048 dimensions). LayerNorm and learned projection to 1024,
  with view and normalized 2D position embeddings.
- Dense readout: 3108 tokens. Query/local: 64 queries plus a pooled 16x16 grid
  per camera, 960 tokens total, two query readout blocks.
- Action DiT: 24 layers, width 1024, 16 heads. Four state-part tokens, one task
  token, 16 action tokens; six cross-attention layers with cached visual K/V.
  Predict noise in 14-D absolute joint/gripper action space; 100 diffusion
  training steps, 20 DDIM inference steps; execute 8 then replan.
- Packed state/action: left six joints, right six joints, left/right grippers.
  The adapter reorders this for the simulator. The server normalizes raw state
  and unnormalizes actions exactly once.
- Budget: 10000 updates, batch 4 x accumulation 2 = 8, 80000 samples.
  Episode split 90/10, seed 42, matched to prior Surflo experiments. Validation
  every 250, at most 32 batches, deterministic 20-step DDIM. LR 3e-4, cosine
  minimum 1e-5, warmup 100; state dropout 20%, condition dropout 10%.
- No surface loss, geometry-head forward pass, flow matching or text encoder.
  `robotwin_surflo_rotate_qrcode` is the inherited registry key; the Omega
  policy does not load Surflo.

## Cleanup and validation

The standalone `examples/simBenchmarks/Robotwin/train_files/omega512_robotwin.yaml`
replaces the launcher's dependency on an old Surflo results directory. Set
`OMEGA_ROOT`, `OMEGA_CHECKPOINT`, `ROBOTWIN_DATA_ROOT`, or explicit config paths.
The live production jobs already have explicit paths and retain their launch
configuration. Their original launch code is saved in the experiment directory.

Deployment fixes: load complete `config.full.yaml` with legacy `config.yaml`
fallback; keep float32 Omega backbone parameters when the server casts the
policy to bfloat16; interpret image metadata as width/height. The accessed-only
snapshot omitted native-image and fixed-instruction fields in these runs.
Four migration tests cover these issues and fresh target configuration budgets.

The downloader pins the verified reference revision. The payload carries the
exact installed Omega source, including the Q/K normalization and DINOv3 RoPE
compatibility corrections used for successful reproduction. Preserve this
source until a replacement passes strict loading and preprocessing checks.

Local validation: **45 tests passed**: Omega policy 8, migration 4, RoboTwin
contracts 11, trainer resume 1, config overrides 21. The standalone launcher
also successfully prepared matched configs without launching jobs. GPU
preflights are prepared for GAAS; the local production GPUs were not reused.

GAAS validation: the payload integrity check passed and **all 45 CPU tests
passed in its existing torch 2.4.1 environment**. Logs are under
`results/TransferBundles/omega512_20261005/gaas_cpu_checks` on GAAS and the
bundle's `validation/gaas_cpu_checks` locally. GPU, real-backbone backward,
websocket inference and simulator rollouts remain pending; jobs are unsubmitted.

## GAAS workspace and environment

Existing checkout: `/home/n2506021c/PVG-starVLA`.
Payload: `results/TransferBundles/omega512_20261005` under that checkout.
Both checkouts started at Git base `149305e7a5c63381101e9aca9b561e58186677cd`;
the transfer includes current uncommitted implementations. Nothing was pushed
or committed. SSH works as `n2506021c@172.21.33.7` with cipher `aes256-ctr` and
MAC `hmac-sha2-256-etm@openssh.com`; the local `gaas` alias is absent.

GAAS uses PBS, project `gs_ccds_chuanxia.zheng`, queue `gpu_as`. The login node
has no GPU. The account cannot create `/scratch/n2506021c`, so files are in
`/home`. The payload contains source, public Omega weights (~4.58 GB), the
RoboTwin task dataset (~1.7 GB), split/normalization metadata and provenance;
no local policy weights or optimizer states, no credentials. `manifest.json`
records hashes and source package versions. Rebuild absolute-path pickle
caches on the target.

Omega reference SHA256:
`c02da418b18bb01d0392598d3f6147366bcde1bb70fd08a5e3bf7925b0667934`.
Reference revision: `1041e80fc0e911235d3426b0a3d9a81075111a53`.

Local reference: Python 3.10.20, torch 2.7.1+cu128, torchvision 0.22.1+cu128,
transformers 4.57.0, accelerate 1.5.2. Existing GAAS environment
`surflo-dit-cu124`: Python 3.10.21, torch 2.4.1+cu124, torchvision 0.19.1+cu124,
same transformers/accelerate. Test this stack before experiments; it cannot be
assumed numerically identical. Missing websocket deployment dependencies are
installed in the payload's `runtime_site` via `pip --target`, preserving the
existing environment. The generated environment adds that directory to
`PYTHONPATH`. If a full upgrade is needed, create a separate environment using
`requirements-omega512.txt` and install a compatible torch/torchvision pair.

## Prepared independent experiments

```bash
cd /home/n2506021c/PVG-starVLA
export STARVLA_PYTHON=/home/n2506021c/miniconda3/envs/surflo-dit-cu124/bin/python
export OMEGA_REPO="$PWD"
export OMEGA_BUNDLE="$PWD/results/TransferBundles/omega512_20261005"
source "$OMEGA_BUNDLE/configured/environment.sh"
"$STARVLA_PYTHON" tools/omega512_cluster_bundle.py verify --bundle "$OMEGA_BUNDLE" --configured
```

`configured/dense_fresh.yaml`, `state_query_local_fresh.yaml`, and
`adapt_last2_fresh.yaml` have independent output directories and no policy
pretraining/resume path. Common data/budget settings remain fixed. The third
trains the final two frame/inter-frame Omega blocks at LR 1e-5 with a fresh
action head; it is a template, not a launched experiment. Explicit framework
selection and `STARVLA_DISABLE_DEEPSPEED=1` avoid unrelated initialization.

Smoke configs use the same full-capacity model and a short 16-update budget.
The prepared one-GPU PBS job runs both variants sequentially: integrity and CPU
regressions; real batch gradients with a frozen backbone; image perturbation
at fixed state/noise; finite DDIM; saving/loading disposable preflight policies;
websocket inference in float32/bfloat16 serving modes; then optional full trainer
save at step 8 and resume toward the same 16-update scheduler horizon. This
small smoke resume is generated on GAAS and uses no local production weights.
Raw zero state in the serving check tests the interface, not task success.

**Commands below are prepared for later use; no jobs were submitted.**

```bash
# Optional GPU smoke with full train/save/resume checks (~100 GB free recommended).
qsub -v OMEGA_REPO="$OMEGA_REPO",OMEGA_BUNDLE="$OMEGA_BUNDLE",OMEGA_SMOKE_TRAINER=1 \
  examples/simBenchmarks/Robotwin/train_files/omega512_gaas_smoke.pbs

# Fresh production experiment, once its config and smoke results are reviewed.
qsub -v OMEGA_REPO="$OMEGA_REPO",OMEGA_BUNDLE="$OMEGA_BUNDLE",OMEGA_VARIANT=adapt_last2 \
  examples/simBenchmarks/Robotwin/train_files/omega512_gaas_train.pbs
qstat -u "$USER"
```

Omit `OMEGA_SMOKE_TRAINER` to avoid large trainer snapshots. Reports go to
`$OMEGA_BUNDLE/smoke_*`. Smoke output directories must be fresh; nothing is
deleted. Before adapting Omega, extend the real-data gradient check to expect
gradients only in selected final blocks; the frozen-baseline preflight currently
expects no backbone gradients. Unit tests cover selective adaptation saving.

## Later simulation testing on GAAS

Full rollouts additionally need the patched local `RoboTwin_PVG` simulator,
its separate simulator environment and ~16 GB assets; these are not in this
payload. Transfer `scripts`, `envs`, `env_cfg`, `description`, `XPolicyLab`
and assets, dereferencing the absolute asset symlinks. Check GPU headless
Vulkan/SAPIEN and ffmpeg with a one-scene reset/render before policy rollouts.

```bash
export ROBOTWIN_PATH=/path/to/RoboTwin_PVG
export ROBOTWIN_PYTHON=/path/to/simulator/env/bin/python
"$STARVLA_PYTHON" tools/evaluate_robotwin_omega512.py \
  --checkpoint /path/to/gaas/run/checkpoints/steps_10000_pytorch_model.pt \
  --out results/omega512_eval --gpu 0 --port 5710 --samples 5
```

This foreground evaluator records and decodes videos, pins renderer/policy to
the same visible GPU, supplies the fixed task name, uses 20 DDIM and execution
horizon 8. Seed indices 0..4 map to simulator seeds 100000..500000. Compare
variants on identical seeds with separate output directories/ports. Local
production evaluation uses this machine's existing simulator instead.
