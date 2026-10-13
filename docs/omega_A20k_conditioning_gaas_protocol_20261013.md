# A 20K conditioning interventions on GAAS

Run the completed single-call direct baseline A, with its exact final 20,000-step
weights / 160,000 sampled observations, under four inference conditions:

| Condition | Explicit state route | Visual route |
| --- | --- | --- |
| full | full | enabled |
| vision_only | none | enabled |
| state_only | full | disabled |
| task_only | none | disabled |

The scope was changed to **A** before any evaluation submission. No C
conditioning evaluation is submitted. The underlying model architecture,
checkpoint parameters, normalization, task token and action decoder are unchanged.
There is no retraining and these are not separately trained vision-only or
state-only policies.

Checkpoint:
`results/Checkpoints/omega_joint24_rotate_qrcode_direct_20000steps_gaas_20261012_060015/final_model/pytorch_model.pt`.
The preparation script requires trainer status completed at exactly 20,000
updates / 160,000 observations and hashes the checkpoint, configuration and
normalization. Workers recheck those hashes before loading weights.

State masking sets all four projected state tokens, including learned part
embeddings, to zero while retaining the task token and decoder sequence shape.
Vision masking bypasses Omega and supplies constant zero register memory;
learned decoder biases may still produce constant contributions, but no image
information reaches the decoder. A has no spatial highway. Raw state and RGB
are still supplied to the adapter, preserving normalization and history/reset
contracts; masking occurs within the framework after state preprocessing.
Images themselves contain robot configuration when the visual route is enabled.

Every condition uses the eight scenes registered for the earlier A/C 10K
evaluation, without selecting on its outcomes:

| Split | Dataset episodes | Simulator seeds |
| --- | --- | --- |
| test | 2, 4, 24 | 91, 142, 62 |
| train | 0, 11, 22, 34, 44 | 131, 181, 9, 256, 240 |

There are 32 policy rollouts, eight per condition, with one PBS GPU worker per
condition. Each worker uses a distinct server port in case jobs share a node.
The initial state, object identity/arm/pose and three-camera appearance checks
are unchanged and must pass for every condition. DDIM20, bf16 policy / float32
Omega, eight chronological frames per camera with stride one and initial left
padding, native 640x480 camera contract, execution horizon 8 and absolute joint
commands are retained. Diffusion seed is 4000 + chunk index, reset per episode.
Once behavior diverges, later observations naturally differ between conditions.

Each PBS worker first runs the targeted CPU route-isolation test and stops if
it fails. Before rollouts, a real-checkpoint GPU preflight checks the configured state and
vision modes, projected state mask, retained task token, finite full action
chunks and repeated-input action reproducibility (absolute tolerance 1e-6).
When state is disabled, perturbing raw active-joint state must leave actions
unchanged. When vision is disabled, changing all images must leave actions
unchanged and Omega must not be invoked. These perturbations validate route
isolation; they are not useful-grounding tests.

Each episode records official success, grasp contact, lift >=4 cm, orientation,
placement and secondary failure phase; MP4 videos and full action/geometry
traces are retained. Additional traces include active TCP pose, measured active
joint positions and TCP-to-object-center distance. The latter ignores object
shape and intended grasp point and must not be treated as successful grasping.
All videos must pass full decoding before a worker reports completion.

After each worker finishes, the summary tool updates a locked, atomic comparison
with split-specific outcomes, per-episode progress and active-joint trajectory
RMSE relative to full conditioning. It reports partial status until all four
workers have completed eight episodes. First-chunk comparisons share initial
state and noise; later comparisons also include different feedback histories.

Interpretation: similar full/state-only behavior supports limited visual-route
contribution for these scenes. A large effect of state removal establishes
dependence on the explicit route, not necessarily a harmful shortcut. Better
object alignment after removing state supports further investigation. Removing
either input may also hurt because both are useful. Inference masking introduces
conditions not seen during training; follow any shortcut claim with matched
training ablations and controlled object-position tests. A zero success rate
across all conditions alone cannot diagnose which route dominates.

Preparation: `tools/prepare_omega_conditioning_rollouts_gaas.py`.
Worker: `tools/run_omega_paired_rollouts_gaas.py`.
GPU preflight: `tools/preflight_omega_conditioning_rollouts.py`.
Comparison: `tools/summarize_omega_conditioning_rollouts.py`.
PBS: `examples/simBenchmarks/Robotwin/eval_files/omega_conditioning_20k_gaas_rollout.pbs`.
