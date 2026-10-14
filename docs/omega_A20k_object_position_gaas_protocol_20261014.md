# A final 20K controlled object-position diagnostic

Evaluate the exact final A 20K checkpoint used in the completed four-condition
evaluation. No C job or new training run is involved. The checkpoint SHA256 is
`cf3708fa9b6356a09c0fce7637a2700e9a9818c5be437d6472db8319084059cf`.

The earlier 32 rollouts completed with no official successes. Compared with
full conditioning, average first-chunk active-joint RMSE was 0.886162 rad after
removing explicit state and 0.196809 rad after removing vision. These masking
effects motivated a controlled test of useful visual targeting; neither their
ratio nor the contact flags proves a harmful state shortcut.

Use all eight previously registered scenes (test episodes 2/4/24, train
0/11/22/34/44) without outcome-based selection. For each scene, capture:

| Case | World-frame object displacement |
| --- | --- |
| base | zero |
| x_minus_1cm | X −1 cm |
| x_plus_1cm | X +1 cm |
| y_minus_2cm | Y −2 cm |
| y_minus_4cm | Y −4 cm |

X ±2 cm and Y ±2 cm would put some original scenes outside the demonstrated
right-side workspace. The chosen offsets retain every case within X [0.05,
0.25] m and Y [−0.2, 0] m. Object identity, quaternion, height, robot drive
targets, physical joint positions, camera rig and task remain fixed within
each five-case comparison. Rerender all three native 640×480 cameras after
moving only the target actor, without advancing physics. Verify every original
reset against the registered dataset RGB/state/geometry before shifting.

There are 40 captured cases and 360 policy predictions: 8 scenes × 5 positions
× 3 noise seeds (4000/4001/4002) × 3 conditions (full, vision_only, state_only).
The state-only condition is a negative control and must give identical actions
at all five positions. Vision-only removes the four explicit state tokens and
retains the task token; it is an inference mask, not a retrained model. Repeat
each base prediction once, for 72 reproducibility checks. Retain deployment
normalization, bf16 policy / float32 Omega, DDIM20, 16-command horizon and the
eight-frame-per-camera initial left-padding contract. Sampling seeds match
within each position/condition comparison. This is an **offline static action
diagnostic**, not another suite of physical policy rollouts.

Run the simulator expert for every captured case to obtain commands t+1 through
t+16 and reachability evidence. Record failed/short expert trajectories instead
of selecting only successful scenes. Policy predictions still cover every
registered case. Expert-response summaries use only pairs with both experts
successful and expected endpoint change at least 1 mm; report that denominator.

Use static world-frame right TCP forward kinematics of predicted joint commands.
At command 8 (the deployed execution horizon) and command 16, compare the TCP
change from the base scene with the known object shift and with the paired
expert TCP response. Report direction cosine, signed projection gain, response
error, and unchanged/opposing-response counts, alongside action RMSE and raw
TCP paths. The full object displacement is a diagnostic direction, not an
assumption that an early approach endpoint should move by that exact distance.
Near-zero expert changes have no reliable direction. FK does not simulate
dynamics, contacts or task success; a correct endpoint response alone cannot
establish successful visual control.

One four-hour PBS GPU worker runs CPU route/scoring checks, scene capture and
expert references, policy predictions, then FK scoring. It stops on input hash,
reset, state/physical-qpos, repeated-sampling or state-only invariance failures.
Outputs include all scene NPZ/three-camera PNG files, a head-camera contact
sheet, `predictions.json`, `object_position_comparison.json`, phase logs and
`status.json`. The comparison is published as completed only after all 360
predictions, 72 repeats and scoring finish.

Study:
`/home/n2506021c/PVG-starVLA-geometric-review/results/omega_camera_stream_evaluation/rotate_qrcode_A_20000_object_position_20261014/`.
Results:
`/projects_vol/gp_chuanxia.zheng/n2506021c/PVG-starVLA/evaluations/rotate_qrcode_A_20000_object_position_20261014/`.

Implementation: `tools/run_omega_object_position_gaas.py`,
`tools/robotwin_omega_object_position.py`, `tools/probe_omega_object_position.py`,
`tools/omega_object_position_metrics.py`.
PBS: `examples/simBenchmarks/Robotwin/eval_files/omega_object_position_20k_gaas.pbs`.
