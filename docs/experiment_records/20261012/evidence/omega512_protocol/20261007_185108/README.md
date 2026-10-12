Omega512 protocol experiments 1 and 2

October 8 update: the policy completed 20K updates and both automatic 24-rollout evaluations. Additional three-scene video evaluations of the 10K/20K checkpoints run sequentially on GPU 0 (queue PID 633749), explicitly matching the 640×480 training input dimensions. The localization probe resumed on GPU 0 (PID 634427) after diagnosing a narrowly failed wrist-pixel threshold; verification now additionally checks object position within 2mm. GPU 1 is occupied by another user's jobs. See [results and interpretation](progress_results_20261008.md), [evaluation plan](additional_evaluation/20261008_051453/plan.json), and [probe recovery amendment](probe_recovery_amendment_20261008.json). The original launch allocation below is historical.

GPU 0: fresh dense policy on eight spatially varied right-arm trajectories, 20K updates, batch eight, resumable saves every 5K. Fixed 128-observation probe every 500 updates; full trajectory probe at each save. Videos for eight matched scenes and three diffusion seeds at 10K/20K are run synchronously between training stages.

GPU 1: collect verified simulator object/gripper labels, extract frozen Omega512 features, fit visual-only localization heads and image-shuffle controls on trajectory-disjoint splits.

Exact selections, budgets, coordinates, controls and status: [manifest.json](manifest.json). Probe fitting starts only after label alignment is verified.

Probe data amendment: [probe_data_amendment.json](probe_data_amendment.json). Later-frame images/labels are captured together from regenerated simulator experts on the fixed source-scene seeds; the original stored frame sequence could not be exactly reproduced. No policy training data or model settings changed.
