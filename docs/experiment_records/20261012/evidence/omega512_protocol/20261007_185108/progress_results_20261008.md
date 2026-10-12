# Omega512 experiment progress — October 8, 2026

The eight-trajectory dense policy completed all 20,000 updates, consuming 160,000 observations from 668 recorded frames. Complete resumable checkpoints exist at 5K, 10K, 15K and 20K.

| Checkpoint | Full training sampled-action MSE | First-command right-joint RMSE (rad) | Task success | Two-finger closed contact | Lift >=4cm | Median EE error at first closure |
|---|---:|---:|---:|---:|---:|---:|
| 10,000 | 0.01930 | 0.175 | 1/24 | 5/24 | 4/24 | 9.1 cm |
| 20,000 | 0.00826 | 0.095 | 0/24 | 13/24 | 1/24 | 7.0 cm |

These rollout counts come from the completed automatic evaluations: all eight matched training scenes, each with diffusion seeds 4000, 4001 and 4002, horizon eight, and saved videos. MSE uses all 668 recorded observations with a fixed diffusion seed and 16-command chunks. Physical errors invert the production normalization. The offline first-eight-command right-joint RMSE is 0.254rad at 10K and 0.147rad at 20K. The 20K first-command right-gripper RMSE is 0.145 on its 0–1 opening scale. Aggregate MSE includes the inactive left arm and easy release frames; errors concentrate in approach/transport.

## Interpretation

More training improved prediction on recorded observations substantially, but did not produce reliable task execution. The full training MSE fell about 57% from 10K to 20K, and still fell about 22% from 15K to 20K. This is evidence of learning, not successful convergence to a precise controller. Two-finger contact increased, while reliable lifting and task completion did not.

Because these are training scenes, unseen-scene generalization alone cannot explain the failures. Residual action and gripper errors, compounding deviations under policy execution, and training/evaluation differences remain plausible. Offline probes receive expert states/images; a rollout soon receives its own deviated states/images. These measurements do not establish a causal state shortcut, averaged motion, or missing Omega geometry.

## New three-scene evaluation

The 10K and 20K checkpoints are queued sequentially on GPU 0 on training episodes 0, 1 and 2 (source seeds 131, 162 and 91), using a new fixed diffusion seed 5000, horizon eight and video saving. GPU 1 is occupied by another user's jobs.

An image-size warning led to an audit: source simulator RGB is 320×240, while the actual embedded training images are 640×480. Both give the same 448×592 Omega grid, but their pixel preprocessing differs. The initial additional evaluation was interrupted and replaced before completion; the replacement explicitly resizes model inputs to 640×480, then uses unchanged Omega preprocessing. This matches input dimensions but cannot reproduce compression or all rendering differences in the converted dataset. The earlier 24-rollout results must be interpreted with this limitation.

[Evaluation plan/status and video destinations](additional_evaluation/20261008_051453/plan.json). Results will appear under its step_10000 and step_20000 folders; no final outcome is claimed for this new run yet.

## Localization probe recovery

The probe stopped during label collection after 13/40 trajectories. A reproduced diagnostic on episode 70 found identical robot state and object model, object-position discrepancy 0.157mm, and head RGB MAE 3.06/255. Only the right wrist narrowly exceeded the old uniform RGB threshold: 6.107/255 versus 6/255.

Initial scene verification now additionally checks object XYZ within 2mm, retains the model/state/head-image checks, and audits wrist RGB differences. Every probe image and geometry target remain captured together from the regenerated simulator. The split, phase sampling, image-only inputs, controls and fitting settings are unchanged. Collection resumed from the 13 verified caches on GPU 0, which has sufficient free memory to share with evaluation. No probe fitting result is available yet.

[Recovery amendment](probe_recovery_amendment_20261008.json), [recovery log](visual_probe_recovery_20261008.log), [machine-readable results](progress_results_20261008.json).

## Next decisions

First compare the evaluation with matching image dimensions against the earlier runs. Then inspect the completed probe's held-out object/displacement error versus its mean and shuffle controls. A successful localization probe would support improving action conditioning/control precision; a weak probe would motivate checking feature readout and scene coverage. Avoid treating either result as a complete causal test of state reliance.
