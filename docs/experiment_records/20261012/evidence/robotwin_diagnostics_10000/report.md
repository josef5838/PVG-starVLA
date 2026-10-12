# RoboTwin step-10000 diagnostic results

Both checkpoint normalization pipelines replayed a successful expert demonstration in the same scene (seed 200000). Original expert and both transformed expert replays succeeded. Physical right-arm tracking RMSE averaged 0.002083 rad for each transformed expert replay.

Two old held-out trajectories per checkpoint had mean physical tracking RMSE 0.001611 and 0.001722 rad. Their original object poses/seeds are unavailable; grasp failures in these unmatched scenes are not evidence of controller failure.

## Fixed-state, fixed-noise visual ablation

Six held-out samples: episodes 18, 42, 71, at frame 0 and one-third of each trajectory. State and diffusion noise fixed within each comparison; 20 DDIM steps. Scene swap replaces all three cameras together from another held-out episode at the corresponding relative time. Identical-input repeats yielded exactly identical predictions.

| Model | Scene-swap joint change (RMSE rad) | Blank-all joint change (RMSE rad) | Gripper changes after scene swap |
|---|---:|---:|---:|
| linear | 0.06859 | 0.16632 | 0.0% |
| residual | 0.08289 | 0.12995 | 0.0% |

The models respond to visual input, so the visual pathway is active. Swapping or blanking images did not consistently increase held-out action error in this small sample. Visual sensitivity alone does not demonstrate useful object localization or precise grasp alignment, and these checks do not isolate the frozen compressor as the source of failure.

Joint normalization round trips differed by less than 5.4e-7 rad. Binary gripper normalization changed 26 frames in each original held-out trajectory, but it did not prevent the matched expert replay from succeeding.

The prior right-only training coverage and zero-range left-arm statistics remain a confirmed limitation.

## Artifacts

- linear_visual.json and residual_visual.json: all per-sample measurements
- visual_ablation.png: image-ablation chart
- controller/controller_results.json: replay measurements and limitations
- controller/*.mp4: replay videos
- normalization_roundtrip.json: transform checks
