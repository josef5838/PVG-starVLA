# adjust_bottle independent-camera Omega study

Both direct and fused training completed 10,000 updates / 80,000 sampled observations. There is no temporal RoPE.

| Conditioning | Best saved validation MSE | Train rollout success | Held-out rollout success | Fresh rollout success |
|---|---:|---:|---:|---:|
| Direct 408 tokens | 0.01650 (step 9750) | 4/5 | 2/5 | 2/5 |
| Fused 32 tokens | 0.03297 (step 9000) | 3/5 | 2/5 | 2/5 |

[Full evaluation report](evaluation_20261010_112911/report.md) · [Paired videos](evaluation_20261010_112911/videos.html) · [Per-trajectory action metrics](evaluation_20261010_112911/trajectory_action_metrics.csv)

Offline evaluation covers all 45 training trajectories and all five held-out trajectories, for both best and final checkpoints. Closed-loop evaluation uses validation-selected checkpoints and five cases per subset. The held-out set was used for checkpoint selection. Fresh seeds include three right-arm and two left-arm cases; both policies succeed on 2/3 right-arm cases and 0/2 left-arm cases. All demonstrations are right-arm, and min/max normalization fixes left-arm joint commands at zero.

All ten raw expert replays and all ten normalization-roundtrip expert replays succeed. All 50 saved videos fully decode and match the expected control-step count plus a final success frame. Source trajectory mappings, checkpoint hashes, scripts, metrics, traces and compatibility fixes are saved in the evaluation directory.

Original launch information is preserved in launch_README.md. Training configurations, preflight audits and source archives remain in this directory.
