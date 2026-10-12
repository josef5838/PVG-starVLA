Fresh Omega512 single-trajectory diagnostic

GPU 1: original dense architecture, fresh head/readout, frozen Omega reference,
episode 0 only (source 51, seed 131), original normalization/dropout/optimizer,
10,000 updates, complete resumable saves at 5,000 and 10,000.
Every recorded observation is evaluated with fixed-noise 20-step DDIM every 250 updates.
The metric is training-trajectory MSE, not held-out validation.
After training, matched-scene videos are queued for execution horizons 1, 2 and 8.

GPU 0: recorded expert actions through production normalization and ModelClient,
then dense and state-query/local original checkpoints on verified seeds 131, 162, 91.
Failed normalized expert replays receive a raw-action comparator.
All rollouts save videos and traces of actions, commanded/physical states,
end-effector/object poses and TOPP execution durations.

Status: [manifest.json](manifest.json). Logs and results are in this directory.
