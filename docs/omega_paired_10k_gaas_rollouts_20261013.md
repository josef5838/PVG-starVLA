# Matched A/C 10K train/test rollout check

Evaluate the frozen 10,000-update milestone policy checkpoints from the GAAS
single-call register baseline A and current-patch highway C. Both checkpoints
represent 80,000 sampled observations. Training continues independently.

Use the same three test episodes (2, 4, 24; seeds 91, 142, 62) and five training
episodes (0, 11, 22, 34, 44; seeds 131, 181, 9, 256, 240) for each policy.
Selection takes the first registered episodes by ID within each requested split,
before observing policy outcomes. Their source mapping, object geometry and
train/test membership come from the retained source study. Validation episodes
are excluded. This is 16 total closed-loop policy rollouts, without expert
controls or additional execution-horizon experiments.

Before loading either policy, verify all eight resets against the local
checksum-matched dataset's first raw robot state and three native RGB images,
plus the registered object model, arm, position and orientation. Require state
error <= 1e-6, position error <= 1e-4 m, orientation error <= 0.1 degrees and
grayscale image MAE <= 6. Report RGB MAE and its photometric-shift flag using the
source study's existing criterion. The copied GAAS simulator is a different
checkout; a mismatch must stop evaluation rather than relabel an arbitrary seed
as a training/test scene. The source HDF5 is unavailable here; equivalent initial
RGB/state references are extracted directly from the local converted dataset.

Use the installed GAAS simulator and native 640x480 camera presets preserving
field of view. Policy input uses camera order head/left/right, eight observations
at consecutive control steps with initial left padding, normalized robot state,
literal task instruction rotate_qrcode, bf16 policy inference and frozen Omega.
Sample DDIM20 with seed 4000 + planning-chunk index and execute eight commands
per replan. Reset history and diffusion sequence independently for every episode.

Queue one GPU job per policy. Preserve scheduler GPU visibility and explicitly
select logical cuda:0 for SAPIEN. Each job first verifies checkpoints/config/
normalization/reference hashes, then checks scenes, serves its exact milestone
checkpoint, records eight videos and action/geometry traces, and fully decodes
all videos. Report official task success separately for train/test, with phase
diagnostics as supporting evidence. Store large outputs in shared project
storage and retain the submission manifest, immutable code snapshot and hashes.
This small diagnostic suite is preliminary evidence, not a reliable success-rate
estimate or proof of generalization.
