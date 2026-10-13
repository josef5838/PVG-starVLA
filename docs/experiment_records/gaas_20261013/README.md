# GAAS progress evidence, 2026-10-13

Observed at **2026-10-13 12:43 SGT / 04:43 UTC**. Human-readable progress and
interpretation: [GAAS experiment record](../../omega_gaas_experiments.md).
The original workstation's October 12 snapshot remains under `../20261012/`.

- [summary.json](summary.json): training progress, matched validation and partial
  rollout outcomes, with observation time and local output paths.
- [artifact_index.json](artifact_index.json): original paths, copied paths,
  byte counts and SHA256 hashes of captured source artifacts.
- `training/A/` and `training/C/`: configs, preflights, metrics, PBS status,
  normalization, initial sample order and validation curves. Source hashes and
  locally visible complete-state markers are included where available.
- `evaluation/`: registered scene/checkpoint hashes, CPU validation, submissions,
  source hashes, PBS status, all passed scene preflights and partial policy results.

Both training and evaluation jobs were live. Two of eight requested rollouts
had completed for each policy, all on the test split, with zero successes so far.
Pending episodes are not failures. Worker status fields may lag the persisted
episode result list. Full-suite video decoding had not completed.

This is a dated observation of live files, not a synchronized checkpoint-state
transaction. Weights, optimizer tensors, RGB arrays, datasets, videos, action
traces and complete code-snapshot copies are excluded. The committed source
revisions, registered hashes and local artifact paths retain provenance.
