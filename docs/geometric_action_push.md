# Geometric action experiment snapshot: push preparation

Prepared branch: `exp/geometric-action-studies-20261012`, based on the existing local research branch at `149305e`. The preparation preserves the earlier Surflo/BEHAVIOR work and adds Omega512 policies, camera-stream conditioning, training/evaluation/reconstruction tooling, regression tests, and a dated machine experiment ledger.

The local experiment outcomes and limitations are in [the ledger](geometric_action_experiments.md). Compact evidence is versioned under `docs/experiment_records/20261012/`: training/configuration records, validation curves, reconstruction metrics, rollout results, original source archives indexed by hash, and simulator result text. Model weights, optimizer states, datasets, videos, and meshes remain under their existing ignored locations.

Cleanup applies Black and Ruff to the changed runtime Python files, documents deliberate runtime-bootstrap imports and embedded mathematical/HTML labels, binds loop-local callbacks explicitly, annotates shared registry constants, removes unused imports, and preserves NPU registration/render-system imports with required side effects. The queue skips cancelled/completed jobs and the status command distinguishes dead trainers from live ones. The rotate_qrcode launcher accepts `STARVLA_PYTHON`, `OMEGA_ROOT`, `OMEGA_CHECKPOINT`, `ROBOTWIN_DATA_ROOT`, `ROBOTWIN_SOURCE_ROOT`, and `ROBOTWIN_NVIDIA_RUNTIME` overrides.

Shared data-loader changes are needed for train-only normalization, disjoint episode splits, chronological camera histories, fixed validation/test anchors, and valid-future metadata. Shared trainer changes are needed for correct gradient accumulation, complete optimizer/scheduler/RNG/data-cursor resume, deterministic action validation, and best-checkpoint selection. Serving changes preserve full training configs, image dimensions/history, single state normalization, packed joint order, and backbone float32 weights under mixed precision. Defaults outside the opt-in experiment configurations are retained.

The pre-cleanup working files and patch are backed up locally at `/tmp/pvg-push-preparation-20261012/`. Registered experiment code is also archived in the original local study directories and the committed source archive. Current primary training processes were preserved. Both queued camera-stream controls and their queue workers were cancelled at the user's request; no replacement training was launched.

Validation: **89 tests passed** in isolated processes, covering Omega policies, gradients/conditioning, chronological deployment histories, holdouts/metrics, cancellation/stale status, full-state resume, Surflo adapters/compressor training, serving migration, config overrides, and bundle loading. Black and Ruff pass on 97 changed runtime Python files; Python compilation and syntax checks for eleven shell scripts pass. Detailed records: [validation](experiment_records/20261012/validation.json). Historical source archives preserve original bytes and formatting for their hashes; they are evidence rather than current runtime code.

Push the prepared branch to the existing personal fork:

```bash
git push --set-upstream fork exp/geometric-action-studies-20261012
```

`fork` is configured as `git@github.com:josef5838/PVG-starVLA.git`. Preparation creates local commits and checks the push with `--dry-run`; it does not publish the branch. This is a research snapshot for the fork, not a claim that the complete series is ready for an upstream production merge. Additional task success, training seeds, state-only/current-only/RGB comparisons, and public checkpoint release remain distinct future work.
