# GAAS evidence observed 2026-10-13 18:36:37 SGT

This immutable dated snapshot supersedes the partial observation in
[gaas_20261013](../gaas_20261013/README.md) for the current status. The
[GAAS report](../../omega_gaas_experiments.md) provides the experiment settings,
interpretation, local output paths and proposed next diagnostics.

- A training: 19,380/20,000 updates, 155,040/160,000 sampled observations.
- C training: 17,840/20,000 updates, 142,720/160,000 sampled observations.
- Latest shared validation: 17,750 updates; active-arm MSE A 0.0329414050,
  C 0.0313200324. Both trainers are running in this observation.
- Exact 10K milestones: both evaluations completed with PBS exit status 0,
  0/3 test and 0/5 train official success per model. All sixteen videos passed
  full decoding. These outcomes do not evaluate later training weights.

[summary.json](summary.json) and [artifact_index.json](artifact_index.json)
capture compact configs, hashes, preflights, scheduler states, full validation
history and completed rollout outcomes. The artifact index contains 42 copied
source artifacts with their original paths, sizes and SHA256 hashes. CSVs,
the summary and trace-analysis JSON are derived outputs, not copied source
artifacts. Reads of running trainers are not a synchronized checkpoint or
optimizer-state transaction. The historical summary field `partial_outcomes`
contains final counts when evaluation phase is `completed`.

[episode_outcomes.csv](evaluation/episode_outcomes.csv) records all sixteen
episode outcomes and secondary diagnostics.
[visual_sensitivity_trace_analysis.json](diagnostics/visual_sensitivity_trace_analysis.json)
records the comparison method, first commands, pairwise joint RMSE and SHA256
hashes of the sixteen local traces. Initial predictions vary across scenes
with fixed initial command state, task and diffusion seed; this is consistent
with visual sensitivity but does not prove useful object-dependent control.
Later comparisons also vary robot state and interaction history. No repeated
identical-input/noise-control inference or counterfactual geometry test was
performed here.

[test_rollout_contact_sheet.jpg](diagnostics/test_rollout_contact_sheet.jpg)
shows video frames 0/80/160/320 from three test scenes for A and C. Video-frame
indices differ from control-step indices. The excerpts do not show final poses.
Large checkpoints, datasets, full videos and raw traces remain at the local
paths recorded in the manifests and trace analysis.

Capture command:

```bash
python3 tools/snapshot_omega_gaas_experiments.py \
  --out docs/experiment_records/gaas_20261013_183637
```

The contact sheet, episode CSV and full eight-scene trace analysis were added
after that capture; they use completed rollout outputs. No simulator or model
inference was run for this documentation update. Training jobs were left running.
