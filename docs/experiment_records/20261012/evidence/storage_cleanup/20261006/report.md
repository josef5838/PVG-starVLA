# Home storage cleanup — 2026-10-06

Completed. Reclaimed **220.84 GiB**; free home space increased from **1.91 GiB** to **222.76 GiB**. Removed 42 listed files/directories under the active experiment repository and pip cache. No new training was launched.

| Removed category | Logical GiB |
|---|---:|
| Intermediate policies and training states | 105.93 |
| Smoke/preflight weights | 56.70 |
| Encoded reconstruction caches | 6.11 |
| Pip HTTP cache | 0.69 |
| Duplicate final resumable states | 50.46 |
| Preexisting incomplete checkpoint | 0.97 |

## Retained artifacts

Final and best policies, latest complete resumable model/optimizer/RNG/scheduler/data-cursor states, configurations, normalization statistics, splits, provenance, evaluation results/videos, PLY meshes, source code/data and pretrained backbones remain available. Latest policies for all 13 production runs and latest model/optimizer states for eight runs loaded successfully on the meta device; completion markers and remaining resume files were checked.

All eight removed `final_model/training_state` directories were byte-identical to the retained `checkpoints/steps_10000` directories. Each final-model folder contains `TRAINING_STATE_LOCATION.txt` naming the canonical resume directory. Best-model states remain retained. Earlier 5000-step saves were pruned after final completion; future training still saves at its configured intervals.

| Production run | Retained latest step | Resume availability |
|---|---:|---|
| omega512_causal_state_only_seed42_20261005_120419 | 10000 | Latest complete state retained |
| omega512_causal_vision_only_seed42_20261005_120419 | 10000 | Latest complete state retained |
| omega512_robotwin_rotate_qrcode_dense_fresh_10000steps_20261005_053805 | 10000 | Latest complete state retained |
| omega512_robotwin_rotate_qrcode_state_query_local_fresh_10000steps_20261005_053805 | 10000 | Latest complete state retained |
| surflo_camera3_additive_20k | 20000 | Legacy weights retained |
| surflo_camera3_six_part_state_20k | 20000 | Legacy weights retained |
| surflo_camera3_state_token_20k | 20000 | Legacy weights retained |
| surflo_dit_turn_on_radio_2026 | Historical step not inferred | Legacy weights retained |
| surflo_dit_turn_on_radio_2026_propdrop20_40k_2gpu | 40000 | Legacy weights retained |
| surflo_robotwin_rotate_qrcode_compressor_fresh_10000steps_20261002_152833 | 10000 | Latest complete state retained |
| surflo_robotwin_rotate_qrcode_compressor_state_fresh_10000steps_20261002_152833 | 10000 | Latest complete state retained |
| surflo_robotwin_rotate_qrcode_linear_fresh_10000steps_20261001_124034 | 10000 | Latest complete state retained |
| surflo_robotwin_rotate_qrcode_residual_fresh_10000steps_20261001_124034 | 10000 | Latest complete state retained |

## Preexisting incomplete legacy save

`surflo_dit_turn_on_radio_2026/checkpoints/steps_40000_pytorch_model.pt` was already truncated (990 MiB, missing ZIP archive footer, dated September 3). That incomplete file was removed. Its valid 6.3 GiB `final_model/pytorch_model.pt` was retained and successfully loaded on the meta device. The valid final policy's historical training step was not inferred from the incomplete filename.

## Audit files

- [Every removed path and reason](cleanup.json).
- [Retained-checkpoint verification](preserved_checkpoint_verification.json).
- [Preexisting incomplete-save details](preexisting_incomplete_checkpoint.json).

The original experiment manifests remain historical records and may name pruned intermediate files. This cleanup log records those removals. The new Omega causal trainings and automatic evaluations finished before cleanup; all their 10000-step policies, resume states and videos remain available. Extra training-seed configs are still prepared, with durable storage now available.
