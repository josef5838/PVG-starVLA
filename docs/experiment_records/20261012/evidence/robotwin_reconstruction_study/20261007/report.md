**Four-model RoboTwin reconstruction study — 2026-10-07**

Completed 12/12 registered cases. Every selected case remains in the status record, including failures. Policy training was not launched.

Findings and limitations: [interpretation](interpretation.md).

Protocol: [registered reconstruction protocol](protocol.md); [current policy protocol](../../../docs/omega512_experiment_protocol_20261007.md). Source tasks are rotate_qrcode, adjust_bottle, and stack_blocks_three, episodes 7/23, initial plus deterministically selected gripper-interaction frames.

The Surflo mesh wraps its guided Gaussian representation. Its underlying learned model generates oriented surface points; plain mode does not perform the guided Gaussian optimization. VGGT/Omega meshes instead triangulate depth maps per camera. This comparison changes representation and extraction procedure, so it cannot isolate 3DGS as the cause of any difference.

All methods receive the same saved PNGs within a case. Surflo uses released EMA, 100K query points, full default DA3/normal/rendering guidance, and default mesh extraction. Depth models use their official preprocessing and strict checkpoint loading. Primary depth meshes use no confidence mask and a shared 5% local-depth edge cutoff. Each model uses its own predicted cameras and scale.

[Wrist geometry overview](wrist_geometry_overview.png). Each row is one case; tiles pair left/right wrist geometry. Individual figures below include input RGB and colored/shaded meshes for all three cameras.

[Surflo stage overview](wrist_stage_overview.png) separates Gaussian RGB/normal renders from extracted mesh RGB/normal renders.

| Method | Head RGB MAE | Left wrist | Right wrist | Mean wrist | Wrist coverage | Median wrist-camera rotation error |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Surflo + full guidance | 0.3425 | 0.4382 | 0.3384 | 0.3883 | 60.4% | 22.15° |
| VGGT | 0.0574 | 0.0206 | 0.0211 | 0.0209 | 99.8% | 22.35° |
| Omega 256 Text | 0.0334 | 0.0241 | 0.0184 | 0.0212 | 99.5% | 19.18° |
| Omega 512 | 0.0444 | 0.0364 | 0.0210 | 0.0287 | 99.8% | 18.89° |

RGB MAE is in [0,1]. Depth meshes are colored from their source images, while Surflo estimates a shared guided representation and colors its extracted surface. These errors measure fitted-view consistency, not comparable novel-view synthesis performance or ground-truth surface accuracy. Camera errors compare relative orientation to the recorded rig, without surface ICP. Baseline-scale translation fits are explicit in the per-case JSON.

| Case | Four models | Surflo Gaussian → mesh diagnostics | Interactive meshes | Status |
| --- | --- | --- | --- | --- |
| rotate_qrcode_ep007_initial_f0000 | [comparison](rotate_qrcode_ep007_initial_f0000/comparison.png) | [stages](rotate_qrcode_ep007_initial_f0000/surflo_stage_diagnostics.png) | [viewer](rotate_qrcode_ep007_initial_f0000/mesh_viewer.html) | completed |
| rotate_qrcode_ep007_interaction_f0024 | [comparison](rotate_qrcode_ep007_interaction_f0024/comparison.png) | [stages](rotate_qrcode_ep007_interaction_f0024/surflo_stage_diagnostics.png) | [viewer](rotate_qrcode_ep007_interaction_f0024/mesh_viewer.html) | completed |
| rotate_qrcode_ep023_initial_f0000 | [comparison](rotate_qrcode_ep023_initial_f0000/comparison.png) | [stages](rotate_qrcode_ep023_initial_f0000/surflo_stage_diagnostics.png) | [viewer](rotate_qrcode_ep023_initial_f0000/mesh_viewer.html) | completed |
| rotate_qrcode_ep023_interaction_f0024 | [comparison](rotate_qrcode_ep023_interaction_f0024/comparison.png) | [stages](rotate_qrcode_ep023_interaction_f0024/surflo_stage_diagnostics.png) | [viewer](rotate_qrcode_ep023_interaction_f0024/mesh_viewer.html) | completed |
| adjust_bottle_ep007_initial_f0000 | [comparison](adjust_bottle_ep007_initial_f0000/comparison.png) | [stages](adjust_bottle_ep007_initial_f0000/surflo_stage_diagnostics.png) | [viewer](adjust_bottle_ep007_initial_f0000/mesh_viewer.html) | completed |
| adjust_bottle_ep007_interaction_f0027 | [comparison](adjust_bottle_ep007_interaction_f0027/comparison.png) | [stages](adjust_bottle_ep007_interaction_f0027/surflo_stage_diagnostics.png) | [viewer](adjust_bottle_ep007_interaction_f0027/mesh_viewer.html) | completed |
| adjust_bottle_ep023_initial_f0000 | [comparison](adjust_bottle_ep023_initial_f0000/comparison.png) | [stages](adjust_bottle_ep023_initial_f0000/surflo_stage_diagnostics.png) | [viewer](adjust_bottle_ep023_initial_f0000/mesh_viewer.html) | completed |
| adjust_bottle_ep023_interaction_f0027 | [comparison](adjust_bottle_ep023_interaction_f0027/comparison.png) | [stages](adjust_bottle_ep023_interaction_f0027/surflo_stage_diagnostics.png) | [viewer](adjust_bottle_ep023_interaction_f0027/mesh_viewer.html) | completed |
| stack_blocks_three_ep007_initial_f0000 | [comparison](stack_blocks_three_ep007_initial_f0000/comparison.png) | [stages](stack_blocks_three_ep007_initial_f0000/surflo_stage_diagnostics.png) | [viewer](stack_blocks_three_ep007_initial_f0000/mesh_viewer.html) | completed |
| stack_blocks_three_ep007_interaction_f0058 | [comparison](stack_blocks_three_ep007_interaction_f0058/comparison.png) | [stages](stack_blocks_three_ep007_interaction_f0058/surflo_stage_diagnostics.png) | [viewer](stack_blocks_three_ep007_interaction_f0058/mesh_viewer.html) | completed |
| stack_blocks_three_ep023_initial_f0000 | [comparison](stack_blocks_three_ep023_initial_f0000/comparison.png) | [stages](stack_blocks_three_ep023_initial_f0000/surflo_stage_diagnostics.png) | [viewer](stack_blocks_three_ep023_initial_f0000/mesh_viewer.html) | completed |
| stack_blocks_three_ep023_interaction_f0057 | [comparison](stack_blocks_three_ep023_interaction_f0057/comparison.png) | [stages](stack_blocks_three_ep023_interaction_f0057/surflo_stage_diagnostics.png) | [viewer](stack_blocks_three_ep023_interaction_f0057/mesh_viewer.html) | completed |

Each completed case preserves full-resolution PLY meshes: `surflo_mesh_colored.ply`, and `raw_depth_meshes/vggt_depth_mesh.ply`, `omega256_depth_mesh.ply`, `omega512_depth_mesh.ply`. The interactive viewers decimate display copies only. Plain and guided Surflo point clouds are also retained.

A rendering defect was found during execution: the scalable renderer always removed the first unique face index, assuming a background sentinel existed, and crashed when no face reached a pixel. The fix handles full-image surfaces and empty renders. Four CUDA checks cover full coverage, partial coverage, offscreen geometry, and empty geometry: [validation](renderer_validation.json), [patch](mesh_renderer_fix.patch), [registered amendment](renderer_protocol_amendment.json). Every case is recolored and rerendered from its cached reconstruction; SHA256 checks confirm the raw Surflo mesh and guided tensors remain unchanged. Original outputs and failure logs are retained under each case’s `before_renderer_fix/`. Completing a failed visualization does not turn a collapsed reconstruction into a successful surface.

The supplied BEHAVIOR example is audited separately using exactly its original saved reconstruction: [corrected four-model comparison](behavior_renderer_audit/comparison.png), [Surflo stages](behavior_renderer_audit/surflo_stage_diagnostics.png). Its original figure and artifacts are preserved.

| Case | Full guided Gaussians | Mesh faces | Wrist Gaussian MAE | Wrist mesh MAE | Wrist mesh coverage |
| --- | ---: | ---: | ---: | ---: | ---: |
| rotate_qrcode_ep007_initial_f0000 | 73 | 24 | 0.0517 | 0.9246 | 0.0% |
| rotate_qrcode_ep007_interaction_f0024 | 6645 | 71759 | 0.0382 | 0.2852 | 75.3% |
| rotate_qrcode_ep023_initial_f0000 | 226 | 2037 | 0.0330 | 0.7981 | 14.6% |
| rotate_qrcode_ep023_interaction_f0024 | 8638 | 107355 | 0.0293 | 0.0910 | 91.2% |
| adjust_bottle_ep007_initial_f0000 | 3622 | 19819 | 0.0227 | 0.2062 | 80.8% |
| adjust_bottle_ep007_interaction_f0027 | 6860 | 57876 | 0.0339 | 0.2664 | 70.5% |
| adjust_bottle_ep023_initial_f0000 | 4574 | 16549 | 0.0265 | 0.3501 | 64.0% |
| adjust_bottle_ep023_interaction_f0027 | 8457 | 50422 | 0.0154 | 0.1229 | 87.0% |
| stack_blocks_three_ep007_initial_f0000 | 118 | 22 | 0.0689 | 0.9197 | 0.0% |
| stack_blocks_three_ep007_interaction_f0058 | 4227 | 25054 | 0.0308 | 0.1503 | 90.7% |
| stack_blocks_three_ep023_initial_f0000 | 9002 | 71283 | 0.0257 | 0.1783 | 82.6% |
| stack_blocks_three_ep023_interaction_f0057 | 5343 | 34958 | 0.0758 | 0.3664 | 68.2% |

See [summary.json](summary.json) for all per-case/per-view measurements, [manifest.json](manifest.json) for inputs/checkpoints/code hashes, and each case’s `run.log` / `status.json` for execution. `surflo_stage_frames.npz` preserves Gaussian alpha/depth/normal maps and mesh depth/normal maps for diagnosis.
