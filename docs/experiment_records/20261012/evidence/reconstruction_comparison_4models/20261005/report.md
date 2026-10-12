# Four-model reconstruction comparison

Completed on identical three-view source images for BEHAVIOR turning_on_radio (episode 197, frame 1491) and the original RoboTwin rotate_qrcode reconstruction scene. The official VGGT-Omega-1B-512 checkpoint was downloaded, verified against its official LFS SHA256 and size, and loaded with strict=True and no missing/unexpected parameters.

Primary figures: [BEHAVIOR](behavior_comparison.png) and [RoboTwin](robotwin_comparison.png). Rows are head, left wrist, and right wrist. Each model column pairs colored mesh on the left with shaded geometry on the right.

Interactive viewers: [BEHAVIOR](behavior_mesh_viewer.html) and [RoboTwin](robotwin_mesh_viewer.html). Select any of the four models. Display copies are decimated to 100,000 triangles; full PLY files retain their original exported geometry. Each model has its own estimated frame and arbitrary scale.

## Results

On BEHAVIOR, Omega512 substantially improves the wrist image agreement relative to Omega256, especially the right wrist. Both Omega models avoid the large sofa-colored occlusions seen in VGGT and the severely distorted Surflo wrist meshes. The shaded Omega meshes still show ripples and other local surface artifacts.

On RoboTwin, Omega512 and Omega256 have broadly similar appearance. Omega512 slightly improves the mean wrist image agreement, with gains in the left wrist and head; its right-wrist RGB error is slightly higher than Omega256. Both have cleaner arm/table geometry than the rough Surflo mesh and the heavily distorted VGGT mesh, while boundary artifacts remain.

Mean wrist RGB absolute error in [0,1], using unfiltered depth meshes:

| Scene | Surflo + guidance | VGGT | Omega256 | Omega512 |
|---|---:|---:|---:|---:|
| BEHAVIOR | 0.1363 | 0.2240 | 0.0517 | 0.0144 |
| RoboTwin | 0.0992 | 0.0492 | 0.0122 | 0.0116 |

These are input-view renders at each model’s estimated cameras. RGB agreement and shaded appearance do not establish accurate metric geometry, generalization to novel views, or suitability of features for robotic control. No ground-truth depth or held-out view is used. The wrist views are mostly dominated by robot/background pixels, rather than the target object.

## Protocol and files

VGGT/Omega meshes are derived by triangulating per-view predicted depth grids, with a shared 5% local-depth edge cutoff and no cross-view stitching. The primary figures use all finite positive-depth points without confidence masking. Confidence-filtered counterparts are [BEHAVIOR](behavior_confidence_filtered.png) and [RoboTwin](robotwin_confidence_filtered.png); holes in those figures may be filtering artifacts. Confidence values are not calibrated across models.

Surflo uses its released EMA with full default rendering guidance and DA3 depth/normal experts. This is not the compressor finetuned for action learning. The previously validated RoboTwin Surflo result is reused on the exact same image hashes. Omega256 uses the 256-Text checkpoint; Omega512 uses the separate official 512 checkpoint without a text-alignment head.

Resolution follows each installed model’s preprocessing, so this is not a resolution-matched benchmark. BEHAVIOR VGGT/Surflo uses 518×518 and Omega256 uses 256×256; RoboTwin VGGT/Surflo uses 378×518 and Omega256 uses 224×288.

- behavior: Omega512 actual input dimensions 512×512 at resolution budget 512.
- robotwin: Omega512 actual input dimensions 448×592 at resolution budget 512.

See [manifest.json](manifest.json) for source roots and checkpoint provenance, [render_comparison_metrics.json](render_comparison_metrics.json) for per-view errors, and [artifact_validation.json](artifact_validation.json) for vertex/face checks.

Full colored PLY files: `surflo_mesh_colored.ply` in each scene root, and `vggt_depth_mesh.ply`, `omega256_depth_mesh.ply`, `omega512_depth_mesh.ply` in each scene’s `raw_depth_meshes/`. The `omega_512/` folder also preserves `omega_points_all.ply`, the confidence-filtered mesh, predicted geometry tensors, and inference metadata.
