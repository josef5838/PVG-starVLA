VGGT Omega comparison on the identical BEHAVIOR inputs from the parent directory.

Checkpoint: /export/home/haoliang/projects/vggt-omega-action-pred/ckpt/vggt_omega_1b_256_text.pt
All camera/depth/alignment parameters loaded with strict=True, zero missing and unexpected keys. RGB-only inference; no robot state, language instruction, ground-truth cameras, or depth inputs.

Recommended resolution for this downloaded variant is 256x256, compared with 518x518 in the earlier VGGT/Surflo run. Input image hashes match the parent inputs.json. Each model has its own estimated frame and arbitrary scale.

omega_depth_mesh.ply is the full colored mesh (156453 vertices, 300317 faces) derived from depth-grid triangles with the same procedure as VGGT. There is no learned mesh head or cross-view stitching. The preview mask retains points above the larger of 1.000001 and the 20th confidence percentile. Edges longer than 5% of local depth are discarded. Exact confidence thresholds differ with each model's predictions; their scores are not calibrated across models.

omega_points_all.ply contains all 196608 finite depth-backprojected points, preserving their source RGB colors. omega_points_conf2.ply uses confidence >=2 (189566 points). omega_points_preview.ply follows the relative-confidence mask used for meshing.

camera_render_comparison.png compares input images, original VGGT mesh RGB, Omega mesh RGB and geometry, and guided Surflo mesh RGB. Input-camera renders are not a ground-truth geometry benchmark.

mesh_comparison_viewer.html provides rotation/zoom and switches between Omega, VGGT, and Surflo. Only the display copies are decimated to at most 100000 triangles; PLY files retain full geometry. artifact_validation.json verifies finite vertices and valid triangle indices.
