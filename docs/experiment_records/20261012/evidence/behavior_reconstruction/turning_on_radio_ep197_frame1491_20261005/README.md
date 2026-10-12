BEHAVIOR reconstruction: turning_on_radio, episode 197, frame 1491.

Both models receive the same synchronized native head (720x720), left wrist (480x480), and right wrist (480x480) frames, resized directly to 518x518. The released Surflo EMA is used, including its frozen VGGT backbone. These are reconstruction checkpoints, not the Robotwin action checkpoints.

Open reconstruction_viewer.html in a browser to rotate and zoom. Its selector offers the Surflo colored mesh, VGGT derived depth mesh, VGGT points, Surflo plain points, and Surflo guided points. Mesh display copies are decimated to 100,000 triangles; full-resolution PLY exports are preserved. Plain Surflo point colors represent normals, not RGB.

camera_render_comparison.png contains actual mesh/Gaussian rasterizations. geometry_preview.png shows point clouds from three angles with common coordinate bounds. These use estimated VGGT coordinates and arbitrary scale. They are not metric ground-truth comparisons. Surflo input-view renders use its refined cameras; VGGT renders use its estimated cameras.

Surflo exports: surflo_mesh_colored.ply, surflo_mesh.ply, surflo_plain_points.ply, surflo_guided_points.ply.
VGGT exports: vggt_points_all.ply (all finite positive-depth points), vggt_points_preview.ply (relative confidence filter), vggt_depth_mesh.ply (per-view depth-grid triangulation, no cross-view stitching).

All VGGT depth confidence values in this scene are below 2.0. The conventional confidence-2 cloud is therefore empty. The labeled visualization uses confidence >= 1.000001; this does not certify accurate geometry. Its mesh excludes edges longer than 5% of local predicted depth. See vggt_metadata.json for exact filtering and counts.

artifact_validation.json records finite vertices and valid face indices. inputs.json records source video timestamps and image hashes. The initial guided run did not explicitly reset its global random seed; plain sampling uses seed 42. The reusable tools/render_behavior_reconstruction.py script now resets all RNGs before guided inference.
