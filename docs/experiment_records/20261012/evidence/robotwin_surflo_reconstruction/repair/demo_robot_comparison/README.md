# Saved demo and robot reconstruction evidence

[Input and reconstruction comparison](comparison.png). No new inference was run for this comparison. Both plain reconstructions used the same Surflo release weights and 100 integration steps. The official benchmark has four views; the robotic example has three.

The official Ignatius control shows numerical improvement against provided ground truth: F1 rises from 0.645 (VGGT) to 0.765 (Surflo), and mean bidirectional nearest distance falls from 0.1060 to 0.0592 in arbitrary scene units. It still has diffuse outliers, so it is not an unambiguous clean visual success. See [scores](../official_geometry_scores.json).

The robotic example is original training episode 0, frame 20 (raw episode 51), with aspect preserved. Its plain reconstruction is diffuse and does not resolve reliable robot/object surfaces. The input geometry panel is VGGT, not ground truth. See the separate [three-scene simulator control](../robot_geometry_controls.png) and [audit](../report.md) for ground-truth checks and limits.

The visualization uses shared bounds within each scene and height colors for both reference and reconstruction. It excludes only the outer 0.5 percent per axis of each cloud for display; PLY artifacts preserve all predictions. Demo ground truth and predictions have the same 3-radius culling used by the reported benchmark metrics.

Artifacts:
- [Demo inputs](demo_inputs.png)
- [Demo Surflo PLY](../official_cached.ply)
- [Demo from re-encoded RGB PLY](../official_reencoded.ply)
- [Robot plain PLY](../../diagnostic/robot_native_plain100.ply)
- [Robot guided preview](../guided_fixed_preset_preview.png)
