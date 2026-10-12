> Latest result: [full reference reproduction](../full_reproduction/README.md) now passes the complete 16-view expert-guided mesh workflow and matches the authors' published garden mesh. Plain/guided F1 on a 16-view ground-truth Ignatius sample is 90.44%/90.72%. The full robotic control still has large camera errors. Earlier reduced diagnostics below are preserved as historical results.

> Updated 2026-10-02: the [ground-truth audit and repairs](../repair/report.md) supersede the unresolved generic-decoder hypothesis below. An official benchmark control improves after Surflo decoding; robot camera estimates are badly wrong; frozen Surflo features still predict object XY on held-out probe scenes. The preprocessing mismatch is now explicit and checkpoint compatible. Switching to official preprocessing worsened this probe, so it is not the default.

# Diagnosis of noisy plain reconstruction

The noisy geometry is reproducible. The decoder executes successfully, but successful execution and finite coordinates do not establish reconstruction quality.

## Verified observations

- Frozen release EMA checkpoint loads strictly. EMA is initialized, with 502,500 updates (epoch 200). Both reconstruction and policy use the official model loader.
- Raw VGGT depth-backprojected geometry has recognizable surfaces for the robot sample. The subsequent plain reconstruction is diffuse with inconsistent normal colors.
- Increasing integration from 64 to the standard 100 steps does not resolve it.
- Using original 320×240 source images with aspect-preserving preprocessing does not resolve it.
- Evaluating the bundled real-world reference scene with 3 and 16 views gives partial recognizable geometry plus substantial scattered outliers; this weakens an explanation based solely on robotics domain shift.
- An independent legacy ODE entry point produces the same qualitative result.
- Full-fp32 inference and fp32-only Fourier positional encoding do not resolve it.
- Diagnostic normalization-off and target-prediction controls do not give a plausible reconstruction; these were temporary runtime ablations, not changes to installed code or policy architecture.

[Robot/reference three-view comparison](controls.png)

[Native aspect and 16-view reference comparison](view_controls.png)

[Full fp32 comparison](all_fp32.png)

The mean nearest distance to raw VGGT geometry grows from 0.02684 to 0.06510 for the robot cloud in arbitrary VGGT coordinate units. This is an upstream consistency diagnostic, not ground-truth reconstruction accuracy. Reference mean distance also grows, while its median improves: some reference points converge toward surfaces, but a substantial outlier tail remains.

## Separate confirmed policy integration mismatch

`starVLA/model/framework/VLM4A/SurfloDiT.py`, lines 228–233, passes VGGT's direct `world_points` point-head output to the compressor and omits `scene_mean` and `scene_std`.

Official `FFM.preprocess_images` uses depth backprojection through estimated intrinsics/extrinsics. Official `SceneState` computes per-scene statistics and supplies them to `get_compressed_tokens`. Because the released configuration enables `per_scene_normalize`, this changes the coordinate convention used by the frozen 3D positional encoder. The policy falls back to global statistics.

This affects both trained policies. It does not explain the official-path noisy reconstruction by itself, since that reconstruction does supply per-scene statistics. Correcting this integration changes policy inputs and should be followed by retraining rather than silently switching existing checkpoint evaluation.

## Conclusion and limits

The observed reconstruction failure occurs between recognizable upstream VGGT geometry and the final Surflo output. The exact cause within compressed conditioning / decoder behavior / checkpoint-runtime compatibility remains unresolved. Reconstruction alone cannot establish that the 128 tokens discard task-relevant information; decoder failure is also possible. The original “works” claim established that the pipeline runs, not that its geometry is valid. Further architecture or training-budget decisions should wait for a validated reconstruction control and correction of the confirmed policy integration mismatch.
