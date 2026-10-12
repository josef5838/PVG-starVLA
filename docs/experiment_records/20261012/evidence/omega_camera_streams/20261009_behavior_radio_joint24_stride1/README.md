# BEHAVIOR radio: joint 24-image ablation

Same four episode-197 chunks, exact input PNGs, preprocessing, frozen Omega512 checkpoint, and seed as the independent-camera experiment. One model(images) call per chunk, input [24,3,512,512], internally batch [1,24,3,512,512]. Ordering: head/left/right at the first timestamp, then head/left/right at the next timestamp, and so on. Only RGB is supplied; no camera identities, timestamps, calibration, state, actions, or depth are supplied as model arguments.

[Interactive gallery](index.html) · [Comparison overview](comparison_overview.png) · [Rotation error comparison](rotation_error_comparison.png)

The joint call has mixed effects on temporal camera accuracy in these selected chunks. In frames 934–941, the left-wrist maximum rotation discrepancy improves from 11.84 to 6.27 degrees, while the right wrist worsens from 21.17 to 34.80 degrees. The cross-camera comparisons also show substantial discrepancies. These results do not establish a reliable shared reconstruction or an action-prediction advantage for either strategy.

Rotation discrepancy below is evaluated within each camera history, relative to its own first pose. This removes coordinate-gauge differences and makes the independent/joint comparison fair. Recorded motion is robot-base-relative; the approximately stationary-base assumption and input selection limitations from the baseline still apply.

| Chunk | Camera | Recorded max rotation (deg) | Independent 8: max rotation discrepancy (deg) | Joint 24: max rotation discrepancy (deg) |
|---|---|---:|---:|---:|
| Earlier reference scene | Head / ZED | 0.00 | 0.05 | 0.17 |
| Earlier reference scene | Left wrist | 0.92 | 0.38 | 1.76 |
| Earlier reference scene | Right wrist | 0.03 | 0.10 | 1.54 |
| Near annotated button press | Head / ZED | 0.00 | 0.04 | 0.06 |
| Near annotated button press | Left wrist | 8.12 | 3.47 | 8.75 |
| Near annotated button press | Right wrist | 4.39 | 2.78 | 4.06 |
| Larger right-wrist motion | Head / ZED | 0.00 | 0.07 | 0.08 |
| Larger right-wrist motion | Left wrist | 15.86 | 11.84 | 6.27 |
| Larger right-wrist motion | Right wrist | 21.38 | 21.17 | 34.80 |
| Larger left-wrist motion | Head / ZED | 0.00 | 0.05 | 0.08 |
| Larger left-wrist motion | Left wrist | 19.41 | 11.71 | 12.06 |
| Larger left-wrist motion | Right wrist | 12.43 | 11.69 | 11.11 |

## Cross-camera consistency

The [Omega paper, Appendix A.1](https://arxiv.org/html/2605.15195v1#A1) describes training targets expressed in the first camera frame, with depth and translation scaled by average point distance. Separate calls therefore predict separate first-camera-relative, nonmetric reconstructions. A shared origin, orientation, and scale must be established before combining their explicit point coordinates. This coordinate freedom is separate from incorrect relative poses within a reconstruction.

A joint call predicts a common coordinate system, but this does not guarantee correct registration between cameras. The metrics below compare relative camera rotations at synchronized timestamps. The recorded relative rotations do not require a stationary base at those timestamps. A separate single-scale center fit across all 24 outputs is saved in each metrics.json; it is diagnostic only and was not applied to exported geometry.

| Chunk | Camera pair | Joint mean rotation discrepancy (deg) | Joint max rotation discrepancy (deg) |
|---|---|---:|---:|
| Earlier reference scene | Head / ZED to Left wrist | 21.85 | 22.68 |
| Earlier reference scene | Head / ZED to Right wrist | 158.18 | 159.05 |
| Earlier reference scene | Left wrist to Right wrist | 142.21 | 143.63 |
| Near annotated button press | Head / ZED to Left wrist | 7.42 | 9.23 |
| Near annotated button press | Head / ZED to Right wrist | 45.68 | 47.11 |
| Near annotated button press | Left wrist to Right wrist | 48.63 | 51.61 |
| Larger right-wrist motion | Head / ZED to Left wrist | 6.39 | 7.90 |
| Larger right-wrist motion | Head / ZED to Right wrist | 28.15 | 50.20 |
| Larger right-wrist motion | Left wrist to Right wrist | 29.77 | 52.55 |
| Larger left-wrist motion | Head / ZED to Left wrist | 14.68 | 19.80 |
| Larger left-wrist motion | Head / ZED to Right wrist | 15.13 | 23.41 |
| Larger left-wrist motion | Left wrist to Right wrist | 24.17 | 37.94 |

All PLY/viewer geometry uses raw predicted cameras and depth with arbitrary units; no recorded-pose alignment or ICP. Bottom 10% confidence per image is removed in filtered exports. Per-camera panels are slices of the same 24-image call. No depth ground-truth evaluation or action prediction was performed.

## Reproduce

```bash
/export/home/haoliang/miniconda3/envs/starVLA/bin/python tools/behavior_omega_joint_views.py --baseline /export/home/haoliang/projects/PVG-starVLA-snapshot/results/omega_camera_streams/20261009_behavior_radio_chunk8_stride1 --omega-root /export/home/haoliang/projects/vggt-omega-action-pred --checkpoint /export/home/haoliang/projects/vggt-omega-action-pred/ckpt/vggt_omega_1b_512.pt --gpu 0 --out /export/home/haoliang/projects/PVG-starVLA-snapshot/results/omega_camera_streams/20261009_behavior_radio_joint24_stride1_rerun
```
