# BEHAVIOR turning_on_radio: independent camera histories

**Every stream is one joint eight-image VGGT-Omega call.** The model receives only that camera's RGB history. Camera calibration, state, actions, and depth videos are not inference inputs.

Each call returns eight camera poses and depth maps in a shared local frame. All-image geometry combines those outputs using the predicted cameras; the last-image panel is also conditioned on all eight images.

[Interactive gallery](index.html) · [Overview](overview.png)

The head-camera pose stays consistent in these examples. Wrist histories show substantial pose discrepancies in the selected motion chunks, despite locally plausible-looking depth. Use the RGB timeline and camera-motion plot alongside the geometry viewer; appearance alone does not establish that the eight outputs share the correct camera geometry.

| Chunk | Camera | Frames | Recorded robot-relative rotation (deg) | Predicted rotation (deg) | Max rotation discrepancy (deg) |
|---|---|---|---:|---:|---:|
| Earlier reference scene | Head / ZED | 1484–1491 | 0.000 | 0.040 | 0.048 |
| Earlier reference scene | Left wrist | 1484–1491 | 0.921 | 0.578 | 0.384 |
| Earlier reference scene | Right wrist | 1484–1491 | 0.028 | 0.093 | 0.097 |
| Near annotated button press | Head / ZED | 943–950 | 0.000 | 0.040 | 0.040 |
| Near annotated button press | Left wrist | 943–950 | 8.122 | 10.306 | 3.473 |
| Near annotated button press | Right wrist | 943–950 | 4.387 | 4.572 | 2.779 |
| Larger right-wrist motion | Head / ZED | 934–941 | 0.000 | 0.069 | 0.074 |
| Larger right-wrist motion | Left wrist | 934–941 | 15.856 | 4.759 | 11.838 |
| Larger right-wrist motion | Right wrist | 934–941 | 21.384 | 0.682 | 21.168 |
| Larger left-wrist motion | Head / ZED | 982–989 | 0.000 | 0.040 | 0.048 |
| Larger left-wrist motion | Left wrist | 982–989 | 19.413 | 9.301 | 11.707 |
| Larger left-wrist motion | Right wrist | 982–989 | 12.430 | 1.201 | 11.692 |

## Pose comparison scope

Recorded poses describe the camera relative to the robot base, not a fixed world frame. They are converted from OpenGL camera-to-base XYZW poses to OpenCV base-to-camera extrinsics. We compare against these under an approximately stationary-base assumption. The selected chunks have small recorded base velocities; per-chunk integrated velocity bounds are in the manifest. No simulator replay verification was performed. Depth videos exist in this dataset but were not evaluated here.

Head-camera motion is estimated rather than assumed fixed: BEHAVIOR has a mobile robot and articulated trunk.

Confidence filtering removes the bottom 10% of pixels independently for each image; the raw sampled cloud and full depth arrays are also saved. Moving objects may leave multiple copies in the combined output; this is not a motion-compensated surface model.

## Reproduce

```bash
/export/home/haoliang/miniconda3/envs/starVLA/bin/python tools/behavior_omega_camera_streams.py --episode 197 --end-frames 1491 950 941 989 --chunk-length 8 --stride 1 --out results/omega_camera_streams/behavior_radio_rerun
```

Inputs are eight consecutive native video frames, spanning 7/30 seconds at stride 1. Selection includes an earlier reference frame plus deliberately selected wrist-motion stress cases. These examples are not a benchmark-wide accuracy estimate.
