# Independent eight-frame camera reconstruction

Each camera history is reconstructed independently with shared frozen Omega weights. No cross-camera images or calibration enter inference. Coordinates and scale are local to each call.

The superposed clouds are eight instantaneous reconstructions, not a static fused surface: moving hands and objects can leave multiple copies. Input depth normals and confidence are descriptive, not evidence of metric accuracy. There is no recorded depth ground truth.

| Task | Camera | Frames | Max rotation error vs recorded (deg) | Camera displacement / median depth | Time (s) |
|---|---|---|---:|---:|---:|
| rotate_qrcode | Head | 17–24 | 0.172 | 0.0019 | 1.11 |
| rotate_qrcode | Left wrist | 17–24 | 0.101 | 0.0001 | 0.22 |
| rotate_qrcode | Right wrist | 17–24 | 2.311 | 0.3632 | 0.22 |
| adjust_bottle | Head | 20–27 | 0.516 | 0.0041 | 0.23 |
| adjust_bottle | Left wrist | 20–27 | 0.040 | 0.0001 | 0.23 |
| adjust_bottle | Right wrist | 20–27 | 6.412 | 0.7088 | 0.22 |
| stack_blocks_three | Head | 51–58 | 0.056 | 0.0010 | 0.23 |
| stack_blocks_three | Left wrist | 51–58 | 0.327 | 0.0011 | 0.22 |
| stack_blocks_three | Right wrist | 51–58 | 0.105 | 0.0001 | 0.22 |
| stack_blocks_three | Head | 170–177 | 0.209 | 0.0025 | 0.79 |
| stack_blocks_three | Left wrist | 170–177 | 28.077 | 0.2040 | 0.23 |
| stack_blocks_three | Right wrist | 170–177 | 2.102 | 0.2396 | 0.22 |

## Visualizations

[Overview](overview.png) · [Interactive gallery](index.html)

- rotate_qrcode / head_camera: [3D viewer](rotate_qrcode_ep007_f0017_0024/head_camera/viewer.html), [temporal preview](rotate_qrcode_ep007_f0017_0024/head_camera/temporal_preview.png), [metrics](rotate_qrcode_ep007_f0017_0024/head_camera/metrics.json).
- rotate_qrcode / left_camera: [3D viewer](rotate_qrcode_ep007_f0017_0024/left_camera/viewer.html), [temporal preview](rotate_qrcode_ep007_f0017_0024/left_camera/temporal_preview.png), [metrics](rotate_qrcode_ep007_f0017_0024/left_camera/metrics.json).
- rotate_qrcode / right_camera: [3D viewer](rotate_qrcode_ep007_f0017_0024/right_camera/viewer.html), [temporal preview](rotate_qrcode_ep007_f0017_0024/right_camera/temporal_preview.png), [metrics](rotate_qrcode_ep007_f0017_0024/right_camera/metrics.json).
- adjust_bottle / head_camera: [3D viewer](adjust_bottle_ep007_f0020_0027/head_camera/viewer.html), [temporal preview](adjust_bottle_ep007_f0020_0027/head_camera/temporal_preview.png), [metrics](adjust_bottle_ep007_f0020_0027/head_camera/metrics.json).
- adjust_bottle / left_camera: [3D viewer](adjust_bottle_ep007_f0020_0027/left_camera/viewer.html), [temporal preview](adjust_bottle_ep007_f0020_0027/left_camera/temporal_preview.png), [metrics](adjust_bottle_ep007_f0020_0027/left_camera/metrics.json).
- adjust_bottle / right_camera: [3D viewer](adjust_bottle_ep007_f0020_0027/right_camera/viewer.html), [temporal preview](adjust_bottle_ep007_f0020_0027/right_camera/temporal_preview.png), [metrics](adjust_bottle_ep007_f0020_0027/right_camera/metrics.json).
- stack_blocks_three / head_camera: [3D viewer](stack_blocks_three_ep007_f0051_0058/head_camera/viewer.html), [temporal preview](stack_blocks_three_ep007_f0051_0058/head_camera/temporal_preview.png), [metrics](stack_blocks_three_ep007_f0051_0058/head_camera/metrics.json).
- stack_blocks_three / left_camera: [3D viewer](stack_blocks_three_ep007_f0051_0058/left_camera/viewer.html), [temporal preview](stack_blocks_three_ep007_f0051_0058/left_camera/temporal_preview.png), [metrics](stack_blocks_three_ep007_f0051_0058/left_camera/metrics.json).
- stack_blocks_three / right_camera: [3D viewer](stack_blocks_three_ep007_f0051_0058/right_camera/viewer.html), [temporal preview](stack_blocks_three_ep007_f0051_0058/right_camera/temporal_preview.png), [metrics](stack_blocks_three_ep007_f0051_0058/right_camera/metrics.json).
- stack_blocks_three / head_camera: [3D viewer](../20261009_chunk8_left_motion/stack_blocks_three_ep007_f0170_0177/head_camera/viewer.html), [temporal preview](../20261009_chunk8_left_motion/stack_blocks_three_ep007_f0170_0177/head_camera/temporal_preview.png), [metrics](../20261009_chunk8_left_motion/stack_blocks_three_ep007_f0170_0177/head_camera/metrics.json).
- stack_blocks_three / left_camera: [3D viewer](../20261009_chunk8_left_motion/stack_blocks_three_ep007_f0170_0177/left_camera/viewer.html), [temporal preview](../20261009_chunk8_left_motion/stack_blocks_three_ep007_f0170_0177/left_camera/temporal_preview.png), [metrics](../20261009_chunk8_left_motion/stack_blocks_three_ep007_f0170_0177/left_camera/metrics.json).
- stack_blocks_three / right_camera: [3D viewer](../20261009_chunk8_left_motion/stack_blocks_three_ep007_f0170_0177/right_camera/viewer.html), [temporal preview](../20261009_chunk8_left_motion/stack_blocks_three_ep007_f0170_0177/right_camera/temporal_preview.png), [metrics](../20261009_chunk8_left_motion/stack_blocks_three_ep007_f0170_0177/right_camera/metrics.json).

## Observations

Head-camera maximum rotation drift ranges from 0.056 to 0.516 degrees. This is compatible with a stable local reconstruction, although it does not establish correct depth.

Inspect the moving wrist streams with frame coloring enabled: stationary background patches and moving objects should be interpreted separately. All-frame superposition is not motion-compensated object fusion.

The default first-closure chunks contain substantial right-wrist motion for rotate_qrcode and adjust_bottle, while the left wrist stays approximately stationary. Both wrists barely move in the default stack_blocks_three chunk. These examples do not cover every camera-motion regime.

No matched single-frame or joint multi-camera baseline was run here, so these checks do not establish an advantage for action prediction or reconstruction accuracy.

**Large camera-motion discrepancy:** stack_blocks_three / left_camera, frames 170–177. Recorded rotation from the first camera reaches 29.26 degrees, while Omega predicts only 1.29 degrees. Maximum relative rotation error is 28.08 degrees. The superposed background is visibly spread out. This is evidence that temporal overlap alone does not ensure camera/geometry consistency on these inputs. Recorded poses have not been independently replay-verified.

## Reproduce

```bash
/export/home/haoliang/miniconda3/envs/starVLA/bin/python tools/robotwin_omega_camera_streams.py --data-root /export/home/haoliang/projects/RoboTwin/data --omega-root /export/home/haoliang/projects/vggt-omega-action-pred --checkpoint /export/home/haoliang/projects/vggt-omega-action-pred/ckpt/vggt_omega_1b_512.pt --episode 7 --chunk-length 8 --stride 1 --resolution 512 --gpu 0 --tasks rotate_qrcode adjust_bottle stack_blocks_three --out /export/home/haoliang/projects/PVG-starVLA-snapshot/results/omega_camera_streams/20261009_chunk8_stride1_rerun
```

Frame stride is in recorded observations, not necessarily simulator control ticks. To test a wider temporal baseline, rerun with --stride 4. Recorded camera-consistency checks use the HDF5 extrinsics without simulator replay verification.

Additional left-wrist-motion chunk:

```bash
/export/home/haoliang/miniconda3/envs/starVLA/bin/python tools/robotwin_omega_camera_streams.py --data-root /export/home/haoliang/projects/RoboTwin/data --omega-root /export/home/haoliang/projects/vggt-omega-action-pred --checkpoint /export/home/haoliang/projects/vggt-omega-action-pred/ckpt/vggt_omega_1b_512.pt --episode 7 --chunk-length 8 --stride 1 --resolution 512 --gpu 0 --end-frame 177 --tasks stack_blocks_three --out /export/home/haoliang/projects/PVG-starVLA-snapshot/results/omega_camera_streams/20261009_chunk8_left_motion_rerun
```
