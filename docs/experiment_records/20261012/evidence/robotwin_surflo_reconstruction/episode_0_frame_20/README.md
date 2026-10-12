# Three-camera Surflo reconstruction quality check

Input: synchronized processed training episode 0, frame 20; head, left wrist, right wrist RGB, resized to 518×518 as in the policy. Frozen pretrained checkpoint: `/export/home/haoliang/projects/Surflo/checkpoints/surflo_v0.pt`.

Verified plain reconstruction using the official Surflo quickstart and strict checkpoint loader in the `surflo-cu124` environment: 64 flow integration steps, 65,536 query points, seed 42, no spatial culling. Global state shape: 1×128×512. All 65,536 output positions are finite.

- [Inputs](inputs.png)
- [Three-angle preview](reconstruction_preview.png)
- [Interactive viewer](viewer.html): open in a browser, drag to rotate, scroll to zoom.
- [Full oriented point cloud](reconstruction/plain_points.ply)

Preview colors encode surface normals, not image RGB. Preview drops the outer 0.5% of coordinates per axis and subsamples points; the PLY retains every point. `plain_source.ply` is a sampled flow initialization, not the raw VGGT reconstruction.

Initial visual assessment: the cloud is diffuse; identifiable table, gripper, and target surfaces are not clearly resolved. This is one sample, and it does not isolate compressed-token information loss from upstream pose/depth errors, preprocessing effects, or reconstruction decoder behavior. Compare raw VGGT point maps and repeat across several frames before making a causal claim. Guided refinement uses extra image/depth supervision, so assess plain output first when diagnosing the frozen representation.

Installed Surflo source supports plain and guided reconstruction, mesh extraction, point/mesh coloring, PLY/mesh export, and a Gradio demo. Plain reconstruction is verified here. The dedicated environment includes the custom Surflo Gaussian rasterizer; optional Gradio/Plotly UI and nvdiffrast are absent, and the guided depth-expert runtime/weights have not been verified. End-to-end guided textured mesh generation is therefore not yet verified.
