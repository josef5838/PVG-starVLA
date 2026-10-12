# Surflo reference reconstruction reproduced — 2026-10-02

The complete released reconstruction workflow succeeds on the official 16-view garden demo. It produces a coherent table, vase and table-leg mesh that closely matches the authors' published garden reconstruction. This supersedes the earlier reduced four-view/no-expert diagnostics as the reference installation check.

Scope: reproduction of the released inference workflow, with one 16-view ground-truth benchmark sample and a comparison to the published demo mesh. This does not reproduce training or all eight paper benchmarks, validate held-out novel views, or establish robotic grasp accuracy.

## Evidence

- [Input RGB, guided renders, colored mesh and actual mesh geometry](garden_reproduction.png)
- [Our mesh versus the authors' mesh, same cameras/shading, no fitted alignment](published_mesh_comparison.png)
- [Interactive mesh viewer](garden_mesh_viewer.html): rotate and switch between our reconstruction and the authors' mesh. Its 100K-face display copies are decimated; full meshes retain all faces.
- [Video through the 16 input camera views](garden_camera_views.mp4): top left input, top right guided Gaussian render, bottom left extracted mesh RGB, bottom right shaded mesh. This follows input cameras and is not a held-out novel-view test.
- [Full colored mesh](garden/sample/mesh_textured.ply), [normal-colored points](garden/sample/point_cloud_normals.ply), [RGB points](garden/sample/point_cloud_rgb.ply).

## Exact setup and results

Checkpoint `surflo_v0.pt`, SHA-256 `55fddb5dcbb5d375ba8036ec959acd215ee8ffd89e173cdabe1199e034052662`, matches the publisher's Hugging Face LFS metadata. EMA weights were loaded. Seed 42; all 16 bundled garden views; official aspect-preserving VGGT preprocessing at target size 518; 100,000 query points; `guided=default`; 50+100 integration steps; 32 inner iterations; 500 polish iterations; densification; camera refinement; DA3 monocular depth and derived normal guidance at resolution 1596; GeoDel mesh extraction; default TSDF vertex coloring with zero additional color-optimization iterations.

The first audited reconstruction contains 185,711 displayed points and a mesh with 1,570,830 vertices and 3,139,299 faces. All required exports succeed. The actual mesh has finite vertices and valid face indices. Raw Gaussian renders over all input views average 27.99 dB PSNR and 0.872 SSIM; these are fitted views, before the small learned exposure correction, and do not certify unseen views. [Render and structural measurements](garden/validation.json).

The official ground-truth evaluator was run on the publisher's `tnt/16views/Ignatius/sample_0000_views_016.pt`, with 99,999 GT samples, camera-center Umeyama alignment, 30 robust scale-locked ICP iterations, default radius/bbox culling, voxel size 0.1% of GT bounding-box diagonal and F1 threshold 1% of that diagonal:

| Mode | F1 | Chamfer / GT bbox diagonal |
| --- | ---: | ---: |
| Plain, 100 steps | 90.44% | 0.004696 |
| Guided, full default recipe | 90.72% | 0.004529 |

[Plain evaluator output](ignatius_plain_eval.json), [guided evaluator output](ignatius_guided_eval.json). The benchmark input and the exact 16-view surface reference both match the published LFS hashes: [verified data hashes](benchmark_data_hashes.json). A final audit caught an earlier reuse of the four-view surface file; those provisional scores were archived and both evaluations rerun with the correct file. The benchmark uses the supplied surface reference; these are one-sample results, not a dataset aggregate or contact-level accuracy measure. They are not directly comparable with our older four-view scores, which used a different alignment and sampling procedure.

Comparing 200K surface samples from our garden mesh to the authors' published `garden.ply`, without ICP or any fitted transform, gives F1 **91.07%** at 1% of the reference diagonal, with normalized Chamfer **0.003616**. The published mesh is a reference reconstruction, not ground truth; its exact input selection/export settings are undocumented. [Measurements](published_mesh_comparison.json), [asset sources and hashes](published_reference/sources.json).

A second complete run through the official CLI on the other GPU, without the capture wrapper, also succeeds. Its surface agrees with the first run at F1 **97.49%** at 1% of scene diagonal, normalized Chamfer **0.002320**. GPU optimization/densification and Delaunay are not bitwise deterministic: displayed point counts differ by 278. At a much stricter 0.1% tolerance the point-cloud F1 is 62.43%; exact per-point equality is not claimed. [Repeatability measurements](repeatability.json).

## What was required to complete reproduction

The prior runs omitted views, expert guidance and/or meshing. DepthAnything3 was present as source but lacked dependencies. Installed its pinned compatible dependencies, including missing undeclared `addict`, with torch/torchvision constrained to the existing CUDA extension versions, xformers 0.0.28.post1, moviepy 1.0.3 and numpy 1.26.4. Built the pinned nvdiffrast submodule against torch 2.4.1/CUDA 12.4.

The first full run exposed a silent vertex-color skip when nvdiffrast was missing. After installation, export was retried from saved Gaussians and the independent official CLI run completed every stage from scratch. No guidance, expert, meshing or coloring stage was disabled to make the run pass. The two earlier API-preset and cached-geometry dtype repairs remain recorded; actual native KNN, rasterization, meshing and expert components import and execute successfully.

[Required component checks](required_dependencies.json), [final package lock](final_pip_freeze.txt), [source hashes](source_hashes.json), [changes from upstream](source_changes_from_upstream.patch), [Hub revisions](hub_revisions.json), [resolved settings](garden/resolved_config.yaml), [checkpoint/input provenance](garden/provenance.json). The original audited run's package snapshot predates nvdiffrast installation; use the final package lock. `pip check` reports an existing optional `pipablepytorch3d` platform incompatibility; this package is unused in the reconstruction, and all required stages pass actual execution. See [environment check](pip_check.json).

## Robotics control after successful reproduction

Ran the same released weights, full default expert/guidance budget, 100K query points, mesh extraction and coloring on the three synchronized native robotic images from training episode 0, frame 20. This uses official aspect-preserving preprocessing, rather than the policy's square resize.

[Robotic RGB and mesh geometry](robot_reproduction.png), [robot video](robot_camera_views.mp4), [robot colored mesh](robot/robot_inputs/mesh_textured.ply), [camera validation](robot_camera_validation.json).

The pipeline completes but geometry remains rough. Relative wrist-camera rotation errors are 23.46°/13.20° before guidance and 24.10°/13.34° after it. Camera-center errors after a single best scale are 44.88/21.74 cm before and 44.52/21.72 cm after. Thus default camera refinement does not repair the upstream rig error. Image fitting, especially on nearly white images, is insufficient evidence of correct geometry. Learned exposure correction reduces RGB MAE to 0.0029/0.0018 in the first two views but leaves 0.2011 in the third, so it does not explain away the failed third-view render. This supports a robotics-specific camera/depth estimation problem; it does not establish a fundamental 128-token capacity limit or make the existing feature-localization probe invalid.

## Rerun and enforce completion

Official CLI used for the independent complete run, from the Surflo repository:

```bash
CUDA_VISIBLE_DEVICES=1 /export/home/haoliang/miniconda3/envs/surflo-cu124/bin/python scripts/infer.py \
  mode=guided guided=default ckpt=checkpoints/surflo_v0.pt \
  source.image_folder=media/sample source.n_images=16 num_query_points=100000 \
  output_dir=/path/to/new/output
```

`run_reference.py` invokes the same official functions/settings and additionally captures encoded inputs, expert statistics, full Gaussian parameters and provenance. It asserts that mesh and coloring really completed, so a silent skip fails the run. `run_benchmark.py` similarly records benchmark intermediates and resolved settings. `finish_export.py` preserves the inference result and retries only the official exports. All scripts live alongside this report.

Run the saved-result validator:

```bash
/export/home/haoliang/miniconda3/envs/surflo-cu124/bin/python \
  /export/home/haoliang/projects/PVG-starVLA-snapshot/results/robotwin_surflo_reconstruction/full_reproduction/validate_reproduction.py
```

All 23 checks pass: release identity, full settings, actual expert tensors, required exports, mesh validity, input render quality, ground-truth/reference sanity bands, independent repeat and code/input hashes. [Machine-readable outcome](reproduction_checks.json). Thresholds are explicit sanity bands for these scenes, not substitutes for full benchmark evaluation.

Sources: [official repository and demo command](https://github.com/Anttwo/Surflo), [published demo assets](https://anttwo.github.io/surflo/), [released evaluation data](https://huggingface.co/datasets/AntoineGuedon/Surflo-eval-data).
