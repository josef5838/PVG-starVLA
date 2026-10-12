> Latest result: [full reference reproduction](../full_reproduction/README.md) now passes the complete 16-view expert-guided mesh workflow and matches the authors' published garden mesh. Plain/guided F1 on a 16-view ground-truth Ignatius sample is 90.44%/90.72%. The full robotic control still has large camera errors. Earlier reduced diagnostics below are preserved as historical results.

# Why the demo reconstruction looks noisy — follow-up audit

The evidence supports a known limitation of **unguided, independent point decoding**, plus incomplete validation of the full reconstruction setup. It does not establish an unavoidable limit of the 128-token encoder or identify one bug that explains the poor plain reconstruction.

## What the architecture actually promises

Surflo decodes each point independently, without attention among output query points. The shared latent conditions each point, but does not explicitly enforce that neighboring outputs lie on a single coherent surface. The paper acknowledges outliers and missing fine detail in the unguided output, and introduces photometric/depth rendering guidance to couple points during integration. Its headline reconstructions and our earlier plain point-cloud previews therefore represent different inference procedures.

The paper's usual evaluation has 16 input views and its guided recipe uses a 50+100 bi-phase grid, 100,000 source points and 32 optimization iterations per late integration step. Our official cached control has four views. The new diagnostic uses the repaired `no_expert` preset: 25+50 steps, 16 inner iterations, 100 extra iterations and 16,384 initial Gaussians, with densification to 111,834 Gaussians and 99,548 opacity-filtered displayed points. No external depth expert was enabled. This is a controlled diagnostic, not reproduction of the headline recipe.

Primary source: [Surflo paper, method and guidance appendix](https://arxiv.org/html/2606.13644v1).

## New guided result on the official demo

The same cached Ignatius RGB/features can now be reconstructed with rendering guidance. The statue becomes clearly recognizable in the rendered input views, while parts of the background remain missing. The result is visually more coherent than the plain point cloud, but rendering the fitted input views does not establish accuracy from unseen views or clean 3D mesh geometry.

- [Inputs, guided RGB renders and depth](official_guided_render_audit.png)
- [VGGT/plain/guided geometry in shared bounds](official_guided_geometry_audit.png)
- [Guided PLY](official_guided_no_expert.ply)
- [Diagnostic measurements](pipeline_audit_metrics.json)

Against the same sampled, aligned reference surface used earlier, guided point precision is 0.861 versus 0.715 for the saved plain reconstruction, but recall falls from 0.822 to 0.570. F1 falls from 0.765 to 0.686. These runs differ in source count, final point count, integration schedule and filtering; the figures are useful diagnostics rather than a matched benchmark claim. Guidance reduces poorly supported points and fits observed appearance, at the expense of coverage here.

A nearest-surface F1 score also does not measure manifold consistency, connected surfaces or fine foreground detail. Good scene-wide proximity can coexist with a noisy-looking point cloud. That is why the earlier F1 improvement should not have been treated as full validation of the installed pipeline.

## Confirmed bugs, and their relevance

1. The earlier **preset inheritance bug** is fixed: nested presets now resolve to the same budgets as Hydra. It affected guided optimization budgets, not the cause of the plain cloud's diffuse geometry.
2. A new **cached geometry dtype bug** was reproduced and fixed in `Surflo/surflo/structures/cameras.py`: the official cache has FP16 world points and guided inference has FP32 camera transforms. Their matrix multiplication previously raised a dtype error. The helper now promotes operands to a common precision and device. A regression check validates the known transformed coordinates and retained gradients; the actual cached guided reconstruction completes after this repair. This bug blocked guided validation and does not explain the plain outputs, which completed successfully before the repair.

The published checkpoint loads strictly, the learned Fourier buffers are present and finite, and the decoder/modulation weights are nonzero. The supplied alignment has singular values 0.11153, 0.11089 and 0.11048, so it is nearly isotropic rather than a collapsed affine map. The alignment formula matches the supplied dataset loader. These checks rule out specific faults; strict tensor loading alone cannot prove all training/inference computations agree.

## Normal-parameterization discrepancy tested

The paper describes a normal offset of 0.001 times scene scale; the installed implementation's default is 0.01 times the global average scale. This is a real paper/code discrepancy, but it is not confirmed as a checkpoint bug: the training convention of these weights is not stored as recoverable configuration in the release checkpoint.

Keeping the same 8,192 XYZ source samples and 100 steps, reducing the initial normal offset by ten, or using the paper's scene-scaled offset, does not improve the official reconstruction:

| Initial offset | F1 | Mean bidirectional distance |
| --- | ---: | ---: |
| Installed default | 0.683 | 0.0689 |
| One tenth of installed default | 0.655 | 0.0765 |
| Paper scene-scaled offset | 0.653 | 0.0765 |

Distances are arbitrary VGGT units. These were temporary diagnostic controls; no normal parameterization was changed in production. Earlier precision, aspect-ratio, ODE-direction and independent integration-path controls also failed to resolve the plain output.

## Decision

The plain architecture has an acknowledged coherence weakness, and the guided demo supports the practical value of its omitted correction. The current data still does not establish how much of the remaining poor geometry comes from released-weight behavior, runtime/configuration differences, sparse views or latent compression. A matched full-recipe reconstruction with a known successful reference output is required to settle that attribution.

For robotic images there is additional direct evidence of large camera-estimation errors. That is separate from the demo's coherence problem. Neither a failed reconstruction nor a better rendered image alone establishes whether compressed features are suitable for an action policy; the separate localization probe demonstrates useful coarse information, with contact-level precision still unvalidated.

Validation: cached guided run completed; dtype/gradient regression and existing preset/Hydra check passed; syntax and whitespace checks passed. Reproduction scripts are under `reproduction/`.
