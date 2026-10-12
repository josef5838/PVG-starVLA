> Latest result: [full reference reproduction](../full_reproduction/README.md) now passes the complete 16-view expert-guided mesh workflow and matches the authors' published garden mesh. Plain/guided F1 on a 16-view ground-truth Ignatius sample is 90.44%/90.72%. The full robotic control still has large camera errors. Earlier reduced diagnostics below are preserved as historical results.

> Follow-up: the [pipeline audit](pipeline_audit.md) adds a guided official-demo reconstruction, fixes a cached FP16/FP32 camera-transform bug, and tests the normal-offset discrepancy. Guidance recovers recognizable rendered structure but loses coverage; the 128-token capacity limit remains unproven.

# Surflo robotic reconstruction and feature audit — 2026-10-02

**Decision:** the frozen Surflo encoder retains useful information about object location. Its current plain reconstruction is unreliable on these robotic three-view inputs, and this audit does not validate the 128 compressed tokens as a sufficient visual representation for precise grasping. I recommend preserving local VGGT/RGB features alongside the compressed Surflo tokens in the next policy experiment.

The measurements below supersede the earlier hypothesis that a generally broken decoder or a corrupt checkpoint explains the noisy plots. They do not establish successful robotic grasping.

## Frozen-feature localization

A separate ridge predictor was fitted to frozen visual features from 100 recorded initial scenes. It used 80 scenes for fitting, with the regularization selected by four-fold cross-validation inside those 80, then evaluated on 20 held-out probe scenes. The input was RGB only: no robot state, actions, seed, episode ID, or instruction was supplied to the predictor. All cameras were simultaneous, with the policy's 518×518 preprocessing.

These are features **before either policy's learned linear/residual projection and DiT**. These held-out scenes are held out from the probe, not a new end-to-end policy rollout evaluation. Both policies use the same frozen release backbone.

| Visual representation | Mean XY error | Median XY error |
| --- | ---: | ---: |
| Constant training-mean position | 7.30 cm | 7.12 cm |
| Current policy Surflo features, 132×512 | 2.63 cm | 2.23 cm |
| Surflo with official spatial preprocessing, 132×512 | 4.53 cm | 4.43 cm |
| VGGT final-layer head-camera patches, pooled to 4×4 | 2.23 cm | 1.62 cm |
| Head-camera RGB downsampled to 24×32 | 3.67 cm | 3.75 cm |

The exact JSON is authoritative for rounded figures: [probe results](../feature_probe/features/probe_results.json). [Localization plot](feature_localization.png) includes bootstrap intervals over the 20 test scenes. The small Surflo/VGGT difference on this split does not establish a statistically robust advantage for VGGT.

Object XY labels were recovered from the recorded simulator seeds. The deterministic placement RNG was checked against 14 recreated simulator scenes, with maximum XY discrepancy below 1.4 micrometres, and the recovered object model IDs match all 100 recorded episodes. The other 86 labels use that validated RNG procedure. The probe uses only XY, not the unsettled Z value.

**Limits:** all 100 demonstrations are right-side examples. The probe does not establish left-side generalization, orientation accuracy, gripper timing, contact accuracy, temporal reasoning, or success of either trained action model. Centimetre-scale localization can still miss the QR object during grasping.

## Camera geometry is wrong before the surface decoder

For training episode 0, frame 20 (raw episode 51), the three recorded camera calibrations were compared with VGGT estimates. Extrinsics were compared relative to the first/head camera; translation scale was fitted once from all camera centers, without fitting a rotation or using surface ICP.

| Input preprocessing | Wrist relative rotation error | Wrist center error after best scale | Estimated/true focal length |
| --- | --- | --- | --- |
| Policy square 518×518 | 21.5°, 59.7° | 50.6 cm, 28.0 cm | fx 2.53–3.68×; fy 1.88–2.70× |
| Aspect preserved 518×378 | 23.5°, 13.2° | 44.9 cm, 21.7 cm | 1.44–1.56× |

[Exact calibration measurements](camera_calibration.json); [camera-center plot](camera_calibration.png).

Aspect preservation helps some estimates but does not produce a consistent robotic camera rig. More ODE steps cannot fix these upstream errors. The three initial training scenes recreated for a depth control are also problematic: the signed best head-frame scale is negative in all three, meaning the predicted rig points broadly in the wrong direction. Positive baseline-length alignment leaves wrist center errors of roughly 50–121 cm. This is direct evidence of bad camera geometry, not a determination of its single cause. Narrow views, weak overlap, robot appearances and transparent objects are plausible contributors, not separately established explanations.

## Ground-truth spatial input control

Raw episodes 51, 60 and 38 were recreated with seeds 131, 162 and 91. Original recorded RGB was used as model input. Simulator depth and calibration were captured independently; recreated RGB differs from recorded RGB by approximately 2.3–5.2 intensity levels on the 0–255 scale. See [capture verification](robot_ground_truth/records.json).

The privileged control replaces the explicit depth, calibration and backprojected spatial conditioning with simulator values. It retains the original VGGT appearance features and implicit camera embeddings. The head-frame orientation/origin and a positive scale from camera baseline lengths determine one common transform for all clouds in each scene. No surface ICP is performed.

This is a diagnostic control, **not a deployable RGB-only fix**, and it does not fully condition VGGT on known poses. The original camera estimates are so inconsistent that metric comparisons are dominated by camera alignment; their very large distances must not be interpreted as an intrinsic Surflo scale error.

The spatial correction brings the output back into the tabletop workspace, but does not recover clean surfaces. Compared with the random source cloud under the same corrected spatial input, decoding **worsens** the distance to observed ground-truth surfaces in all three scenes:

| Raw episode | Random source mean distance | Decoded mean distance | Source / decoded F1 at 5 cm |
| --- | ---: | ---: | ---: |
| 51 | 5.69 cm | 10.32 cm | 0.679 / 0.551 |
| 60 | 5.62 cm | 13.43 cm | 0.686 / 0.274 |
| 38 | 5.64 cm | 12.28 cm | 0.676 / 0.370 |

Thus incorrect explicit spatial inputs are a substantial problem, and a simple calibration/depth substitution is insufficient to fix robotic plain reconstruction. Appearance and implicit camera conditioning remain estimated in this control, so it does not isolate information loss in the compressor from decoder or camera-feature mismatch. It also does not refute the useful location information measured by the frozen-feature probe.

[Robot geometry measurements](robot_geometry_scores.json); [three-scene comparison](robot_geometry_controls.png); [alignment protocol](robot_geometry_controls.json). Scoring compares visible simulator surfaces in a fixed tabletop workspace, with up to 8192 sampled points per cloud. Prediction sampling includes finite outliers outside the workspace, while the preview shows workspace points only. Predicted occluded surfaces can be penalized by this visible-surface reference.

## The release decoder improves metrics on an official benchmark control

The downloaded release checkpoint SHA-256 is `55fddb5dcbb5d375ba8036ec959acd215ee8ffd89e173cdabe1199e034052662`, matching the published file. Strict EMA loading succeeds: initialized EMA, 502,500 updates, epoch 200.

The official preprocessed TNT/Ignatius four-view sample and its supplied mesh-surface ground truth were evaluated. Ground truth was transformed into the cached VGGT frame using the provided alignment. All candidates share that alignment, radius culling and sampling protocol; no candidate receives its own ICP fit.

| Reconstruction | F1 at 1% GT bounding-box diagonal | Mean bidirectional nearest distance |
| --- | ---: | ---: |
| Raw VGGT | 0.645 | 0.1060 |
| Standard Surflo, 100 steps | 0.765 | 0.0592 |
| Reversed velocity sign | 0.164 | 0.1823 |
| Reversed time integration | 0.319 | 0.1956 |

Distances here are arbitrary VGGT coordinate units, not metres. This is one benchmark control rather than reproduction of the paper's full evaluation. It is not a clean visual reconstruction success, and therefore does not fully validate the installed reconstruction pipeline or establish that the remaining failure is specific to robotics. Surflo improves coverage and overall geometric accuracy on this control while retaining scattered outliers. Plot appearance and agreement with VGGT alone were misleading quality checks. Reversing the flow sign makes some plots look cleaner but damages actual reconstruction accuracy; it was not installed as a fix.

[Exact benchmark scores](official_geometry_scores.json); [cached/re-encoded previews](official_control.png).

## Implemented fixes and verification

1. **Preset inheritance in `Surflo/surflo/api.py`.** `load_preset` previously merged only `default` and the requested preset, ignoring intermediate declared parents. `no_expert → minimal → default` therefore used the wrong optimization budget. It now resolves same-group parents recursively, honors `_self_` order and detects cycles. The supported presets were checked against independent Hydra composition. The repaired `no_expert` sets `num_steps=50`, with the enabled biphase schedule taking 25+50=75 flow steps, 16 inner iterations and 100 additional iterations.
2. **Explicit policy geometry convention in `starVLA/model/framework/VLM4A/SurfloDiT.py`.** `framework.surflo.geometry_preprocessing` accepts `legacy` or `official`. The official mode uses depth-backprojected points and passes the released model's per-scene normalization statistics. The default remains `legacy`, because existing checkpoints were trained with it and official preprocessing worsened this probe. Changing the convention at evaluation time would change the learned policy's inputs. It is an option for controlled retraining, not a proven performance fix.
3. **Reproducible diagnostics.** Added label extraction/verification, feature probing, simulator geometry capture, privileged geometry control and scoring tools under `tools/robotwin_surflo_*.py`. Additional control scripts are preserved under [reproduction](reproduction/).

Fifteen regression checks passed, including policy conditioning/checkpoint checks and the preset/Hydra comparison. Tests were invoked directly because pytest is unavailable in the installed environments. A real GPU check confirms both policy modes match their respective reference feature paths exactly: shape `(1, 132, 512)`, maximum absolute error 0. Syntax compilation and diff whitespace checks passed.

[GPU equivalence result](policy_geometry_equivalence.json). The guided no-expert reconstruction was rerun after the preset repair with explicit global RNG seeding. It produces 5,170 displayed points from 5,450 retained Gaussians and reduces scattered noise, but does not validate QR-object or grasp geometry: [updated guided preview](guided_fixed_preset_preview.png), [PLY](robot_guided_no_expert_fixed_preset.ply).

## Recommended next training decision

Keep Surflo available as a coarse scene representation. Add a small spatial grid of VGGT patch features, with camera identity, as a separate visual input to the action model, so fine visual detail has a path around the 128-token compressor. Compare a VGGT-only policy with a hybrid policy under the same demonstration split and rollout budget. The current feature probe motivates this comparison but does not establish which policy will win.

For reconstruction specifically, known calibration/depth helps diagnose the problem but does not yet produce reliable grasp geometry. A robotics-specific geometry/encoder adaptation would need simulator depth and camera supervision, including correct wrist-camera poses over time. A learnable projection after a frozen compressor can adapt the features the compressor retains; it cannot repair faulty camera/depth estimation or recover detail already discarded upstream.
