"""Summarize completed cases without dropping failures from the registered study."""

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

MODELS = ["surflo", "vggt", "omega256", "omega512"]
LABELS = ["Surflo + full guidance", "VGGT", "Omega 256 Text", "Omega 512"]


def rgb_image(a, size):
    a = np.asarray(a)
    if a.ndim == 2:
        a = np.repeat(a[..., None], 3, -1)
    return Image.fromarray(np.uint8(np.clip(a, 0, 1) * 255)).resize(size, Image.Resampling.LANCZOS)


def fonts(size):
    for path in [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    ]:
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def overview(root, completed):
    margin, tile_w, tile_h = 240, 512, 190
    image = Image.new("RGB", (margin + 5 * tile_w, 130 + len(completed) * tile_h + 90), "white")
    draw = ImageDraw.Draw(image)
    draw.text((20, 12), "RoboTwin: registered 12-case reconstruction study", fill="black", font=fonts(27))
    draw.text(
        (20, 50),
        "Each tile: left wrist | right wrist. Model tiles show extracted mesh geometry; source column is RGB.",
        fill="black",
        font=fonts(21),
    )
    for col, name in enumerate(["Input RGB", *LABELS]):
        draw.text((margin + col * tile_w + 10, 94), name, fill="black", font=fonts(23))
    for row, case in enumerate(completed):
        base = np.load(Path(case["root"]) / "render_frames.npz")
        raw = np.load(Path(case["root"]) / "raw_depth_meshes/render_frames.npz")
        y = 130 + row * tile_h
        text = f"{case['task']}\nep {case['episode']} / frame {case['frame']}\n{case['phase']}"
        draw.multiline_text((8, y + 45), text, fill="black", font=fonts(19), spacing=6)
        for i in [1, 2]:
            image.paste(rgb_image(base[f"input_{i}"], (256, tile_h - 6)), (margin + (i - 1) * 256, y))
        for col, model in enumerate(MODELS, 1):
            a = base if model == "surflo" else raw
            for i in [1, 2]:
                image.paste(
                    rgb_image(a[f"{model}_shaded_{i}"], (256, tile_h - 6)), (margin + col * tile_w + (i - 1) * 256, y)
                )
    draw.text(
        (20, image.height - 72),
        "Identical source images per case; released model preprocessing; full Surflo guidance; no confidence mask for"
        " depth meshes.",
        fill="black",
        font=fonts(20),
    )
    draw.text(
        (20, image.height - 40),
        "Estimated cameras and scales differ. Shading and input-view agreement do not establish metric surface"
        " accuracy.",
        fill="black",
        font=fonts(20),
    )
    image.save(root / "wrist_geometry_overview.png")


def stage_overview(root, completed):
    margin, tile_w, tile_h = 240, 512, 190
    image = Image.new("RGB", (margin + 5 * tile_w, 130 + len(completed) * tile_h + 90), "white")
    draw = ImageDraw.Draw(image)
    draw.text((20, 12), "Surflo stages: Gaussian image fitting versus extracted surfaces", fill="black", font=fonts(27))
    draw.text(
        (20, 50),
        "Each tile: left wrist | right wrist. Same refined cameras; black mesh pixels indicate missing surface.",
        fill="black",
        font=fonts(21),
    )
    keys = ["input", "gaussian", "gaussian_normal", "mesh", "mesh_normal"]
    labels = ["Input RGB", "Gaussian RGB", "Gaussian normals", "Mesh RGB", "Mesh normals"]
    for col, name in enumerate(labels):
        draw.text((margin + col * tile_w + 10, 94), name, fill="black", font=fonts(23))
    for row, case in enumerate(completed):
        frames = np.load(Path(case["root"]) / "surflo_stage_frames.npz")
        meta = json.loads((Path(case["root"]) / "surflo_metadata.json").read_text())
        y = 130 + row * tile_h
        text = (
            f"{case['task']}\nep {case['episode']} / {case['phase']}\nGaussians:"
            f" {meta.get('full_guided_gaussians', '?')}"
        )
        draw.multiline_text((8, y + 45), text, fill="black", font=fonts(19), spacing=6)
        for col, key in enumerate(keys):
            for i in [1, 2]:
                image.paste(
                    rgb_image(frames[f"{key}_{i}"], (256, tile_h - 6)), (margin + col * tile_w + (i - 1) * 256, y)
                )
    draw.text(
        (20, image.height - 72),
        "Raw fitted-view RGB, alpha-composited Gaussian normals, and mesh normals. No metric depth ground truth.",
        fill="black",
        font=fonts(20),
    )
    draw.text(
        (20, image.height - 40),
        "A good Gaussian RGB fit can coexist with poor surface geometry. Guidance and mesh extraction are distinct"
        " stages.",
        fill="black",
        font=fonts(20),
    )
    image.save(root / "wrist_stage_overview.png")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    manifest = json.loads((root / "manifest.json").read_text())
    completed, statuses, metrics = [], [], []
    for case in manifest["cases"]:
        path = Path(case["root"]) / "status.json"
        status = json.loads(path.read_text()) if path.exists() else dict(status="prepared")
        statuses.append(
            dict(case=case["name"], status=status["status"], stage=status.get("stage"), error=status.get("error"))
        )
        if status["status"] == "completed":
            completed.append(case)
            values = json.loads((Path(case["root"]) / "comparison_metrics.json").read_text())
            stages = json.loads((Path(case["root"]) / "stage_metrics.json").read_text())
            meta = json.loads((Path(case["root"]) / "surflo_metadata.json").read_text())
            raw = json.loads((Path(case["root"]) / "raw_depth_meshes/metadata.json").read_text())
            coverage = {
                x["model"]: float(np.mean([v["coverage"] for v in x["scores"] if v["view"] in [1, 2]]))
                for x in raw["records"]
            }
            coverage["surflo"] = float(np.mean([v["mesh_coverage"] for v in stages["scores"][1:]]))
            metrics.append(
                dict(
                    case=case,
                    image_consistency=values,
                    camera_consistency=stages["camera_scores"],
                    surflo_geometry=meta,
                    mean_wrist_coverage=coverage,
                )
            )
    aggregates = {}
    if completed:
        overview(root, completed)
        stage_overview(root, completed)
        for model in MODELS:
            scores = [x["image_consistency"]["models"][model] for x in metrics]
            ckey = "surflo_refined" if model == "surflo" else model
            cameras = [view for x in metrics for view in x["camera_consistency"][ckey]]
            aggregates[model] = dict(
                mean_head_rgb_mae=float(np.mean([x["head_rgb_mae"] for x in scores])),
                mean_left_wrist_rgb_mae=float(np.mean([x["left_wrist_rgb_mae"] for x in scores])),
                mean_right_wrist_rgb_mae=float(np.mean([x["right_wrist_rgb_mae"] for x in scores])),
                mean_wrist_rgb_mae=float(np.mean([x["mean_wrist_rgb_mae"] for x in scores])),
                mean_wrist_coverage=float(np.mean([x["mean_wrist_coverage"][model] for x in metrics])),
                median_relative_camera_rotation_error_deg=float(
                    np.median([x["relative_rotation_error_deg"] for x in cameras])
                ),
                median_camera_center_direction_error_deg=float(
                    np.median([x["center_direction_error_deg"] for x in cameras])
                ),
            )
    surflo_diagnostics = {}
    if completed:
        views = [view for x in metrics for view in x["image_consistency"]["surflo_stages"][1:]]
        surflo_diagnostics = dict(
            mean_wrist_gaussian_rgb_mae=float(np.mean([v["gaussian_rgb_mae"] for v in views])),
            mean_wrist_gaussian_alpha_coverage=float(np.mean([v["gaussian_alpha_coverage"] for v in views])),
            cases_mesh_wrist_mae_exceeds_gaussian=int(
                sum(
                    x["image_consistency"]["models"]["surflo"]["mean_wrist_rgb_mae"]
                    > np.mean([v["gaussian_rgb_mae"] for v in x["image_consistency"]["surflo_stages"][1:]])
                    for x in metrics
                )
            ),
            cases_surflo_mesh_wrist_mae_exceeds_each_depth_method=sum(
                all(
                    x["image_consistency"]["models"]["surflo"]["mean_wrist_rgb_mae"]
                    > x["image_consistency"]["models"][m]["mean_wrist_rgb_mae"]
                    for m in MODELS[1:]
                )
                for x in metrics
            ),
            minimum_full_guided_gaussians=min(
                x["surflo_geometry"].get("full_guided_gaussians", 100000) for x in metrics
            ),
            cases_zero_wrist_mesh_coverage=sum(x["mean_wrist_coverage"]["surflo"] == 0 for x in metrics),
            scope=(
                "Continuous fitted-view measurements; comparisons are exploratory and are not metric geometry accuracy."
            ),
        )
    summary = dict(
        status="completed" if len(completed) == len(statuses) else "partial",
        registered_cases=len(statuses),
        completed_cases=len(completed),
        cases=statuses,
        aggregates=aggregates,
        surflo_diagnostics=surflo_diagnostics,
        records=metrics,
        scope="Input-view image consistency and recorded-camera consistency; no metric surface ground truth.",
    )
    (root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    lines = [
        "**Four-model RoboTwin reconstruction study — 2026-10-07**",
        "",
        (
            f"Completed {len(completed)}/{len(statuses)} registered cases. Every selected case remains in the status"
            " record, including failures. Policy training was not launched."
        ),
        "",
        "Findings and limitations: [interpretation](interpretation.md).",
        "",
        (
            "Protocol: [registered reconstruction protocol](protocol.md); [current policy"
            " protocol](../../../docs/omega512_experiment_protocol_20261007.md). Source tasks are rotate_qrcode,"
            " adjust_bottle, and stack_blocks_three, episodes 7/23, initial plus deterministically selected"
            " gripper-interaction frames."
        ),
        "",
        (
            "The Surflo mesh wraps its guided Gaussian representation. Its underlying learned model generates oriented"
            " surface points; plain mode does not perform the guided Gaussian optimization. VGGT/Omega meshes instead"
            " triangulate depth maps per camera. This comparison changes representation and extraction procedure, so it"
            " cannot isolate 3DGS as the cause of any difference."
        ),
        "",
        (
            "All methods receive the same saved PNGs within a case. Surflo uses released EMA, 100K query points, full"
            " default DA3/normal/rendering guidance, and default mesh extraction. Depth models use their official"
            " preprocessing and strict checkpoint loading. Primary depth meshes use no confidence mask and a shared 5%"
            " local-depth edge cutoff. Each model uses its own predicted cameras and scale."
        ),
        "",
        (
            "[Wrist geometry overview](wrist_geometry_overview.png). Each row is one case; tiles pair left/right wrist"
            " geometry. Individual figures below include input RGB and colored/shaded meshes for all three cameras."
        ),
        "",
        (
            "[Surflo stage overview](wrist_stage_overview.png) separates Gaussian RGB/normal renders from extracted mesh"
            " RGB/normal renders."
        ),
        "",
        (
            "| Method | Head RGB MAE | Left wrist | Right wrist | Mean wrist | Wrist coverage | Median wrist-camera"
            " rotation error |"
        ),
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for model, label in zip(MODELS, LABELS, strict=False):
        if model in aggregates:
            x = aggregates[model]
            lines.append(
                f"| {label} | {x['mean_head_rgb_mae']:.4f} | {x['mean_left_wrist_rgb_mae']:.4f} |"
                f" {x['mean_right_wrist_rgb_mae']:.4f} | {x['mean_wrist_rgb_mae']:.4f} |"
                f" {100*x['mean_wrist_coverage']:.1f}% | {x['median_relative_camera_rotation_error_deg']:.2f}° |"
            )
    lines += [
        "",
        (
            "RGB MAE is in [0,1]. Depth meshes are colored from their source images, while Surflo estimates a shared"
            " guided representation and colors its extracted surface. These errors measure fitted-view consistency, not"
            " comparable novel-view synthesis performance or ground-truth surface accuracy. Camera errors compare"
            " relative orientation to the recorded rig, without surface ICP. Baseline-scale translation fits are"
            " explicit in the per-case JSON."
        ),
        "",
        "| Case | Four models | Surflo Gaussian → mesh diagnostics | Interactive meshes | Status |",
        "| --- | --- | --- | --- | --- |",
    ]
    for case, status in zip(manifest["cases"], statuses, strict=False):
        name = case["name"]
        if status["status"] == "completed":
            lines.append(
                f"| {name} | [comparison]({name}/comparison.png) | [stages]({name}/surflo_stage_diagnostics.png) |"
                f" [viewer]({name}/mesh_viewer.html) | completed |"
            )
        else:
            lines.append(f"| {name} | pending | pending | pending | {status['status']}: {status.get('stage')} |")
    lines += [
        "",
        (
            "Each completed case preserves full-resolution PLY meshes: `surflo_mesh_colored.ply`, and"
            " `raw_depth_meshes/vggt_depth_mesh.ply`, `omega256_depth_mesh.ply`, `omega512_depth_mesh.ply`. The"
            " interactive viewers decimate display copies only. Plain and guided Surflo point clouds are also retained."
        ),
        "",
        (
            "A rendering defect was found during execution: the scalable renderer always removed the first unique face"
            " index, assuming a background sentinel existed, and crashed when no face reached a pixel. The fix handles"
            " full-image surfaces and empty renders. Four CUDA checks cover full coverage, partial coverage, offscreen"
            " geometry, and empty geometry: [validation](renderer_validation.json), [patch](mesh_renderer_fix.patch),"
            " [registered amendment](renderer_protocol_amendment.json). Every case is recolored and rerendered from its"
            " cached reconstruction; SHA256 checks confirm the raw Surflo mesh and guided tensors remain unchanged."
            " Original outputs and failure logs are retained under each case’s `before_renderer_fix/`. Completing a"  # noqa: RUF001 - intentional mathematical/visual labels
            " failed visualization does not turn a collapsed reconstruction into a successful surface."
        ),
        "",
        (
            "The supplied BEHAVIOR example is audited separately using exactly its original saved reconstruction:"
            " [corrected four-model comparison](behavior_renderer_audit/comparison.png), [Surflo"
            " stages](behavior_renderer_audit/surflo_stage_diagnostics.png). Its original figure and artifacts are"
            " preserved."
        ),
        "",
        "| Case | Full guided Gaussians | Mesh faces | Wrist Gaussian MAE | Wrist mesh MAE | Wrist mesh coverage |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for x in metrics:
        meta = x["surflo_geometry"]
        ss = x["image_consistency"]["surflo_stages"][1:]
        lines.append(
            f"| {x['case']['name']} | {meta.get('full_guided_gaussians', '?')} | {meta['mesh_faces']} |"
            f" {np.mean([v['gaussian_rgb_mae'] for v in ss]):.4f} |"
            f" {x['image_consistency']['models']['surflo']['mean_wrist_rgb_mae']:.4f} |"
            f" {100*x['mean_wrist_coverage']['surflo']:.1f}% |"
        )
    lines += [
        "",
        (
            "See [summary.json](summary.json) for all per-case/per-view measurements, [manifest.json](manifest.json) for"
            " inputs/checkpoints/code hashes, and each case’s `run.log` / `status.json` for execution."  # noqa: RUF001 - intentional mathematical/visual labels
            " `surflo_stage_frames.npz` preserves Gaussian alpha/depth/normal maps and mesh depth/normal maps for"
            " diagnosis."
        ),
    ]
    (root / "report.md").write_text("\n".join(lines) + "\n")
    if len(completed) == len(statuses):
        manifest["status"] = "completed"
        manifest["summary"] = str(root / "summary.json")
        manifest["report"] = str(root / "report.md")
        (root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(dict(status=summary["status"], completed_cases=len(completed), aggregates=aggregates)), flush=True)


if __name__ == "__main__":
    main()
