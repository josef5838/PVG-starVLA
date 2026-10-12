"""Package completed four-model scene reconstructions and their provenance."""

import argparse
import json
from pathlib import Path

import numpy as np
from group_reconstruction_comparisons import group


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def mesh_viewer(root, out, label):
    import open3d as o3d
    import plotly.graph_objects as go

    cases = [
        ("Surflo released EMA + guidance", root / "surflo_mesh_colored.ply"),
        ("VGGT-1B", root / "raw_depth_meshes/vggt_depth_mesh.ply"),
        ("VGGT-Omega-1B-256-Text", root / "raw_depth_meshes/omega256_depth_mesh.ply"),
        ("VGGT-Omega-1B-512", root / "raw_depth_meshes/omega512_depth_mesh.ply"),
    ]
    traces, validations = [], []
    for name, path in cases:
        mesh = o3d.io.read_triangle_mesh(str(path))
        vertices, faces = np.asarray(mesh.vertices), np.asarray(mesh.triangles)
        assert len(vertices) and len(faces) and np.isfinite(vertices).all()
        assert (faces >= 0).all() and (faces < len(vertices)).all()
        validations.append(
            dict(
                model=name,
                file=str(path),
                vertices=len(vertices),
                faces=len(faces),
                finite_vertices=True,
                valid_face_indices=True,
            )
        )
        if len(faces) > 100000:
            mesh = mesh.simplify_quadric_decimation(100000)
        vertices, faces = np.asarray(mesh.vertices), np.asarray(mesh.triangles)
        colors = np.asarray(mesh.vertex_colors)
        traces.append(
            go.Mesh3d(
                x=vertices[:, 0],
                y=vertices[:, 2],
                z=-vertices[:, 1],
                i=faces[:, 0],
                j=faces[:, 1],
                k=faces[:, 2],
                vertexcolor=colors,
                name=name,
                visible=len(traces) == 3,
                flatshading=True,
            )
        )
    title = f"{label}: individual estimated frames and scales"
    buttons = [
        dict(
            label=trace.name,
            method="update",
            args=[{"visible": [i == j for j in range(len(traces))]}, {"title": title + " — " + trace.name}],
        )
        for i, trace in enumerate(traces)
    ]
    fig = go.Figure(traces)
    fig.update_layout(
        title=title + " — VGGT-Omega-1B-512",
        scene=dict(aspectmode="data", xaxis_title="X", yaxis_title="Z", zaxis_title="−Y"),  # noqa: RUF001
        updatemenus=[dict(buttons=buttons, active=3)],
        margin=dict(l=0, r=0, b=0, t=100),
    )
    fig.write_html(out, include_plotlyjs=True, full_html=True)
    return validations


def scene_metrics(root):
    records = {row["model"]: row["scores"] for row in read(root / "raw_depth_meshes/metadata.json")["records"]}
    records["surflo"] = [row for row in read(root / "render_metadata.json")["scores"] if row["model"] == "surflo"]
    result = {}
    for model, scores in records.items():
        scores = sorted(scores, key=lambda row: row["view"])
        assert [row["view"] for row in scores] == [0, 1, 2]
        result[model] = dict(
            head_rgb_mae=scores[0]["rgb_mae"],
            left_wrist_rgb_mae=scores[1]["rgb_mae"],
            right_wrist_rgb_mae=scores[2]["rgb_mae"],
            mean_wrist_rgb_mae=(scores[1]["rgb_mae"] + scores[2]["rgb_mae"]) / 2,
        )
    assert set(result) == {"vggt", "omega256", "omega512", "surflo"}
    return result


def write_report(out, metrics, inference):
    lines = [
        "# Four-model reconstruction comparison",
        "",
        (
            "Completed on identical three-view source images for BEHAVIOR turning_on_radio "
            "(episode 197, frame 1491) and the original RoboTwin rotate_qrcode reconstruction scene. "
            "The official VGGT-Omega-1B-512 checkpoint was downloaded, verified against its official "
            "LFS SHA256 and size, and loaded with strict=True and no missing/unexpected parameters."
        ),
        "",
        (
            "Primary figures: [BEHAVIOR](behavior_comparison.png) and [RoboTwin](robotwin_comparison.png). "
            "Rows are head, left wrist, and right wrist. Each model column pairs colored mesh on the left "
            "with shaded geometry on the right."
        ),
        "",
        (
            "Interactive viewers: [BEHAVIOR](behavior_mesh_viewer.html) and "
            "[RoboTwin](robotwin_mesh_viewer.html). Select any of the four models. Display copies are "
            "decimated to 100,000 triangles; full PLY files retain their original exported geometry. "
            "Each model has its own estimated frame and arbitrary scale."
        ),
        "",
        "## Results",
        "",
        (
            "On BEHAVIOR, Omega512 substantially improves the wrist image agreement relative to Omega256, "
            "especially the right wrist. Both Omega models avoid the large sofa-colored occlusions seen "
            "in VGGT and the severely distorted Surflo wrist meshes. The shaded Omega meshes still show "
            "ripples and other local surface artifacts."
        ),
        "",
        (
            "On RoboTwin, Omega512 and Omega256 have broadly similar appearance. Omega512 slightly improves "
            "the mean wrist image agreement, with gains in the left wrist and head; its right-wrist RGB "
            "error is slightly higher than Omega256. Both have cleaner arm/table geometry than the rough "
            "Surflo mesh and the heavily distorted VGGT mesh, while boundary artifacts remain."
        ),
        "",
        "Mean wrist RGB absolute error in [0,1], using unfiltered depth meshes:",
        "",
        "| Scene | Surflo + guidance | VGGT | Omega256 | Omega512 |",
        "|---|---:|---:|---:|---:|",
    ]
    for scene, values in metrics.items():
        cells = [f"{values[model]['mean_wrist_rgb_mae']:.4f}" for model in ["surflo", "vggt", "omega256", "omega512"]]
        lines.append("| " + " | ".join([scene, *cells]) + " |")
    lines += [
        "",
        (
            "These are input-view renders at each model’s estimated cameras. RGB agreement and "  # noqa: RUF001 - intentional mathematical/visual labels
            "shaded appearance do not establish accurate metric geometry, generalization to novel views, "
            "or suitability of features for robotic control. No ground-truth depth or held-out view is used. "
            "The wrist views are mostly dominated by robot/background pixels, rather than the target object."
        ),
        "",
        "## Protocol and files",
        "",
        (
            "VGGT/Omega meshes are derived by triangulating per-view predicted depth grids, with a shared "
            "5% local-depth edge cutoff and no cross-view stitching. The primary figures use all finite "
            "positive-depth points without confidence masking. Confidence-filtered counterparts are "
            "[BEHAVIOR](behavior_confidence_filtered.png) and [RoboTwin](robotwin_confidence_filtered.png); "
            "holes in those figures may be filtering artifacts. Confidence values are not calibrated "
            "across models."
        ),
        "",
        (
            "Surflo uses its released EMA with full default rendering guidance and DA3 depth/normal experts. "
            "This is not the compressor finetuned for action learning. The previously validated RoboTwin "
            "Surflo result is reused on the exact same image hashes. Omega256 uses the 256-Text checkpoint; "
            "Omega512 uses the separate official 512 checkpoint without a text-alignment head."
        ),
        "",
        (
            "Resolution follows each installed model’s preprocessing, so this is not a resolution-matched "  # noqa: RUF001 - intentional mathematical/visual labels
            "benchmark. BEHAVIOR VGGT/Surflo uses 518×518 and Omega256 uses 256×256; "  # noqa: RUF001 - intentional mathematical/visual labels
            "RoboTwin VGGT/Surflo uses 378×518 and Omega256 uses 224×288."  # noqa: RUF001 - intentional mathematical/visual labels
        ),
        "",
    ]
    for scene, info in inference.items():
        height, width = info["input_shape"][-2:]
        lines.append(
            f"- {scene}: Omega512 actual input dimensions {height}×{width} at resolution budget 512."  # noqa: RUF001
        )
    lines += [
        "",
        (
            "See [manifest.json](manifest.json) for source roots and checkpoint provenance, "
            "[render_comparison_metrics.json](render_comparison_metrics.json) for per-view errors, "
            "and [artifact_validation.json](artifact_validation.json) for vertex/face checks."
        ),
        "",
        (
            "Full colored PLY files: `surflo_mesh_colored.ply` in each scene root, and "
            "`vggt_depth_mesh.ply`, `omega256_depth_mesh.ply`, `omega512_depth_mesh.ply` "
            "in each scene’s `raw_depth_meshes/`. The `omega_512/` folder also preserves "  # noqa: RUF001 - intentional mathematical/visual labels
            "`omega_points_all.ply`, the confidence-filtered mesh, predicted geometry tensors, "
            "and inference metadata."
        ),
        "",
    ]
    (out / "report.md").write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--behavior-root", type=Path, required=True)
    parser.add_argument("--robotwin-root", type=Path, required=True)
    parser.add_argument("--download-metadata", type=Path, required=True)
    args = parser.parse_args()
    out = args.out.resolve()
    download = read(args.download_metadata)
    assert download["status"] == "verified" and download["checksum_matches"]
    scenes = dict(behavior=args.behavior_root.resolve(), robotwin=args.robotwin_root.resolve())
    labels = dict(
        behavior="BEHAVIOR turning_on_radio — episode 197, frame 1491",
        robotwin="RoboTwin rotate_qrcode — original three-view reconstruction scene",
    )
    metrics, validations, inference = {}, {}, {}
    for scene, root in scenes.items():
        completed = read(root / "omega_512/manifest.json")
        assert completed["status"] == "completed"
        info = read(root / "omega_512/inference.json")
        assert not info["missing_keys"] and not info["unexpected_keys"]
        assert info["checkpoint_variant"] == "vggt_omega_1b_512" and info["image_resolution"] == 512
        inference[scene] = info
        group(root, out / f"{scene}_comparison.png", labels[scene], unfiltered=True)
        group(root, out / f"{scene}_confidence_filtered.png", labels[scene], unfiltered=False)
        validations[scene] = mesh_viewer(root, out / f"{scene}_mesh_viewer.html", labels[scene])
        metrics["BEHAVIOR" if scene == "behavior" else "RoboTwin"] = scene_metrics(root)
        launch_path = root / "omega_512/launch_manifest.json"
        launch = read(launch_path) if launch_path.exists() else {}
        launch.pop("blocked_by", None)
        launch.update(
            status="completed",
            source_root=str(root),
            checkpoint=download["path"],
            image_resolution=512,
            download=download,
            completed_manifest=str(root / "omega_512/manifest.json"),
        )
        write(launch_path, launch)
        manifest_path = root / "manifest.json"
        manifest = read(manifest_path)
        manifest.pop("pending_models", None)
        manifest.update(
            comparison_status="all four models reconstructed and visualized",
            omega_256=str(root / "omega_256/manifest.json"),
            omega_512=str(root / "omega_512/manifest.json"),
            grouped_comparison=str(out / f"{scene}_comparison.png"),
        )
        if scene == "robotwin":
            manifest["status"] = "completed"
        write(manifest_path, manifest)
    scope = (
        "Input-view consistency, not ground-truth geometry. Model resolutions and camera estimates differ. "
        "VGGT/Omega meshes use no confidence masking, with shared depth-grid connectivity and 5% edge cutoff. "
        "Surflo includes per-scene default rendering guidance."
    )
    write(out / "render_comparison_metrics.json", dict(scope=scope, records=metrics))
    write(out / "artifact_validation.json", validations)
    manifest = read(out / "manifest.json")
    manifest.pop("pending_checkpoint", None)
    manifest.pop("download_block", None)
    manifest.update(
        status="completed: all four models on both scenes",
        download=download,
        omega512_inference=inference,
        scenes={k: str(v) for k, v in scenes.items()},
        comparison_notes=[
            "Identical source image hashes within each scene.",
            "Official Omega resolution budgets; VGGT/Surflo use original 518px preprocessing.",
            "Independent estimated cameras and arbitrary scale; no metric ground truth.",
            "Primary VGGT/Omega depth meshes use no confidence mask and shared 5% edge cutoff.",
            "Surflo released EMA includes default guidance with depth/normal experts.",
            "Interactive viewers decimate display copies only; PLY files retain full geometry.",
        ],
    )
    write(out / "manifest.json", manifest)
    write_report(out, metrics, inference)
    print(json.dumps(dict(status=manifest["status"], metrics=metrics)), flush=True)


if __name__ == "__main__":
    main()
