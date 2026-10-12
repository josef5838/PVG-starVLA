"""Run the local VGGT-Omega geometry heads on saved BEHAVIOR comparison images."""

import argparse
import hashlib
import json
import logging
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import trimesh
from render_behavior_reconstruction import cpu, export_vggt, write_json

OMEGA_ROOT = Path("/export/home/haoliang/projects/vggt-omega-action-pred")
sys.path.insert(0, str(OMEGA_ROOT))


def infer(args, out):
    from surflo.utils.geometry import depths_to_points_parallel_batched
    from vggt_omega.models import VGGTOmega
    from vggt_omega.utils.load_fn import load_and_preprocess_images
    from vggt_omega.utils.pose_enc import encoding_to_camera

    paths = sorted((args.root / "inputs").glob("view_*.png"))
    assert len(paths) == 3
    original = json.loads((args.root / "inputs.json").read_text())
    hashes = {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}
    assert list(hashes.values()) == [item["sha256"] for item in original["cameras"]]
    logging.info("Loading VGGT-Omega camera, depth, and alignment heads: %s", args.checkpoint)
    state = torch.load(args.checkpoint, map_location="cpu", mmap=True, weights_only=True)
    enable_alignment = any(key.startswith("text_alignment_head.") for key in state)
    model = VGGTOmega(enable_camera=True, enable_depth=True, enable_alignment=enable_alignment).cuda().eval()
    loaded = model.load_state_dict(state, strict=True)
    del state
    images = load_and_preprocess_images([str(p) for p in paths], image_resolution=args.resolution).cuda()
    with torch.inference_mode():
        prediction = model(images)
        extrinsics, intrinsics = encoding_to_camera(prediction["pose_enc"], images.shape[-2:])
        points = depths_to_points_parallel_batched(intrinsics, extrinsics, prediction["depth"], to_world=True)
    batch = dict(
        rgb_images=images.unsqueeze(0),
        vggt_world_points=points,
        vggt_depth=prediction["depth"],
        vggt_depth_conf=prediction["depth_conf"],
        vggt_intrinsics=intrinsics,
        vggt_extrinsics=extrinsics,
    )
    for key, value in batch.items():
        if not torch.isfinite(value).all():
            raise RuntimeError(f"Nonfinite Omega output: {key}")
    torch.save(cpu(batch), out / "geometry_batch.pt")
    stats = export_vggt(batch, out, prefix="omega")
    write_json(
        out / "inference.json",
        dict(
            checkpoint=str(args.checkpoint.resolve()),
            checkpoint_bytes=args.checkpoint.stat().st_size,
            checkpoint_variant=args.checkpoint.stem,
            checkpoint_loading="strict=True; all checkpoint parameters loaded",
            text_alignment_head=enable_alignment,
            missing_keys=loaded.missing_keys,
            unexpected_keys=loaded.unexpected_keys,
            source_images=hashes,
            source_manifest=str(args.root / "manifest.json"),
            input_shape=list(images.shape),
            image_resolution=args.resolution,
            preprocessing="Official installed VGGT-Omega balanced loader; native source PNGs identical to Surflo/VGGT.",
            comparison_resolution_note=(
                f"Omega resolution budget {args.resolution}; actual input height/width {list(images.shape[-2:])}."
                " Earlier model input dimensions are stored in the parent geometry_batch.pt."
            ),
            coordinate_frame=(
                "Omega predicted camera frame, arbitrary scale; separate gauge from VGGT/Surflo. No GT geometry or ICP."
            ),
            mesh=stats,
        ),
    )
    logging.info("Omega PLY exports complete: %s", stats)


def render(args, out):
    from surflo.rendering.mesh import MeshRasterizer, ScalableMeshRenderer
    from surflo.structures.cameras import is_in_view_frustum
    from surflo.structures.mesh import Meshes
    from surflo.utils.guided_result import build_refined_cameras_from_guided_result

    batch = torch.load(out / "geometry_batch.pt", map_location="cuda", weights_only=True)
    cameras = build_refined_cameras_from_guided_result(batch, 0, {}, apply_camera_correction=False, data_device="cuda")
    source = trimesh.load(out / "omega_depth_mesh.ply", process=False)
    mesh = Meshes(
        verts=torch.tensor(np.asarray(source.vertices), dtype=torch.float32, device="cuda"),
        faces=torch.tensor(np.asarray(source.faces), dtype=torch.int32, device="cuda"),
        verts_colors=torch.tensor(
            np.array(source.visual.vertex_colors[:, :3]) / 255, dtype=torch.float32, device="cuda"
        ),
    )
    renderer = ScalableMeshRenderer(rasterizer=MeshRasterizer(cameras=cameras, use_opengl=False))
    frames = {}
    records = []
    with torch.no_grad():
        for index, camera in enumerate(cameras):
            keep = is_in_view_frustum(mesh.verts, camera)[mesh.faces.long()].any(-1)
            visible = Meshes(verts=mesh.verts, faces=mesh.faces[keep], verts_colors=mesh.verts_colors)
            image = renderer(visible, cam_idx=index, return_depth=True, return_normals=True, use_antialiasing=False)
            rgb = image["rgb"].squeeze(0).clamp(0, 1)
            normals = image["normals"].squeeze(0)
            light = torch.tensor([0.3, -0.6, -0.7], device="cuda")
            light /= light.norm()
            gray = (0.2 + 0.8 * (normals * light).sum(-1).abs()).clamp(0, 1)
            gray = torch.where(image["depth"].squeeze() > 0, gray, torch.zeros_like(gray))
            frames[f"omega_{index}"] = rgb.cpu().numpy()
            frames[f"omega_shaded_{index}"] = gray.cpu().numpy()
            target = batch["rgb_images"][0, index].permute(1, 2, 0).float()
            records.append(
                dict(
                    view=index,
                    rgb_mae=float((rgb - target).abs().mean()),
                    coverage=float((image["depth"] > 0).float().mean()),
                )
            )
    np.savez_compressed(out / "render_frames.npz", **frames)
    previous = np.load(args.root / "render_frames.npz")
    rows = [
        ("Input RGB", [previous[f"input_{i}"] for i in range(3)]),
        ("VGGT depth mesh (518px)", [previous[f"vggt_{i}"] for i in range(3)]),
        (f"VGGT Omega depth mesh ({args.resolution}px)", [frames[f"omega_{i}"] for i in range(3)]),
        ("VGGT Omega mesh geometry", [frames[f"omega_shaded_{i}"] for i in range(3)]),
        ("Surflo guided mesh (518px)", [previous[f"surflo_{i}"] for i in range(3)]),
    ]
    if args.resolution != 256 and (args.root / "omega_256/render_frames.npz").exists():
        omega256 = np.load(args.root / "omega_256/render_frames.npz")
        rows.insert(2, ("VGGT Omega depth mesh (256px)", [omega256[f"omega_{i}"] for i in range(3)]))
    fig, axes = plt.subplots(len(rows), 3, figsize=(12, 3.2 * len(rows)))
    for row, (label, images) in enumerate(rows):
        for index, image in enumerate(images):
            axes[row, index].imshow(image, cmap="gray", vmin=0, vmax=1)
            axes[row, index].axis("off")
            if row == 0:
                axes[row, index].set_title(["Head", "Left wrist", "Right wrist"][index])
            if index == 0:
                axes[row, index].text(
                    -0.03, 0.5, label, transform=axes[row, index].transAxes, rotation=90, ha="right", va="center"
                )
    input_metadata = json.loads((args.root / "inputs.json").read_text())
    scene_label = input_metadata.get("scene_label", "BEHAVIOR turning_on_radio")
    fig.suptitle(
        f"{scene_label}: same source images\nDepth meshes use the same triangulation/filtering procedure; each model"
        " uses its estimated cameras",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(out / "camera_render_comparison.png", dpi=150)
    plt.close(fig)
    write_json(
        out / "render_metadata.json",
        dict(
            records=records,
            scope=(
                "Input-camera renders, not held-out views or ground-truth geometry. Models have different input"
                " resolutions and arbitrary scales; confidence scores are not calibrated across models."
            ),
        ),
    )


def viewer(args, out):
    import open3d as o3d
    import plotly.graph_objects as go

    traces = []
    cases = [
        (f"VGGT Omega depth mesh ({args.resolution}px)", out / "omega_depth_mesh.ply"),
        ("VGGT depth mesh", args.root / "vggt_depth_mesh.ply"),
        ("Surflo guided mesh", args.root / "surflo_mesh_colored.ply"),
    ]
    if args.resolution != 256 and (args.root / "omega_256/omega_depth_mesh.ply").exists():
        cases.insert(1, ("VGGT Omega depth mesh (256px)", args.root / "omega_256/omega_depth_mesh.ply"))
    for label, path in cases:
        mesh = o3d.io.read_triangle_mesh(str(path))
        if len(mesh.triangles) > 100000:
            mesh = mesh.simplify_quadric_decimation(100000)
        vertices, faces, colors = np.asarray(mesh.vertices), np.asarray(mesh.triangles), np.asarray(mesh.vertex_colors)
        traces.append(
            go.Mesh3d(
                x=vertices[:, 0],
                y=vertices[:, 2],
                z=-vertices[:, 1],
                i=faces[:, 0],
                j=faces[:, 1],
                k=faces[:, 2],
                vertexcolor=colors,
                name=label,
                visible=len(traces) == 0,
                flatshading=True,
            )
        )
    buttons = [
        dict(label=trace.name, method="update", args=[{"visible": [i == j for j in range(len(traces))]}])
        for i, trace in enumerate(traces)
    ]
    fig = go.Figure(traces)
    scene_label = json.loads((args.root / "inputs.json").read_text()).get("scene_label", "BEHAVIOR")
    fig.update_layout(
        title=f"{scene_label} mesh comparison — each model has its own estimated scale/frame",
        scene=dict(aspectmode="data", xaxis_title="X", yaxis_title="Z", zaxis_title="−Y"),  # noqa: RUF001
        updatemenus=[dict(buttons=buttons)],
        margin=dict(l=0, r=0, b=0, t=90),
    )
    fig.write_html(out / "mesh_comparison_viewer.html", include_plotlyjs=True, full_html=True)
    validations = []
    for path in out.glob("*.ply"):
        if path.name.endswith("_conf2.ply"):
            continue
        source = trimesh.load(path, process=False)
        vertices = np.asarray(source.vertices)
        record = dict(file=path.name, vertices=len(vertices), finite_vertices=bool(np.isfinite(vertices).all()))
        assert len(vertices) > 0 and record["finite_vertices"]
        if isinstance(source, trimesh.Trimesh):
            faces = np.asarray(source.faces)
            record.update(
                faces=len(faces), valid_face_indices=bool((faces >= 0).all() and (faces < len(vertices)).all())
            )
            assert record["valid_face_indices"]
        validations.append(record)
    write_json(out / "artifact_validation.json", validations)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=OMEGA_ROOT / "ckpt/vggt_omega_1b_256_text.pt")
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument("--render-only", action="store_true")
    args = parser.parse_args()
    args.root = args.root.resolve()
    out = args.root / f"omega_{args.resolution}"
    out.mkdir(exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    torch.cuda.set_device(0)
    torch.manual_seed(42)
    if not args.render_only:
        infer(args, out)
    render(args, out)
    viewer(args, out)
    write_json(
        out / "manifest.json",
        dict(
            status="completed",
            source_root=str(args.root),
            inference=json.loads((out / "inference.json").read_text()),
            mesh=str(out / "omega_depth_mesh.ply"),
            raw_points=str(out / "omega_points_all.ply"),
            camera_render=str(out / "camera_render_comparison.png"),
            viewer=str(out / "mesh_comparison_viewer.html"),
        ),
    )
    logging.info("Completed: %s", out)


if __name__ == "__main__":
    main()
