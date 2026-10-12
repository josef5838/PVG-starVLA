"""Independent VGGT-Omega reconstruction of synchronized RoboTwin camera histories.

Every model call contains exactly one camera's past frames. No calibration is
fed into inference. Recorded calibration is used only for descriptive checks.
"""

import argparse
import gc
import hashlib
import io
import json
import os
import shlex
import shutil
import sys
import time
from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import trimesh
from PIL import Image

CAMERAS = ["head_camera", "left_camera", "right_camera"]
LABELS = ["Head", "Left wrist", "Right wrist"]
ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OMEGA = Path("/export/home/haoliang/projects/vggt-omega-action-pred")


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def reproduce_command(manifest):
    args = manifest["args"]
    command = [manifest["python"], "tools/robotwin_omega_camera_streams.py"]
    for key in ["data_root", "omega_root", "checkpoint", "episode", "chunk_length", "stride", "resolution", "gpu"]:
        command += ["--" + key.replace("_", "-"), str(args[key])]
    if args.get("end_frame") is not None:
        command += ["--end-frame", str(args["end_frame"])]
    command += ["--tasks", *args["tasks"], "--out", str(args["out"]) + "_rerun"]
    return shlex.join(command)


def select_frames(f, length, stride, end_frame):
    n = len(f["joint_action/vector"])
    span = (length - 1) * stride
    if n <= span:
        raise ValueError(f"Episode has {n} frames, insufficient for {length} at stride {stride}")
    if end_frame is None:
        grip = np.stack([f[f"joint_action/{side}_gripper"][:] for side in ["left", "right"]], -1)
        closures = [t for side in range(2) for t in range(1, n) if grip[t, side] < 0.5 and np.any(grip[:t, side] > 0.8)]
        end = min(closures) if closures else n // 2
        rule = "first closure below 0.5 after open above 0.8" if closures else "episode midpoint"
        end = max(end, span)
    else:
        end, rule = end_frame, "explicit end frame"
    if not span <= end < n:
        raise ValueError(f"End frame must be in [{span}, {n-1}], got {end}")
    return list(range(end - span, end + 1, stride)), rule


def unproject(depth, intrinsics, extrinsics):
    n, h, w = depth.shape
    yy, xx = np.mgrid[:h, :w]
    pixels = np.stack([xx, yy, np.ones_like(xx)], -1).astype(np.float32)
    rays = np.einsum("hwj,nij->nhwi", pixels, np.linalg.inv(intrinsics))
    camera = rays * depth[..., None]
    world = np.einsum("nhwj,nji->nhwi", camera - extrinsics[:, None, None, :, 3], extrinsics[:, :, :3])
    return camera, world


def relative_cameras(e):
    r, t = e[:, :, :3], e[:, :, 3]
    c = -np.einsum("nji,nj->ni", r, t)
    return r @ r[0].T, (c - c[0]) @ r[0].T


def angle(r):
    return np.degrees(np.arccos(np.clip((np.trace(r, axis1=-2, axis2=-1) - 1) / 2, -1, 1)))


def diagnostics(e, recorded, depth, conf, images):
    pr, pc = relative_cameras(e)
    gr, gc_ = relative_cameras(recorded)
    rotation_error = angle(pr @ gr.transpose(0, 2, 1))
    median_depth = float(np.median(depth))
    pspan = float(np.max(np.linalg.norm(pc, axis=-1)))
    gspan = float(np.max(np.linalg.norm(gc_, axis=-1)))
    fit = None
    if gspan > 0.001 and pspan > 1e-6:
        s = float(np.sum(pc * gc_) / np.sum(pc**2))
        fit = {
            "positive_scale": s if s > 0 else None,
            "signed_scale": s,
            "rms_center_residual_cm": float(np.sqrt(np.mean(np.sum((s * pc - gc_) ** 2, -1))) * 100) if s > 0 else None,
            "scope": "One fitted scale; relative orientation fixed by first recorded camera. No ICP.",
        }
    return dict(
        relative_rotation_error_deg=rotation_error.tolist(),
        max_relative_rotation_error_deg=float(rotation_error.max()),
        predicted_rotation_from_first_deg=angle(pr).tolist(),
        recorded_rotation_from_first_deg=angle(gr).tolist(),
        predicted_camera_centers_in_first_camera=pc.tolist(),
        recorded_camera_centers_in_first_camera_m=gc_.tolist(),
        recorded_max_camera_displacement_m=gspan,
        predicted_max_camera_displacement_over_median_depth=pspan / median_depth,
        translation_scale_fit=fit,
        median_depth_per_frame=np.median(depth, axis=(1, 2)).tolist(),
        confidence_percentiles=np.percentile(conf, [0, 10, 25, 50, 75, 90, 100]).tolist(),
        consecutive_rgb_mae=np.mean(np.abs(np.diff(images, axis=0)), axis=(1, 2, 3)).tolist(),
        interpretation="Camera consistency against recorded extrinsics, not ground-truth surface accuracy.",
    )


def sample(points, colors, mask, limit, rng):
    p, c = points.reshape(-1, 3), colors.reshape(-1, 3)
    indices = np.flatnonzero(mask.ravel())
    if len(indices) > limit:
        indices = rng.choice(indices, limit, replace=False)
    return p[indices], c[indices]


def latest_mesh(points, colors, mask, path):
    h, w = mask.shape
    ids = np.arange(h * w).reshape(h, w)
    a, b, c, d = ids[:-1, :-1], ids[:-1, 1:], ids[1:, :-1], ids[1:, 1:]
    faces = np.concatenate([np.stack([a, c, b], -1).reshape(-1, 3), np.stack([b, c, d], -1).reshape(-1, 3)])
    vertices = points.reshape(-1, 3)
    valid = mask.ravel()[faces].all(-1)
    triangles = vertices[faces]
    edges = np.linalg.norm(triangles - np.roll(triangles, 1, axis=1), axis=-1).max(-1)
    valid &= edges < 0.05 * np.median(points[..., 2][mask])
    mesh = trimesh.Trimesh(
        vertices=vertices,
        faces=faces[valid],
        vertex_colors=(colors.reshape(-1, 3) * 255).astype(np.uint8),
        process=False,
    )
    mesh.remove_unreferenced_vertices()
    mesh.export(path)
    return dict(
        vertices=len(mesh.vertices),
        faces=len(mesh.faces),
        filtering="Per-frame bottom 10% confidence removed; triangle edges < 5% median camera depth",
    )


def display_xyz(points):
    return points[..., [0, 2, 1]] * np.array([1, 1, -1])


def plot_cloud(ax, points, colors, centers, title, bounds):
    ax.set_facecolor("#101723")
    for axis in [ax.xaxis, ax.yaxis, ax.zaxis]:
        axis.set_pane_color((0.08, 0.11, 0.17, 1))
        axis.label.set_color("white")
        axis._axinfo["grid"]["color"] = (0.35, 0.4, 0.5, 0.5)
    p = display_xyz(points)
    ax.scatter(*p.T, c=np.clip(colors, 0, 1), s=0.35, rasterized=True, depthshade=False)
    cc = display_xyz(centers)
    ax.plot(*cc.T, color="red", marker="o", markersize=3, linewidth=1)
    ax.set_title(title, fontsize=10, color="white")
    for setter, interval in zip([ax.set_xlim, ax.set_ylim, ax.set_zlim], bounds, strict=False):
        setter(interval)
    ax.set_box_aspect([1, 1, 1])
    ax.view_init(elev=22, azim=-65)
    ax.set_xlabel("X")
    ax.set_ylabel("Z")
    ax.set_zlabel("-Y")
    ax.tick_params(labelsize=6, colors="white")


def previews(out, frames, images, depth, conf, cam_points, cloud_sets, centers):
    low, high = np.percentile(depth, [2, 98])
    cmin, cmax = np.percentile(conf, [2, 98])
    normals = np.cross(np.gradient(cam_points, axis=1), np.gradient(cam_points, axis=2))
    normals /= np.maximum(np.linalg.norm(normals, axis=-1, keepdims=True), 1e-8)
    fig, axes = plt.subplots(4, len(frames), figsize=(3 * len(frames), 9))
    for i, frame in enumerate(frames):
        rows = [
            (images[i], None, None, None),
            (depth[i], "viridis", low, high),
            (conf[i], "magma", cmin, cmax),
            ((normals[i] + 1) / 2, None, None, None),
        ]
        for row, (value, cmap, vmin, vmax) in enumerate(rows):
            axes[row, i].imshow(value, cmap=cmap, vmin=vmin, vmax=vmax)
            axes[row, i].set_xticks([])
            axes[row, i].set_yticks([])
            if i == 0:
                axes[row, i].set_ylabel(["RGB", "Depth", "Confidence", "Depth normals"][row])
        axes[0, i].set_title(f"Frame {frame}")
    fig.suptitle(
        f"{out.parent.name} / {out.name}: one independent {len(frames)}-image call\n"
        "Shared depth/confidence color ranges across this stream; depth has arbitrary scale"
    )
    fig.tight_layout()
    fig.savefig(out / "temporal_preview.png", dpi=110)
    plt.close(fig)
    all_p = display_xyz(cloud_sets[0][0])
    limits = np.percentile(all_p, [1, 99], axis=0)
    middle = limits.mean(0)
    extent = (limits[1] - limits[0]).max() * 0.55
    bounds = np.stack([middle - extent, middle + extent], -1)
    fig = plt.figure(figsize=(12, 5), facecolor="#101723")
    for i, (p, c) in enumerate(cloud_sets):
        plot_cloud(
            fig.add_subplot(1, 2, i + 1, projection="3d"),
            p,
            c,
            centers,
            [f"Joint {len(frames)}-image output: all points", "Last-image output (conditioned on all images)"][i],
            bounds,
        )
    fig.suptitle(
        f"{out.parent.name} / {out.name}\n"
        "Separate local coordinate frame; superposed dynamic objects can appear duplicated",
        color="white",
    )
    fig.tight_layout()
    fig.savefig(out / "geometry_preview.png", dpi=150)
    plt.close(fig)


def camera_preview(out, frames, stats):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for key, label in [("predicted_rotation_from_first_deg", "Omega"), ("recorded_rotation_from_first_deg", "Recorded")]:
        axes[0].plot(frames, stats[key], marker=".", label=label)
    axes[0].set_ylabel("Rotation from first camera (degrees)")
    axes[0].legend()
    axes[1].plot(frames, np.linalg.norm(stats["predicted_camera_centers_in_first_camera"], axis=-1), label="Omega")
    axes[1].set_ylabel("Omega displacement (arbitrary units)")
    ax2 = axes[1].twinx()
    ax2.plot(
        frames,
        np.linalg.norm(stats["recorded_camera_centers_in_first_camera_m"], axis=-1),
        color="orange",
        label="Recorded",
    )
    ax2.set_ylabel("Recorded displacement (meters)", color="orange")
    axes[2].plot(frames, stats["median_depth_per_frame"], marker=".")
    axes[2].set_ylabel("Median depth (arbitrary units)")
    for ax in axes:
        ax.set_xlabel("Recorded frame")
        ax.grid(alpha=0.2)
    fig.suptitle(
        f"{out.parent.name} / {out.name}: temporal camera and depth checks\n"
        "Translation traces have different units; recorded extrinsics have not been replay-verified"
    )
    fig.tight_layout()
    fig.savefig(out / "camera_consistency.png", dpi=130)
    plt.close(fig)


def viewer(out, frames, point_sets, centers):
    """Dependency-free orbit viewer: each frame remains individually selectable."""
    payload = dict(
        frames=frames,
        points=[display_xyz(p).round(5).tolist() for p, _ in point_sets],
        colors=[(np.clip(c, 0, 1) * 255).astype(np.uint8).tolist() for _, c in point_sets],
        centers=display_xyz(centers).round(5).tolist(),
    )
    template = ROOT / "tools/robotwin_omega_stream_viewer.html"
    page = template.read_text().replace("__TITLE__", f"{out.parent.name} / {out.name}")
    page = page.replace("__DATA__", json.dumps(payload, separators=(",", ":")))
    (out / "viewer.html").write_text(page)


def infer_stream(model, paths, recorded, frames, out, args):
    from vggt_omega.utils.load_fn import load_and_preprocess_images
    from vggt_omega.utils.pose_enc import encoding_to_camera

    torch.cuda.reset_peak_memory_stats()
    images = load_and_preprocess_images([str(p) for p in paths], image_resolution=args.resolution).cuda()
    torch.cuda.synchronize()
    start = time.perf_counter()
    with torch.inference_mode():
        prediction = model(images)
        e, k = encoding_to_camera(prediction["pose_enc"], images.shape[-2:])
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    arrays = {
        name: prediction[name][0].float().cpu().numpy() for name in ["depth", "depth_conf", "camera_and_register_tokens"]
    }
    depth = arrays["depth"].reshape(len(frames), *images.shape[-2:])
    conf = arrays["depth_conf"].reshape(depth.shape)
    e, k = e[0].float().cpu().numpy(), k[0].float().cpu().numpy()
    rgb = images.float().permute(0, 2, 3, 1).cpu().numpy()
    for value in [depth, conf, e, k, rgb, arrays["camera_and_register_tokens"]]:
        if not np.isfinite(value).all():
            raise RuntimeError(f"Non-finite outputs for {out}")
    if not (depth > 0).all():
        raise RuntimeError("Non-positive depth")
    cam_points, world_points = unproject(depth, k, e)
    # Check world -> camera -> image round-trip independently of the exporter.
    reproj = np.einsum("nhwj,nij->nhwi", world_points, e[:, :, :3]) + e[:, None, None, :, 3]
    np.testing.assert_allclose(reproj, cam_points, atol=2e-4, rtol=2e-4)
    np.savez_compressed(
        out / "predictions.npz",
        depth=depth,
        confidence=conf,
        extrinsics=e,
        intrinsics=k,
        images=rgb,
        frames=frames,
        recorded_extrinsics=recorded,
        camera_and_register_tokens=arrays["camera_and_register_tokens"],
    )
    valid = np.isfinite(world_points).all(-1) & (depth > 0)
    filtered = valid & (conf >= np.percentile(conf, 10, axis=(1, 2))[:, None, None])
    rng = np.random.default_rng(42)
    exports = {}
    for name, mask, limit, p, c in [
        ("all_frames_unfiltered", valid, 150000, world_points, rgb),
        ("all_frames_conf90", filtered, 150000, world_points, rgb),
        ("latest_frame_conf90", filtered[-1], 60000, world_points[-1], rgb[-1]),
    ]:
        pts, colors = sample(p, c, mask, limit, rng)
        trimesh.points.PointCloud(pts, colors=(colors * 255).astype(np.uint8)).export(out / f"{name}.ply")
        exports[name] = len(pts)
    # Build current-frame surface in camera coordinates, then place in local world.
    mesh_info = latest_mesh(cam_points[-1], rgb[-1], filtered[-1], out / "latest_frame_depth_mesh.ply")
    mesh = trimesh.load(out / "latest_frame_depth_mesh.ply", process=False)
    mesh.vertices = (mesh.vertices - e[-1, :, 3]) @ e[-1, :, :3]
    mesh.export(out / "latest_frame_depth_mesh.ply")
    point_sets = [sample(p, c, m, 6500, rng) for p, c, m in zip(world_points, rgb, filtered, strict=False)]
    all_set = (np.concatenate([p for p, _ in point_sets]), np.concatenate([c for _, c in point_sets]))
    latest_set = sample(world_points[-1], rgb[-1], filtered[-1], 30000, rng)
    centers = -np.einsum("nji,nj->ni", e[:, :, :3], e[:, :, 3])
    previews(out, frames, rgb, depth, conf, cam_points, [all_set, latest_set], centers)
    viewer(out, frames, point_sets, centers)
    stats = diagnostics(e, recorded, depth, conf, rgb)
    stats.update(
        inference_seconds=elapsed,
        input_shape=list(images.shape),
        peak_allocated_gpu_gib=torch.cuda.max_memory_allocated() / 2**30,
        exports=exports,
        mesh=mesh_info,
        finite_outputs=True,
        geometry_roundtrip_checked=True,
        retained_fraction_per_frame=filtered.mean(axis=(1, 2)).tolist(),
    )
    write_json(out / "metrics.json", stats)
    camera_preview(out, frames, stats)
    del prediction, images
    gc.collect()
    torch.cuda.empty_cache()
    return stats


def report(out, manifest):
    cases = list(manifest["cases"])
    for extra_root in manifest.get("additional_runs", []):
        extra_root = Path(extra_root)
        extra = json.loads((extra_root / "manifest.json").read_text())
        for original in extra["cases"]:
            case = dict(original)
            case["name"] = os.path.relpath(extra_root / case["name"], out)
            cases.append(case)
    lines = [
        "# Independent eight-frame camera reconstruction",
        "",
        (
            "Each camera history is reconstructed independently with shared frozen Omega weights. "
            "No cross-camera images or calibration enter inference. Coordinates and scale are local to each call."
        ),
        "",
        (
            "The superposed clouds are eight instantaneous reconstructions, not a static fused surface: "
            "moving hands and objects can leave multiple copies. Input depth normals and confidence are descriptive, "
            "not evidence of metric accuracy. There is no recorded depth ground truth."
        ),
        "",
        (
            "| Task | Camera | Frames | Max rotation error vs recorded (deg) | Camera displacement / median depth | Time"
            " (s) |"
        ),
        "|---|---|---|---:|---:|---:|",
    ]
    links = []
    fig, axes = plt.subplots(len(cases), 3, figsize=(18, 3 * len(cases) + 0.5), squeeze=False)
    for row, case in enumerate(cases):
        for i, cam in enumerate(CAMERAS):
            m = case["streams"][cam]
            relative = f"{case['name']}/{cam}"
            lines.append(
                f"| {case['task']} | {LABELS[i]} | {case['frames'][0]}–{case['frames'][-1]} | "  # noqa: RUF001 - intentional mathematical/visual labels
                f"{m['max_relative_rotation_error_deg']:.3f} | "
                f"{m['predicted_max_camera_displacement_over_median_depth']:.4f} | {m['inference_seconds']:.2f} |"
            )
            links.append(
                f'<article><h2>{case["task"]} / {LABELS[i]}</h2>'
                f'<a href="{relative}/viewer.html">Open interactive geometry</a> · '
                f'<a href="{relative}/temporal_preview.png">Eight-frame RGB/depth/confidence/normals</a> · '
                f'<a href="{relative}/camera_consistency.png">Camera motion checks</a> · '
                f'<a href="{relative}/metrics.json">Measurements</a>'
                f'<a href="{relative}/viewer.html"><img src="{relative}/geometry_preview.png"></a></article>'
            )
            axes[row, i].imshow(Image.open(out / relative / "geometry_preview.png"))
            axes[row, i].axis("off")
    fig.suptitle(
        "Independent camera streams: eight-frame superposition and latest-frame geometry\n"
        "Axes and arbitrary scales differ between cameras; no cross-camera alignment",
        fontsize=16,
    )
    fig.tight_layout()
    fig.savefig(out / "overview.png", dpi=130)
    plt.close(fig)
    lines += ["", "## Visualizations", "", "[Overview](overview.png) · [Interactive gallery](index.html)", ""]
    for case in cases:
        for cam in CAMERAS:
            p = f"{case['name']}/{cam}"
            lines.append(
                f'- {case["task"]} / {cam}: [3D viewer]({p}/viewer.html), '
                f"[temporal preview]({p}/temporal_preview.png), [metrics]({p}/metrics.json)."
            )
    heads = [c["streams"]["head_camera"] for c in cases]
    lines += [
        "",
        "## Observations",
        "",
        (
            "Head-camera maximum rotation drift ranges from"
            f" {min(s['max_relative_rotation_error_deg'] for s in heads):.3f} to"
            f" {max(s['max_relative_rotation_error_deg'] for s in heads):.3f} degrees. This is compatible with a stable"
            " local reconstruction, although it does not establish correct depth."
        ),
        "",
        (
            "Inspect the moving wrist streams with frame coloring enabled: stationary background patches and moving"
            " objects should be interpreted separately. All-frame superposition is not motion-compensated object fusion."
        ),
        "",
        (
            "The default first-closure chunks contain substantial right-wrist motion for rotate_qrcode and"
            " adjust_bottle, while the left wrist stays approximately stationary. Both wrists barely move in the default"
            " stack_blocks_three chunk. These examples do not cover every camera-motion regime."
        ),
        "",
        (
            "No matched single-frame or joint multi-camera baseline was run here, so these checks do not establish "
            "an advantage for action prediction or reconstruction accuracy."
        ),
    ]
    for case in cases:
        for cam, stats in case["streams"].items():
            if stats["max_relative_rotation_error_deg"] > 10:
                lines += [
                    "",
                    (
                        f"**Large camera-motion discrepancy:** {case['task']} / {cam}, "
                        f"frames {case['frames'][0]}–{case['frames'][-1]}. Recorded rotation from the first "  # noqa: RUF001 - intentional mathematical/visual labels
                        f"camera reaches {max(stats['recorded_rotation_from_first_deg']):.2f} degrees, "
                        f"while Omega predicts only {max(stats['predicted_rotation_from_first_deg']):.2f} degrees. "
                        f"Maximum relative rotation error is {stats['max_relative_rotation_error_deg']:.2f} degrees. "
                        "The superposed background is visibly spread out. This is evidence that temporal overlap "
                        "alone does not ensure camera/geometry consistency on these inputs. Recorded poses "
                        "have not been independently replay-verified."
                    ),
                ]
    lines += [
        "",
        "## Reproduce",
        "",
        "```bash",
        manifest["reproduce_command"],
        "```",
        "",
        (
            "Frame stride is in recorded observations, not necessarily simulator control ticks. "
            "To test a wider temporal baseline, rerun with --stride 4. "
            "Recorded camera-consistency checks use the HDF5 extrinsics without simulator replay verification."
        ),
    ]
    for extra_root in manifest.get("additional_runs", []):
        extra = json.loads((Path(extra_root) / "manifest.json").read_text())
        lines += ["", "Additional left-wrist-motion chunk:", "", "```bash", reproduce_command(extra), "```"]
    (out / "README.md").write_text("\n".join(lines) + "\n")
    (out / "index.html").write_text(
        '<!doctype html><meta charset="utf-8"><title>Camera stream reconstructions</title>'
        "<style>body{font:16px system-ui;max-width:1400px;margin:auto;padding:24px;background:#f5f7fa}"
        "img{width:100%}article{background:white;padding:18px;margin:18px 0}a{color:#1763a6}</style>"
        "<h1>Independent eight-frame camera reconstructions</h1><p>Each stream has its own coordinate frame "
        "and arbitrary scale. No cross-camera fusion. Superposed dynamic objects may appear duplicated.</p>"
        + "".join(links)
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("/export/home/haoliang/projects/RoboTwin/data"))
    parser.add_argument("--omega-root", type=Path, default=DEFAULT_OMEGA)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_OMEGA / "ckpt/vggt_omega_1b_512.pt")
    parser.add_argument("--tasks", nargs="+", default=["rotate_qrcode", "adjust_bottle", "stack_blocks_three"])
    parser.add_argument("--episode", type=int, default=7)
    parser.add_argument("--chunk-length", type=int, default=8)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--end-frame", type=int)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument(
        "--render-only",
        action="store_true",
        help="Refresh visualizations from existing predictions without GPU inference",
    )
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.chunk_length < 2 or args.stride < 1:
        parser.error("chunk-length must be >= 2 and stride >= 1")
    args.out = args.out.resolve()
    if args.render_only:
        manifest = json.loads((args.out / "manifest.json").read_text())
        for case in manifest["cases"]:
            for cam in CAMERAS:
                out = args.out / case["name"] / cam
                a = np.load(out / "predictions.npz")
                depth, conf, rgb, e = a["depth"], a["confidence"], a["images"], a["extrinsics"]
                cam_points, world = unproject(depth, a["intrinsics"], e)
                mask = conf >= np.percentile(conf, 10, axis=(1, 2))[:, None, None]
                rng = np.random.default_rng(42)
                sets = [sample(p, c, m, 6500, rng) for p, c, m in zip(world, rgb, mask, strict=False)]
                merged = (np.concatenate([p for p, _ in sets]), np.concatenate([c for _, c in sets]))
                latest = sample(world[-1], rgb[-1], mask[-1], 30000, rng)
                centers = -np.einsum("nji,nj->ni", e[:, :, :3], e[:, :, 3])
                previews(out, case["frames"], rgb, depth, conf, cam_points, [merged, latest], centers)
                viewer(out, case["frames"], sets, centers)
                camera_preview(out, case["frames"], case["streams"][cam])
        manifest["rendering_source_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        manifest["reproduce_command"] = reproduce_command(manifest)
        shutil.copy2(__file__, args.out / "rendering_tool.py")
        shutil.copy2(ROOT / "tools/robotwin_omega_stream_viewer.html", args.out / "viewer_template.html")
        report(args.out, manifest)
        write_json(args.out / "manifest.json", manifest)
        print(f"RENDERED {args.out}", flush=True)
        return
    args.out.mkdir(parents=True, exist_ok=False)
    shutil.copy2(__file__, args.out / Path(__file__).name)
    shutil.copy2(ROOT / "tools/robotwin_omega_stream_viewer.html", args.out / "viewer_template.html")
    sys.path.insert(0, str(args.omega_root))
    from vggt_omega.models import VGGTOmega

    torch.cuda.set_device(args.gpu)
    torch.manual_seed(42)
    manifest = dict(
        status="running",
        started_unix=time.time(),
        args={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        cases=[],
        python=sys.executable,
        torch=torch.__version__,
        gpu=torch.cuda.get_device_name(),
        inference_contract=(
            "One camera, eight past observations, independent call; frozen shared weights; no calibration input."
        ),
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        reproduce_command=(
            f"{sys.executable} tools/robotwin_omega_camera_streams.py --out results/omega_camera_streams/new_run"
        ),
    )
    manifest["reproduce_command"] = reproduce_command(manifest)
    write_json(args.out / "manifest.json", manifest)
    try:
        state = torch.load(args.checkpoint, map_location="cpu", mmap=True, weights_only=True)
        model = VGGTOmega(enable_alignment=any(k.startswith("text_alignment_head.") for k in state)).eval()
        model.load_state_dict(state, strict=True)
        del state
        model.cuda().requires_grad_(False)
        manifest["checkpoint_loading"] = "strict=True; no missing or unexpected parameters"
        for task in args.tasks:
            source = args.data_root / task / "demo_clean/data" / f"episode{args.episode}.hdf5"
            with h5py.File(source) as f:
                frames, rule = select_frames(f, args.chunk_length, args.stride, args.end_frame)
                name = f"{task}_ep{args.episode:03d}_f{frames[0]:04d}_{frames[-1]:04d}"
                case = dict(
                    name=name, task=task, source=str(source), frames=frames, selection=rule, streams={}, input_sha256={}
                )
                manifest["cases"].append(case)
                for cam in CAMERAS:
                    out = args.out / name / cam
                    (out / "inputs").mkdir(parents=True)
                    paths = []
                    for frame in frames:
                        path = out / "inputs" / f"frame_{frame:05d}.png"
                        Image.open(io.BytesIO(bytes(f[f"observation/{cam}/rgb"][frame]))).convert("RGB").save(path)
                        paths.append(path)
                    case["input_sha256"][cam] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
                    recorded = f[f"observation/{cam}/extrinsic_cv"][frames]
                    print(f"INFER {task} {cam} frames={frames}", flush=True)
                    case["streams"][cam] = infer_stream(model, paths, recorded, frames, out, args)
                    write_json(args.out / "manifest.json", manifest)
                    print(f'COMPLETE {task} {cam}: {case["streams"][cam]["inference_seconds"]:.2f}s', flush=True)
        report(args.out, manifest)
        manifest.update(status="completed", completed_unix=time.time())
        write_json(args.out / "manifest.json", manifest)
    except Exception as exc:
        manifest.update(status="failed", error=repr(exc))
        write_json(args.out / "manifest.json", manifest)
        raise
    print(f"COMPLETED {args.out}", flush=True)


if __name__ == "__main__":
    main()
