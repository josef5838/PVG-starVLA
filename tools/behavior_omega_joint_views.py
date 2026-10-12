"""Matched BEHAVIOR ablation: all three eight-image histories in one Omega call.

Reuses audited PNGs and recorded poses from behavior_omega_camera_streams.py.
Recorded poses are diagnostics only. Ordering is time-major: head, left, right.
"""

import argparse
import gc
import hashlib
import html
import json
import os
import shlex
import shutil
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import behavior_omega_camera_streams as behavior
import matplotlib.pyplot as plt
import numpy as np
import robotwin_omega_camera_streams as streams
import torch
import trimesh

CAMERAS = behavior.CAMERAS
LABELS = behavior.LABELS
DEFAULT_BASELINE = streams.ROOT / "results/omega_camera_streams/20261009_behavior_radio_chunk8_stride1"


def joint_viewer(out, labels, point_sets, centers):
    streams.viewer(out, labels, point_sets, centers)
    page = (out / "viewer.html").read_text()
    page = page.replace("eight-image", "24-image").replace("eight images", "24 images")
    page = page.replace("eight-frame superposition", "joint 24-image output")
    page = page.replace('max="7" value="7"', 'max="23" value="23"')
    page = page.replace("One image's output from the joint call", "One camera / timestamp output from the joint call")
    page = page.replace("<label>Frame <input", "<label>Image <input")
    page = page.replace(
        "<label>Image <input",
        '<label>Camera <select id="camera"><option value="all">All cameras</option>'
        '<option value="0">Head</option><option value="1">Left wrist</option>'
        '<option value="2">Right wrist</option></select></label> <label>Image <input',
    )
    page = page.replace("const pts=[];", "const selected=document.getElementById('camera').value;\n const pts=[];")
    page = page.replace(
        "for(const i of ids)for(let j=0;", "for(const i of ids.filter(i=>selected==='all'||i%3===+selected))for(let j=0;"
    )
    page = page.replace(
        "for(const el of [mode,frame,timecolor,cams])",
        "for(const el of [mode,frame,timecolor,cams,document.getElementById('camera')])",
    )
    page = page.replace("Camera trajectory", "Camera centers (connected in input order)")
    page = page.replace("Color by frame", "Color by input image")
    (out / "viewer.html").write_text(page)


def camera_previews(out, case, camera, rgb, depth, conf, camera_points, world_points, mask, centers, rng):
    lo, hi = np.percentile(depth, [2, 98])
    clo, chi = np.percentile(conf, [2, 98])
    normals = np.cross(np.gradient(camera_points, axis=1), np.gradient(camera_points, axis=2))
    normals /= np.maximum(np.linalg.norm(normals, axis=-1, keepdims=True), 1e-8)
    fig, axes = plt.subplots(4, 8, figsize=(24, 9))
    for i, frame in enumerate(case["frames"]):
        values = [
            (rgb[i], None, None, None),
            (depth[i], "viridis", lo, hi),
            (conf[i], "magma", clo, chi),
            ((normals[i] + 1) / 2, None, None, None),
        ]
        for row, (value, cmap, vmin, vmax) in enumerate(values):
            axes[row, i].imshow(value, cmap=cmap, vmin=vmin, vmax=vmax)
            axes[row, i].set_xticks([])
            axes[row, i].set_yticks([])
            if i == 0:
                axes[row, i].set_ylabel(["RGB", "Depth", "Confidence", "Depth normals"][row])
        axes[0, i].set_title(f"Frame {frame}")
    fig.suptitle(
        f'{case["name"]} / {camera}: eight outputs from ONE joint 24-image call\n'
        "All three cameras provided context; shared color ranges within this camera; arbitrary depth scale"
    )
    fig.tight_layout()
    fig.savefig(out / "temporal_preview.png", dpi=110)
    plt.close(fig)
    sets = [
        streams.sample(world_points, rgb, mask, 52000, rng),
        streams.sample(world_points[-1], rgb[-1], mask[-1], 30000, rng),
    ]
    limits = np.percentile(streams.display_xyz(sets[0][0]), [1, 99], axis=0)
    midpoint = limits.mean(0)
    extent = np.max(limits[1] - limits[0]) * 0.55
    bounds = np.stack([midpoint - extent, midpoint + extent], -1)
    fig = plt.figure(figsize=(12, 5), facecolor="#101723")
    titles = ["Eight outputs of this camera (joint 24-image call)", "Last output of this camera (joint 24-image call)"]
    for i, (points, colors) in enumerate(sets):
        streams.plot_cloud(fig.add_subplot(1, 2, i + 1, projection="3d"), points, colors, centers, titles[i], bounds)
    fig.suptitle(
        f'{case["name"]} / {camera}\nCoordinates shared across all 24 outputs; each panel shows this camera only',
        color="white",
    )
    fig.tight_layout()
    fig.savefig(out / "geometry_preview.png", dpi=150)
    plt.close(fig)


def cross_camera_metrics(e, recorded):
    pair_errors = {}
    for a, b in [(0, 1), (0, 2), (1, 2)]:
        ia, ib = np.arange(a, 24, 3), np.arange(b, 24, 3)
        pred = e[ib, :, :3] @ e[ia, :, :3].transpose(0, 2, 1)
        truth = recorded[ib, :, :3] @ recorded[ia, :, :3].transpose(0, 2, 1)
        error = streams.angle(pred @ truth.transpose(0, 2, 1))
        pair_errors[f"{LABELS[a]} to {LABELS[b]}"] = dict(
            errors_deg=error.tolist(),
            mean_rotation_error_deg=float(error.mean()),
            max_rotation_error_deg=float(error.max()),
        )
    _, pc = streams.relative_cameras(e)
    _, rc = streams.relative_cameras(recorded)
    scale = float(np.sum(pc * rc) / max(np.sum(pc**2), 1e-12))
    return dict(
        synchronized_relative_rotations=pair_errors,
        single_scale_fit=dict(
            signed_scale_m_per_model_unit=scale,
            rms_center_residual_cm=(
                float(np.sqrt(np.mean(np.sum((scale * pc - rc) ** 2, axis=-1))) * 100) if scale > 0 else None
            ),
            scope="One scale across all 24 cameras; orientation anchored to first head camera; no ICP.",
        ),
    )


def infer(model, baseline, case, out, resolution):
    from vggt_omega.utils.load_fn import load_and_preprocess_images
    from vggt_omega.utils.pose_enc import encoding_to_camera

    out.mkdir(parents=True)
    pose_arrays = {
        c: np.load(baseline / case["name"] / c / "recorded_camera_poses.npz")["base_to_camera_cv"] for c in CAMERAS
    }
    inputs = []
    for _t, frame in enumerate(case["frames"]):
        for cam in CAMERAS:
            source = baseline / case["name"] / cam / "inputs" / f"frame_{frame:05d}.png"
            digest = hashlib.sha256(source.read_bytes()).hexdigest()
            assert digest == case["input_records"][cam]["sha256"][source.name]
            inputs.append(dict(index=len(inputs), camera=cam, frame=frame, path=str(source), sha256=digest))
    streams.write_json(out / "inputs.json", inputs)
    recorded = np.stack([pose_arrays[c][t] for t in range(8) for c in CAMERAS])
    images = load_and_preprocess_images([x["path"] for x in inputs], image_resolution=resolution).cuda()
    assert list(images.shape) == [24, 3, 512, 512]
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    started = time.perf_counter()
    with torch.inference_mode():
        prediction = model(images)  # Exactly one forward call for all 24 images.
        e, k = encoding_to_camera(prediction["pose_enc"], images.shape[-2:])
    torch.cuda.synchronize()
    duration = time.perf_counter() - started
    depth = prediction["depth"][0].float().cpu().numpy().reshape(24, 512, 512)
    conf = prediction["depth_conf"][0].float().cpu().numpy().reshape(depth.shape)
    tokens = prediction["camera_and_register_tokens"][0].float().cpu().numpy()
    e = e[0].float().cpu().numpy()
    k = k[0].float().cpu().numpy()
    rgb = images.float().permute(0, 2, 3, 1).cpu().numpy()
    for a in [depth, conf, tokens, e, k, rgb]:
        assert np.isfinite(a).all()
    assert (depth > 0).all()
    camera_points, world_points = streams.unproject(depth, k, e)
    reprojected = np.einsum("nhwj,nij->nhwi", world_points, e[:, :, :3]) + e[:, None, None, :, 3]
    np.testing.assert_allclose(reprojected, camera_points, atol=2e-4, rtol=2e-4)
    np.savez_compressed(
        out / "predictions.npz",
        depth=depth,
        confidence=conf,
        extrinsics=e,
        intrinsics=k,
        images=rgb,
        camera_and_register_tokens=tokens,
        recorded_extrinsics=recorded,
        frames=[x["frame"] for x in inputs],
        camera_indices=np.tile(np.arange(3), 8),
    )
    mask = conf >= np.percentile(conf, 10, axis=(1, 2))[:, None, None]
    rng = np.random.default_rng(42)
    for name, keep in [("all_images_unfiltered", np.ones_like(mask)), ("all_images_conf90", mask)]:
        points, colors = streams.sample(world_points, rgb, keep, 150000, rng)
        trimesh.points.PointCloud(points, colors=(colors * 255).astype(np.uint8)).export(out / f"{name}.ply")
    centers = -np.einsum("nji,nj->ni", e[:, :, :3], e[:, :, 3])
    point_sets = [streams.sample(p, c, m, 6500, rng) for p, c, m in zip(world_points, rgb, mask, strict=False)]
    joint_viewer(out, [f'{LABELS[x["index"]%3]} / frame {x["frame"]}' for x in inputs], point_sets, centers)
    points = np.concatenate([p for p, _ in point_sets])
    colors = np.concatenate([c for _, c in point_sets])
    limits = np.percentile(streams.display_xyz(points), [1, 99], axis=0)
    midpoint = limits.mean(0)
    extent = (limits[1] - limits[0]).max() * 0.55
    bounds = np.stack([midpoint - extent, midpoint + extent], -1)
    fig = plt.figure(figsize=(10, 7), facecolor="#101723")
    streams.plot_cloud(
        fig.add_subplot(111, projection="3d"),
        points,
        colors,
        centers,
        "All 24 outputs from one call; predicted shared coordinates",
        bounds,
    )
    fig.tight_layout()
    fig.savefig(out / "joint_geometry.png", dpi=150)
    plt.close(fig)
    metrics = dict(
        input_shape=list(images.shape),
        forward_call_count=1,
        inference_seconds=duration,
        peak_allocated_gpu_gib=torch.cuda.max_memory_allocated() / 2**30,
        finite_outputs=True,
        geometry_roundtrip_checked=True,
        streams={},
        cross_camera=cross_camera_metrics(e, recorded),
    )
    for i, cam in enumerate(CAMERAS):
        ids = np.arange(i, 24, 3)
        sub = out / cam
        sub.mkdir()
        stats = streams.diagnostics(e[ids], recorded[ids], depth[ids], conf[ids], rgb[ids])
        stats["output_indices"] = ids.tolist()
        stats["interpretation"] = (
            "Eight camera outputs conditioned on all 24 images. Recorded poses are robot-base-relative; approximately"
            " stationary base assumed."
        )
        metrics["streams"][cam] = stats
        streams.write_json(sub / "metrics.json", stats)
        camera_previews(
            sub,
            case,
            cam,
            rgb[ids],
            depth[ids],
            conf[ids],
            camera_points[ids],
            world_points[ids],
            mask[ids],
            centers[ids],
            rng,
        )
        streams.camera_preview(sub, case["frames"], stats)
    streams.write_json(out / "metrics.json", metrics)
    del prediction, images, world_points, camera_points
    gc.collect()
    torch.cuda.empty_cache()
    return metrics


def report(out, manifest, baseline):
    rows = []
    articles = []
    fig, axes = plt.subplots(4, 3, figsize=(18, 12.5))
    motion_fig, motion_axes = plt.subplots(4, 3, figsize=(15, 11.5))
    for row, case in enumerate(manifest["cases"]):
        source = next(c for c in baseline["cases"] if c["name"] == case["name"])
        articles.append(
            f'<h2>{case["label"]}: frames {source["frames"][0]}–{source["frames"][-1]}</h2>'  # noqa: RUF001 - intentional mathematical/visual labels
            f'<a href="{case["name"]}/viewer.html">Orbit all 24 outputs / select a camera or image</a>'
            f'<img src="{case["name"]}/joint_geometry.png">'
        )
        for i, cam in enumerate(CAMERAS):
            before = source["streams"][cam]
            after = case["metrics"]["streams"][cam]
            ax = motion_axes[row, i]
            ax.plot(case["frames"], before["relative_rotation_error_deg"], marker=".", label="Independent 8")
            ax.plot(case["frames"], after["relative_rotation_error_deg"], marker=".", label="Joint 24")
            ax.set_title(f'{case["label"]} / {LABELS[i]}', fontsize=9)
            ax.set_xlabel("Episode frame")
            ax.set_ylabel("Relative rotation discrepancy (deg)")
            ax.grid(alpha=0.2)
            ax.legend(fontsize=8)
            rows.append(
                f'| {case["label"]} | {LABELS[i]} | {max(before["recorded_rotation_from_first_deg"]):.2f} | '
                f'{before["max_relative_rotation_error_deg"]:.2f} | {after["max_relative_rotation_error_deg"]:.2f} |'
            )
            path = f'{case["name"]}/{cam}'
            # Side-by-side panels use their own bounds and gauges; never overlay independent clouds.
            a = plt.imread(Path(manifest["baseline"]) / path / "geometry_preview.png")
            b = plt.imread(out / path / "geometry_preview.png")
            axes[row, i].imshow(np.concatenate([a, b], axis=0))
            axes[row, i].axis("off")
            axes[row, i].set_title(f'{case["label"]} / {LABELS[i]}\nTop: independent 8; bottom: joint 24', fontsize=9)
            articles.append(
                f'<h3>{LABELS[i]}</h3><p><a href="{path}/temporal_preview.png">Joint RGB/depth timeline</a> · <a'
                f' href="{path}/camera_consistency.png">Joint temporal pose plot</a> · <a'
                f' href="{html.escape(os.path.relpath(Path(manifest["baseline"])/path/"viewer.html",out))}">Independent'
                " eight-image viewer</a></p><p>Maximum temporal rotation discrepancy: independent"
                f' {before["max_relative_rotation_error_deg"]:.2f}°; joint'
                f' {after["max_relative_rotation_error_deg"]:.2f}°.</p>'
            )
    fig.suptitle(
        "BEHAVIOR radio: matched independent 8-image vs joint 24-image calls\n"
        "Each plot uses its own display bounds; appearance alone is not a geometry accuracy measurement",
        fontsize=13,
    )
    fig.tight_layout()
    fig.savefig(out / "comparison_overview.png", dpi=130)
    plt.close(fig)
    motion_fig.suptitle(
        "Matched per-camera rotation discrepancies: independent 8 vs joint 24\n"
        "Each history anchored to its own first camera pose; smaller is better",
        fontsize=13,
    )
    motion_fig.tight_layout()
    motion_fig.savefig(out / "rotation_error_comparison.png", dpi=130)
    plt.close(motion_fig)
    text = [
        "# BEHAVIOR radio: joint 24-image ablation",
        "",
        (
            "Same four episode-197 chunks, exact input PNGs, preprocessing, frozen Omega512 checkpoint, and seed as the"
            " independent-camera experiment. One model(images) call per chunk, input [24,3,512,512], internally batch"
            " [1,24,3,512,512]. Ordering: head/left/right at the first timestamp, then head/left/right at the next"
            " timestamp, and so on. Only RGB is supplied; no camera identities, timestamps, calibration, state, actions,"
            " or depth are supplied as model arguments."
        ),
        "",
        (
            "[Interactive gallery](index.html) · [Comparison overview](comparison_overview.png) · "
            "[Rotation error comparison](rotation_error_comparison.png)"
        ),
        "",
        (
            "The joint call has mixed effects on temporal camera accuracy in these selected chunks. "
            "In frames 934–941, the left-wrist maximum rotation discrepancy improves from 11.84 to 6.27 degrees, "  # noqa: RUF001 - intentional mathematical/visual labels
            "while the right wrist worsens from 21.17 to 34.80 degrees. The cross-camera comparisons also "
            "show substantial discrepancies. These results do not establish a reliable shared reconstruction or "
            "an action-prediction advantage for either strategy."
        ),
        "",
        (
            "Rotation discrepancy below is evaluated within each camera history, relative to its own first pose. This"
            " removes coordinate-gauge differences and makes the independent/joint comparison fair. Recorded motion is"
            " robot-base-relative; the approximately stationary-base assumption and input selection limitations from the"
            " baseline still apply."
        ),
        "",
        (
            "| Chunk | Camera | Recorded max rotation (deg) | Independent 8: max rotation discrepancy (deg) | Joint 24:"
            " max rotation discrepancy (deg) |"
        ),
        "|---|---|---:|---:|---:|",
        *rows,
        "",
        "## Cross-camera consistency",
        "",
        (
            "The [Omega paper, Appendix A.1](https://arxiv.org/html/2605.15195v1#A1) describes training targets"
            " expressed in the first camera frame, with depth and translation scaled by average point distance. Separate"
            " calls therefore predict separate first-camera-relative, nonmetric reconstructions. A shared origin,"
            " orientation, and scale must be established before combining their explicit point coordinates. This"
            " coordinate freedom is separate from incorrect relative poses within a reconstruction."
        ),
        "",
        (
            "A joint call predicts a common coordinate system, but this does not guarantee correct registration between"
            " cameras. The metrics below compare relative camera rotations at synchronized timestamps. The recorded"
            " relative rotations do not require a stationary base at those timestamps. A separate single-scale center"
            " fit across all 24 outputs is saved in each metrics.json; it is diagnostic only and was not applied to"
            " exported geometry."
        ),
        "",
        "| Chunk | Camera pair | Joint mean rotation discrepancy (deg) | Joint max rotation discrepancy (deg) |",
        "|---|---|---:|---:|",
    ]
    for case in manifest["cases"]:
        for pair, s in case["metrics"]["cross_camera"]["synchronized_relative_rotations"].items():
            text.append(
                f'| {case["label"]} | {pair} | {s["mean_rotation_error_deg"]:.2f} | {s["max_rotation_error_deg"]:.2f} |'
            )
    text += [
        "",
        (
            "All PLY/viewer geometry uses raw predicted cameras and depth with arbitrary units; no recorded-pose"
            " alignment or ICP. Bottom 10% confidence per image is removed in filtered exports. Per-camera panels are"
            " slices of the same 24-image call. No depth ground-truth evaluation or action prediction was performed."
        ),
        "",
        "## Reproduce",
        "",
        "```bash",
        manifest["reproduce_command"],
        "```",
    ]
    (out / "README.md").write_text("\n".join(text) + "\n")
    (out / "index.html").write_text(
        '<!doctype html><meta charset="utf-8"><title>Joint 24-image BEHAVIOR ablation</title><style>body{font:16px'
        " system-ui;max-width:1400px;margin:auto;padding:24px;background:#f5f7fa}img{width:100%}a{color:#1763a6}</style><h1>BEHAVIOR"  # noqa: E501 - preserve embedded commands or diagnostic text
        " radio: all 24 images in one Omega call</h1><p>Exact matched inputs; one call per chunk. All outputs share"
        " predicted coordinates. Per-camera slices are conditioned on all three histories.</p><p><a"
        ' href="README.md">Full measurements and experiment scope</a> · <a href="comparison_overview.png">Independent /'
        ' joint geometry comparison</a></p><p><a href="rotation_error_comparison.png">Independent / joint rotation error'
        " comparison</a></p>" + "".join(articles)
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--omega-root", type=Path, default=streams.DEFAULT_OMEGA)
    parser.add_argument("--checkpoint", type=Path, default=streams.DEFAULT_OMEGA / "ckpt/vggt_omega_1b_512.pt")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out = args.out.resolve()
    args.baseline = args.baseline.resolve()
    baseline = json.loads((args.baseline / "manifest.json").read_text())
    assert baseline["status"] == "completed" and len(baseline["cases"]) == 4
    assert str(args.checkpoint) == baseline["args"]["checkpoint"] and baseline["args"]["resolution"] == 512
    args.out.mkdir(parents=True, exist_ok=False)
    sources = [
        Path(__file__),
        Path(streams.__file__),
        Path(behavior.__file__),
        streams.ROOT / "tools/robotwin_omega_stream_viewer.html",
    ]
    for p in sources:
        shutil.copy2(p, args.out / p.name)
    provenance = args.baseline / "checkpoint_provenance.json"
    if provenance.exists():
        shutil.copy2(provenance, args.out / provenance.name)
    manifest = dict(
        status="running",
        started_unix=time.time(),
        baseline=str(args.baseline),
        cases=[],
        python=sys.executable,
        args={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        source_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
        baseline_manifest_sha256=hashlib.sha256((args.baseline / "manifest.json").read_bytes()).hexdigest(),
        reproduce_command=shlex.join(
            [
                sys.executable,
                "tools/behavior_omega_joint_views.py",
                "--baseline",
                str(args.baseline),
                "--omega-root",
                str(args.omega_root),
                "--checkpoint",
                str(args.checkpoint),
                "--gpu",
                str(args.gpu),
                "--out",
                str(args.out) + "_rerun",
            ]
        ),
    )
    streams.write_json(args.out / "manifest.json", manifest)
    sys.path.insert(0, str(args.omega_root))
    from vggt_omega.models import VGGTOmega

    torch.cuda.set_device(args.gpu)
    torch.manual_seed(42)
    try:
        state = torch.load(args.checkpoint, map_location="cpu", mmap=True, weights_only=True)
        model = VGGTOmega(enable_alignment=any(k.startswith("text_alignment_head.") for k in state)).eval()
        model.load_state_dict(state, strict=True)
        del state
        model.cuda().requires_grad_(False)
        manifest["checkpoint_loading"] = "strict=True; no missing or unexpected parameters"
        for source in baseline["cases"]:
            print(f'INFER {source["name"]}: ONE call with all 24 images', flush=True)
            result = infer(model, args.baseline, source, args.out / source["name"], 512)
            manifest["cases"].append(
                dict(name=source["name"], label=source["label"], frames=source["frames"], metrics=result)
            )
            streams.write_json(args.out / "manifest.json", manifest)
            print(
                f'COMPLETE {source["name"]}: {result["inference_seconds"]:.3f}s; temporal errors '
                + ", ".join(
                    f'{LABELS[i]}={result["streams"][c]["max_relative_rotation_error_deg"]:.2f}deg'
                    for i, c in enumerate(CAMERAS)
                ),
                flush=True,
            )
        report(args.out, manifest, baseline)
        manifest.update(status="completed", completed_unix=time.time())
        streams.write_json(args.out / "manifest.json", manifest)
    except Exception as exc:
        manifest.update(status="failed", error=repr(exc))
        streams.write_json(args.out / "manifest.json", manifest)
        raise
    print(f"COMPLETED {args.out}", flush=True)


if __name__ == "__main__":
    main()
