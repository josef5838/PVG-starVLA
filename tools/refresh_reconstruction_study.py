"""Resume saved reconstruction stages and apply the audited mesh renderer uniformly.

Does not rerun Surflo reconstruction or change its geometry. Missing Omega
inferences are completed for cases interrupted by the original rendering error.
"""

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from finalize_reconstruction_comparisons import mesh_viewer, scene_metrics
from group_reconstruction_comparisons import group
from render_behavior_omega_comparison import infer, viewer
from render_behavior_omega_comparison import render as omega_render
from render_behavior_reconstruction import previews, render
from robotwin_reconstruction_study import OMEGA, REPO, SURFLO, release_cuda, stage_diagnostics, write


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def refresh(root, calibration=True):
    import torch
    import trimesh
    from surflo import save_mesh
    from surflo.extraction.texture.optimize import evaluate_mesh_colors_all_vertices
    from surflo.structures.mesh import Meshes
    from surflo.utils.guided_result import build_refined_cameras_from_guided_result

    previous = json.loads((root / "status.json").read_text()) if (root / "status.json").exists() else {}
    if previous.get("status") == "running":
        raise RuntimeError(f"Original worker still running: {root}")
    archive = root / "before_renderer_fix"
    archive.mkdir(exist_ok=True)
    for filename in [
        "status.json",
        "run.log",
        "comparison_metrics.json",
        "stage_metrics.json",
        "render_metadata.json",
        "comparison.png",
        "render_frames.npz",
        "surflo_mesh_colored.ply",
    ]:
        source = root / filename
        if source.exists() and not (archive / filename).exists():
            shutil.copy2(source, archive / filename)
    geometry_hashes = {
        name: digest(root / name) for name in ["geometry_batch.pt", "surflo_guided_result.pt", "surflo_mesh.ply"]
    }
    torch.cuda.set_device(0)
    status = dict(
        status="running",
        stage="renderer_refresh",
        started_unix=time.time(),
        gpu=os.environ.get("CUDA_VISIBLE_DEVICES"),
        previous_status=previous.get("status"),
    )
    write(root / "status.json", status)
    try:
        batch = torch.load(root / "geometry_batch.pt", map_location="cuda", weights_only=False)
        guided = torch.load(root / "surflo_guided_result.pt", map_location="cuda", weights_only=False)
        cameras = build_refined_cameras_from_guided_result(
            batch, 0, guided, apply_camera_correction=True, data_device="cuda"
        )
        source = trimesh.load(root / "surflo_mesh.ply", process=False)
        mesh = Meshes(
            torch.tensor(np.asarray(source.vertices), dtype=torch.float32, device="cuda"),
            torch.tensor(np.asarray(source.faces), dtype=torch.int32, device="cuda"),
        )
        mesh.verts_colors = evaluate_mesh_colors_all_vertices(mesh, cameras, batch["rgb_images"][0])
        save_mesh(mesh, root / "surflo_mesh_colored.ply")
        plain = torch.load(root / "surflo_plain_result.pt", map_location="cpu", weights_only=False)
        write(
            root / "surflo_metadata.json",
            dict(
                plain_points=len(plain["points"]),
                guided_points=len(guided["points"]),
                full_guided_gaussians=len(guided["aux_means"]),
                mesh_vertices=len(mesh.verts),
                mesh_faces=len(mesh.faces),
                guided_preset="default",
                expert_preset="default (DA3 depth + aligned normals)",
                num_query_points=100000,
                seed=42,
                ode_steps_plain=100,
            ),
        )
        del batch, guided, cameras, mesh, plain
        release_cuda()
        inputs = json.loads((root / "inputs.json").read_text())
        label = inputs.get("scene_label", "BEHAVIOR turning_on_radio, episode 197, frame 1491")
        args = SimpleNamespace(
            out=root, root=root, scene_label=label, episode=inputs.get("episode", 197), frame=inputs.get("frame", 1491)
        )
        render(args)
        previews(args)
        release_cuda()
        for resolution in [256, 512]:
            out = root / f"omega_{resolution}"
            out.mkdir(exist_ok=True)
            oargs = SimpleNamespace(
                root=root,
                resolution=resolution,
                checkpoint=OMEGA
                / "ckpt"
                / ("vggt_omega_1b_256_text.pt" if resolution == 256 else "vggt_omega_1b_512.pt"),
            )
            if not (out / "geometry_batch.pt").exists():
                status["stage"] = f"complete_missing_omega_{resolution}"
                write(root / "status.json", status)
                infer(oargs, out)
                release_cuda()
            omega_render(oargs, out)
            viewer(oargs, out)
            release_cuda()
        subprocess.run(
            [sys.executable, str(REPO / "tools/render_unfiltered_depth_meshes.py"), "--root", str(root)], check=True
        )
        group(root, root / "comparison.png", label, unfiltered=True)
        group(root, root / "confidence_filtered.png", label, unfiltered=False)
        stages = stage_diagnostics(root, use_calibration=calibration)
        release_cuda()
        validation = mesh_viewer(root, root / "mesh_viewer.html", label)
        write(root / "artifact_validation.json", validation)
        write(root / "comparison_metrics.json", dict(models=scene_metrics(root), surflo_stages=stages))
        for name, value in geometry_hashes.items():
            assert digest(root / name) == value, f"Geometry unexpectedly changed: {name}"
        write(
            root / "renderer_refresh.json",
            dict(
                geometry_sha256=geometry_hashes,
                renderer_sha256=digest(SURFLO / "surflo/rendering/mesh.py"),
                previous_status=previous,
                finished_unix=time.time(),
                scope=(
                    "Mesh rasterization and TSDF vertex colors refreshed; Surflo inference, full Gaussians and raw mesh"
                    " geometry unchanged."
                ),
            ),
        )
        status.update(
            status="completed",
            stage="completed",
            finished_unix=time.time(),
            duration_seconds=time.time() - status["started_unix"],
        )
        write(root / "status.json", status)
        write(
            root / "manifest.json",
            dict(
                status="completed",
                renderer_refresh=str(root / "renderer_refresh.json"),
                comparison=str(root / "comparison.png"),
                mesh_viewer=str(root / "mesh_viewer.html"),
            ),
        )
    except Exception as error:
        status.update(status="failed", error=repr(error), finished_unix=time.time())
        write(root / "status.json", status)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--worker", type=int, choices=[0, 1])
    parser.add_argument("--no-calibration", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    if args.worker is None:
        refresh(root, not args.no_calibration)
    else:
        for case in json.loads((root / "manifest.json").read_text())["cases"]:
            if case["worker"] == args.worker:
                print(f"Refreshing {case['name']}", flush=True)
                refresh(Path(case["root"]))


if __name__ == "__main__":
    main()
