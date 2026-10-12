"""Render the same raw depth mesh procedure for VGGT and Omega, without confidence masks."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from render_behavior_reconstruction import triangulate_depth, write_json


def main():
    from surflo.rendering.mesh import MeshRasterizer, ScalableMeshRenderer
    from surflo.structures.cameras import is_in_view_frustum
    from surflo.structures.mesh import Meshes
    from surflo.utils.guided_result import build_refined_cameras_from_guided_result
    from surflo.utils.io import write_mesh_ply

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    out = args.root / "raw_depth_meshes"
    out.mkdir(exist_ok=True)
    cases = [("vggt", args.root / "geometry_batch.pt")]
    cases += [
        (f"omega{res}", args.root / f"omega_{res}/geometry_batch.pt")
        for res in [256, 512]
        if (args.root / f"omega_{res}/geometry_batch.pt").exists()
    ]
    frames, records = {}, []
    torch.cuda.set_device(0)
    with torch.no_grad():
        for label, path in cases:
            batch = torch.load(path, map_location="cuda", weights_only=True)
            p = batch["vggt_world_points"][0].float().cpu().numpy()
            d = batch["vggt_depth"][0, ..., 0].float().cpu().numpy()
            c = batch["rgb_images"][0].permute(0, 2, 3, 1).float().cpu().numpy().clip(0, 1)
            valid = np.isfinite(p).all(-1) & np.isfinite(d) & (d > 0)
            v, f, used = triangulate_depth(p, d, valid)
            colors = c.reshape(-1, 3)[used]
            write_mesh_ply(
                str(out / f"{label}_depth_mesh.ply"), v, f, vertex_colors_uint8=(colors * 255).round().astype(np.uint8)
            )
            assert np.isfinite(v).all() and (f >= 0).all() and (f < len(v)).all()
            cameras = build_refined_cameras_from_guided_result(
                batch, 0, {}, apply_camera_correction=False, data_device="cuda"
            )
            mesh = Meshes(
                torch.tensor(v, device="cuda", dtype=torch.float32),
                torch.tensor(f, device="cuda", dtype=torch.int32),
                torch.tensor(colors, device="cuda", dtype=torch.float32),
            )
            renderer = ScalableMeshRenderer(rasterizer=MeshRasterizer(cameras=cameras, use_opengl=False))
            scores = []
            for index, camera in enumerate(cameras):
                keep = is_in_view_frustum(mesh.verts, camera)[mesh.faces.long()].any(-1)
                visible = Meshes(mesh.verts, mesh.faces[keep], mesh.verts_colors)
                result = renderer(visible, cam_idx=index, return_depth=True, return_normals=True, use_antialiasing=False)
                rgb = result["rgb"].squeeze(0).clamp(0, 1)
                normals = result["normals"].squeeze(0)
                light = torch.tensor([0.3, -0.6, -0.7], device="cuda")
                light /= light.norm()
                gray = (0.2 + 0.8 * (normals * light).sum(-1).abs()).clamp(0, 1)
                gray = torch.where(result["depth"].squeeze() > 0, gray, torch.zeros_like(gray))
                frames[f"{label}_{index}"] = rgb.cpu().numpy()
                frames[f"{label}_shaded_{index}"] = gray.cpu().numpy()
                target = batch["rgb_images"][0, index].float().permute(1, 2, 0)
                scores.append(
                    dict(
                        view=index,
                        rgb_mae=float((rgb - target).abs().mean()),
                        coverage=float((result["depth"] > 0).float().mean()),
                    )
                )
            records.append(dict(model=label, vertices=len(v), faces=len(f), scores=scores))
            print(json.dumps(records[-1]), flush=True)
    np.savez_compressed(out / "render_frames.npz", **frames)
    write_json(
        out / "metadata.json",
        dict(
            records=records,
            confidence_filter="none",
            edge_cutoff="5% of local positive depth; same procedure for all depth models",
            scope="Input cameras, no ground-truth metric geometry. Raw PLY is preserved in each model frame.",
        ),
    )


if __name__ == "__main__":
    main()
