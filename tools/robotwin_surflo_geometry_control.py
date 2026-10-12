"""Compare unposed reconstruction with a privileged simulator-geometry control.

The control changes spatial conditioning only. VGGT appearance and implicit
camera tokens remain estimated, so this is not a production calibrated model.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from surflo import Surflo, save_ply
from surflo.api import SceneState

parser = argparse.ArgumentParser()
parser.add_argument("--root", required=True)
parser.add_argument(
    "--source-only",
    action="store_true",
    help="Add random-source baseline to an existing comparison without repeating ODE solves",
)
args = parser.parse_args()
root = Path(args.root)
model = Surflo.from_checkpoint("/export/home/haoliang/projects/Surflo/checkpoints/surflo_v0.pt", device="cuda")
records = []
for ep in [51, 60, 38]:
    d = np.load(root / "robot_ground_truth" / f"episode_{ep:03d}.npz")
    images = torch.tensor(d["images"], device="cuda").permute(0, 3, 1, 2).float() / 255
    images = F.interpolate(images, size=(378, 518), mode="bicubic", align_corners=False).clamp(0, 1)
    with torch.no_grad():
        scene = model.encode(images, cull_radius=3.0)
        gtE = d["extrinsics"].astype(float)
        gtR = gtE[0, :, :3]
        gtT = gtE[0, :, 3]
        e = scene.extrinsics.cpu().numpy().astype(float)
        pR = e[0, :, :3]
        pT = e[0, :, 3]

        def centers_head(es):
            centers = -np.einsum("nij,nj->ni", es[:, :, :3].transpose(0, 2, 1), es[:, :, 3])
            return centers @ es[0, :, :3].T + es[0, :, 3]

        gC = centers_head(gtE)
        pC = centers_head(e)
        unconstrained_scale = float(np.sum(gC * pC) / np.sum(pC * pC))
        # An unposed initial scene can put the cameras on the wrong side of the
        # head. Use positive baseline-length scaling; report the signed fit too.
        scale = float(np.median(np.linalg.norm(gC[1:], axis=1) / np.maximum(np.linalg.norm(pC[1:], axis=1), 1e-8)))

        def to_robot(x, gtR=gtR, gtT=gtT, pR=pR, pT=pT, scale=scale):
            x = np.asarray(x)
            head = scale * (x @ pR.T + pT)
            return (head - gtT) @ gtR

        h, w = d["depth"].shape[1:]
        v, u = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
        xyz = []
        for z, k, ex in zip(d["depth"], d["intrinsics"], gtE, strict=False):
            ray = np.stack([u, v, np.ones_like(u)], -1) @ np.linalg.inv(k).T
            xyz.append((ray * z[..., None] - ex[:, 3]) @ ex[:, :3])
        gt = np.stack(xyz)
        valid = np.isfinite(gt).all(-1) & (d["depth"] > 0) & (d["depth"] < 3)
        depth = (
            F.interpolate(
                torch.tensor(d["depth"], device="cuda", dtype=torch.float32)[:, None], size=(378, 518), mode="nearest"
            )[:, 0]
            / scale
        )
        ks = d["intrinsics"].copy()
        ks[:, 0] *= 518 / w
        ks[:, 1] *= 378 / h
        relR = gtE[:, :, :3] @ gtR.T
        relT = gtE[:, :, 3] - np.einsum("nij,j->ni", relR, gtT)
        correctedE = np.concatenate([relR @ pR, (np.einsum("nij,j->ni", relR, pT) + relT / scale)[..., None]], -1)
        from surflo.utils.geometry import depths_to_points_parallel_batched

        k = torch.tensor(ks, device="cuda", dtype=torch.float32)[None]
        ex = torch.tensor(correctedE, device="cuda", dtype=torch.float32)[None]
        wp = depths_to_points_parallel_batched(k, ex, depth[None, ..., None], to_world=True)
        corrected = dict(scene.batch)
        corrected.update(vggt_intrinsics=k, vggt_extrinsics=ex, vggt_depth=depth[None, ..., None], vggt_world_points=wp)
        # Camera objects are unused by plain reconstruction; do not run guided with
        # this shallow copy, whose renderer cameras still describe the unposed scene.
        privileged = SceneState(model.model, corrected)
        outputs = {
            "ground_truth": gt[valid].astype(np.float32),
            "vggt": to_robot(scene.world_points.cpu().numpy().reshape(-1, 3)).astype(np.float32),
        }
        if args.source_only:
            cached = np.load(root / f"episode_{ep:03d}_geometry_comparison.npz")
            outputs = {key: cached[key] for key in cached.files}
        source = privileged.unlift_from_flow_space(privileged.sample_source(32768, seed=42))[:, :3]
        outputs["gt_spatial_source"] = to_robot(source.cpu().numpy()).astype(np.float32)
        for name, s in [] if args.source_only else [("plain", scene), ("gt_spatial_control", privileged)]:
            result = s.reconstruct(mode="plain", num_steps=100, num_query_points=32768, seed=42)
            outputs[name] = to_robot(result["points"].cpu().numpy()).astype(np.float32)
            save_ply(result, root / f"episode_{ep:03d}_{name}.ply")
        outputs.update(images=d["images"], object_position=d["object_position"])
        np.savez_compressed(root / f"episode_{ep:03d}_geometry_comparison.npz", **outputs)
        rec = {
            "episode": ep,
            "alignment_scale": scale,
            "unconstrained_signed_scale": unconstrained_scale,
            "camera_center_error_cm": (np.linalg.norm(scale * pC - gC, axis=1) * 100).tolist(),
        }
        records.append(rec)
        print(json.dumps(rec), flush=True)
(root / "robot_geometry_controls.json").write_text(
    json.dumps(
        {
            "protocol": (
                "Head-camera orientation/origin anchored to true calibration. A common positive scalar matches the"
                " median wrist-to-head baseline-length ratio; no surface ICP. Privileged control replaces"
                " depth/backprojection/explicit calibration with simulator values but retains original VGGT appearance"
                " and implicit camera tokens. It is an upper-bound diagnostic, not an RGB-only fix."
            ),
            "records": records,
        },
        indent=2,
    )
)
