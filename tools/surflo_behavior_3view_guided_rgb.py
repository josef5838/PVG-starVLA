#!/usr/bin/env python3
"""RGB-guided Surflo reconstruction of one trainer-format BEHAVIOR sample."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from omegaconf import OmegaConf
from torchvision.transforms import functional as VF

from starVLA.dataloader.lerobot_datasets import get_vla_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", default="examples/simBenchmarks/Behavior/train_files/surflo_dit_turn_on_radio_2026.yaml"
    )
    parser.add_argument("--checkpoint", default="/export/home/haoliang/projects/Surflo/checkpoints/surflo_v0.pt")
    parser.add_argument("--model-config", default="/export/home/haoliang/projects/Surflo/configs/model/surflo.yaml")
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--num-query-points", type=int, default=65536)
    parser.add_argument(
        "--out", default="results/Checkpoints/surflo_dit_turn_on_radio_2026/quality_check_3view_gpu1_guided_rgb"
    )
    return parser.parse_args()


def trainer_image_tensor(image, image_size: int = 518) -> torch.Tensor:
    return VF.resize(VF.to_tensor(image.convert("RGB")), [image_size, image_size], antialias=True)


def save_input_views(images, out_path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for _index, (axis, image, label) in enumerate(
        zip(axes, images, ("head", "left wrist", "right wrist"), strict=False)
    ):
        axis.imshow(image)
        axis.set_title(label)
        axis.axis("off")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def preview_indices(points: np.ndarray) -> np.ndarray:
    center = np.median(points, axis=0)
    radial_distance = np.linalg.norm(points - center, axis=1)
    keep = np.flatnonzero(radial_distance <= np.quantile(radial_distance, 0.995))
    rng = np.random.default_rng(42)
    return rng.choice(keep, size=min(30000, len(keep)), replace=False)


def save_rgb_cloud_views(points: np.ndarray, colors: np.ndarray, out_path: Path) -> None:
    chosen = preview_indices(points)
    cloud, rgb = points[chosen], np.clip(colors[chosen], 0.0, 1.0)
    center = np.median(cloud, axis=0)
    radius = np.quantile(np.abs(cloud - center), 0.995)
    fig = plt.figure(figsize=(16, 5))
    for panel, (elev, azim, label) in enumerate(
        ((20, 30, "front-right"), (20, 150, "rear-left"), (75, 30, "top")), start=1
    ):
        axis = fig.add_subplot(1, 3, panel, projection="3d")
        axis.scatter(cloud[:, 0], cloud[:, 1], cloud[:, 2], c=rgb, s=0.65, alpha=0.9, linewidths=0, depthshade=False)
        axis.view_init(elev=elev, azim=azim)
        axis.set_xlim(center[0] - radius, center[0] + radius)
        axis.set_ylim(center[1] - radius, center[1] + radius)
        axis.set_zlim(center[2] - radius, center[2] + radius)
        axis.set_box_aspect((1, 1, 1))
        axis.set_title(label)
        axis.set_axis_off()
    fig.suptitle("Surflo guided RGB point cloud (TSDF-fused input colors)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def save_rgb_orthographic_views(points: np.ndarray, colors: np.ndarray, out_path: Path) -> None:
    chosen = preview_indices(points)
    cloud, rgb = points[chosen], np.clip(colors[chosen], 0.0, 1.0)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for axis, (u, v, depth, label) in zip(axes, ((0, 1, 2, "XY"), (0, 2, 1, "XZ"), (1, 2, 0, "YZ")), strict=False):
        order = np.argsort(cloud[:, depth])
        axis.scatter(cloud[order, u], cloud[order, v], c=rgb[order], s=0.6, linewidths=0, rasterized=True)
        axis.set_aspect("equal", adjustable="box")
        axis.set_title(f"{label} projection")
        axis.set_xlabel(("x", "y", "z")[u])
        axis.set_ylabel(("x", "y", "z")[v])
    fig.suptitle("Guided RGB geometry check")
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = OmegaConf.load(args.config)
    dataset = get_vla_dataset(data_cfg=cfg.datasets.vla_data, mode="train", seed=42)
    source_dataset, trajectory_id, base_index = dataset.sample_step(args.sample_index)
    sample = dataset[args.sample_index]
    images = sample["image"]
    if len(images) != 3:
        raise ValueError(f"Expected 3 views, found {len(images)}")
    for index, image in enumerate(images):
        image.save(out_dir / f"input_view_{index}.png")
    save_input_views(images, out_dir / "input_views.png")
    trainer_images = torch.stack([trainer_image_tensor(image) for image in images])

    # Make unqualified CUDA allocations resolve to the physical GPU 1.
    torch.cuda.set_device(1)
    device = torch.device("cuda:1")
    from surflo import Surflo, load_preset, save_mesh, save_ply

    surflo = Surflo.from_checkpoint(args.checkpoint, model_cfg=args.model_config, device=device, use_ema=True)
    surflo.model.device = device
    scene = surflo.encode(trainer_images, target_size=518, cull_radius=10.0)
    guided_cfg = load_preset("guided", "no_expert")
    result = scene.reconstruct(
        mode="guided",
        config_block=guided_cfg,
        num_query_points=args.num_query_points,
        opacity_threshold=0.1,
    )
    colors = scene.color_points(result["points"], result)
    save_ply(result, out_dir / "guided_rgb_point_cloud.ply", colors=colors)

    points_np = result["points"].detach().float().cpu().numpy()
    colors_np = colors.detach().float().cpu().numpy()
    save_rgb_cloud_views(points_np, colors_np, out_dir / "guided_rgb_3d_views.png")
    save_rgb_orthographic_views(points_np, colors_np, out_dir / "guided_rgb_orthographic.png")

    mesh_status: dict[str, object] = {"written": False}
    try:
        mesh = scene.extract_mesh(result, mesh_cfg=load_preset("mesh"))
        n_verts, n_faces = save_mesh(mesh, out_dir / "guided_mesh.ply")
        mesh_status = {"written": True, "vertices": n_verts, "faces": n_faces}
    except Exception as exc:
        mesh_status = {"written": False, "error": f"{type(exc).__name__}: {exc}"}

    camera_files = {
        key: str(source_dataset.get_video_path(trajectory_id, key)) for key in ("head", "left_wrist", "right_wrist")
    }
    metadata = {
        "sample_index": args.sample_index,
        "task": sample["lang"],
        "trajectory_id": int(trajectory_id),
        "base_index": int(base_index),
        "camera_files": camera_files,
        "trainer_input_shape": list(trainer_images.shape),
        "mode": "guided",
        "guided_preset": "no_expert (inherits minimal; RGB rendering guidance enabled, DA3 priors disabled)",
        "num_query_points": args.num_query_points,
        "visible_colored_points": len(points_np),
        "mesh": mesh_status,
        "device": str(device),
        "checkpoint": args.checkpoint,
    }
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
