"""Audit the recorded left-wrist motion against pose fields and RGB correspondences."""

import io
import json
from pathlib import Path

import cv2
import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
SOURCE = Path("/export/home/haoliang/projects/RoboTwin/data/stack_blocks_three/demo_clean/data/episode7.hdf5")
OUT = ROOT / "results/omega_camera_streams/20261009_chunk8_left_motion/rotation_audit"


def main():
    OUT.mkdir(exist_ok=True)
    frames = [170, 177]
    with h5py.File(SOURCE) as f:
        images = [np.asarray(Image.open(io.BytesIO(bytes(f["observation/left_camera/rgb"][t])))) for t in frames]
        e = f["observation/left_camera/extrinsic_cv"][frames].astype(float)
        k = f["observation/left_camera/intrinsic_cv"][frames].astype(float)
        w = f["observation/left_camera/cam2world_gl"][frames].astype(float)
        ee = f["endpose/left_endpose"][frames]
    r = Rotation.from_matrix(e[:, :, :3])
    delta = r[1] * r[0].inv()
    axis = delta.as_rotvec() / delta.magnitude()
    wr = Rotation.from_matrix(w[:, :3, :3])
    eer = Rotation.from_quat(ee[:, [4, 5, 6, 3]])
    recorded = dict(
        cv_rotation_deg=float(np.degrees(delta.magnitude())),
        gl_rotation_deg=float(np.degrees((wr[1] * wr[0].inv()).magnitude())),
        end_effector_rotation_deg=float(np.degrees((eer[1] * eer[0].inv()).magnitude())),
        relative_axis_in_last_camera=axis.tolist(),
        camera_displacement_m=float(np.linalg.norm(w[1, :3, 3] - w[0, :3, 3])),
        optical_axis_direction_change_deg=float(np.degrees(np.arccos(np.clip(np.dot(e[0, 2, :3], e[1, 2, :3]), -1, 1)))),
        cv_gl_rotation_max_difference=float(
            np.max(np.abs(w[:, :3, :3] @ np.diag([1, -1, -1]) - e[:, :, :3].transpose(0, 2, 1)))
        ),
    )
    sift = cv2.SIFT_create(contrastThreshold=0.01)
    features = [sift.detectAndCompute(cv2.cvtColor(im, cv2.COLOR_RGB2GRAY), None) for im in images]
    kp, desc = zip(*features, strict=False)
    matches = cv2.BFMatcher().knnMatch(*desc, k=2)
    good = [a for a, b in matches if a.distance < 0.75 * b.distance]
    points = [np.array([kp[i][m.queryIdx if i == 0 else m.trainIdx].pt for m in good], np.float32) for i in range(2)]
    _, mask = cv2.findHomography(*points, cv2.RANSAC, 2)
    chosen = mask.ravel().astype(bool)
    p0, p1 = [p[chosen] for p in points]
    # Nominal tabletop height from RoboTwin's create_table_and_wall. Neither pose
    # nor height is fitted to these matches; the data sequence may include shadows.
    height = 0.74
    plane_maps = [k[i] @ np.column_stack([e[i, :, 0], e[i, :, 1], e[i, :, 2] * height + e[i, :, 3]]) for i in range(2)]
    combined_h = plane_maps[1] @ np.linalg.inv(plane_maps[0])
    rotation_h = k[1] @ e[1, :, :3] @ e[0, :, :3].T @ np.linalg.inv(k[0])
    combined = cv2.perspectiveTransform(p0[None], combined_h)[0]
    rotation_only = cv2.perspectiveTransform(p0[None], rotation_h)[0]
    errors = np.linalg.norm(combined - p1, axis=-1)
    audit = dict(
        source=str(SOURCE),
        frames=frames,
        camera="left_camera",
        recorded=recorded,
        rgb_check=dict(
            sift_features=list(map(len, kp)),
            ratio_matches=len(good),
            homography_inliers=len(p0),
            nominal_table_height_m=height,
            points_first=p0.tolist(),
            points_last=p1.tolist(),
            combined_motion_predictions=combined.tolist(),
            residuals_px=errors.tolist(),
            median_residual_px=float(np.median(errors)),
            rotation_only_median_residual_px=float(np.median(np.linalg.norm(rotation_only - p1, axis=-1))),
        ),
        interpretation=(
            "Internal pose agreement and RGB consistency support recorded camera motion. Six automatically matched"
            " background features are a small, planar check, not independent full-scene ground truth."
        ),
    )
    (OUT / "audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(images[0])
    axes[1].imshow(images[1])
    axes[2].imshow(images[1])
    colors = plt.get_cmap("tab10")(np.arange(len(p0)))
    for i in range(len(p0)):
        axes[0].scatter(*p0[i], color=colors[i], s=50)
        axes[0].text(*p0[i], str(i + 1), color="black", fontsize=9)
        axes[1].scatter(*p1[i], facecolors="none", edgecolors=[colors[i]], s=110, linewidths=2)
        axes[1].scatter(*combined[i], color=colors[i], marker="x", s=60, linewidths=2)
        axes[2].scatter(*p1[i], facecolors="none", edgecolors=[colors[i]], s=110, linewidths=2)
        axes[2].scatter(*rotation_only[i], color=colors[i], marker="x", s=60, linewidths=2)
    axes[0].set_title("Frame 170: matched background features")
    axes[1].set_title(
        f"Frame 177: recorded rotation + translation\nMedian prediction error {np.median(errors):.2f} pixels"
    )
    axes[2].set_title("Hypothetical rotation alone\nTranslation omitted; predictions fall outside image")
    for ax in axes[:2]:
        ax.set_xlim(0, 320)
        ax.set_ylim(240, 0)
    axes[2].set_xlim(0, 320)
    axes[2].set_ylim(max(420, rotation_only[:, 1].max() + 15), 0)
    for ax in axes:
        ax.set_xlabel("Pixel x")
        ax.set_ylabel("Pixel y")
    fig.suptitle(
        "Left wrist: recorded 29.26° tilt and 16.94 cm translation act together\nCircles = observed matches; crosses ="
        " projections using recorded poses and nominal tabletop height. No pose fit.",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(OUT / "rgb_motion_check.png", dpi=150)
    plt.close(fig)
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
