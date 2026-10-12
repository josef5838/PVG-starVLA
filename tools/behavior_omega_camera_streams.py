"""Eight-image, per-camera VGGT-Omega reconstruction on BEHAVIOR turning_on_radio.

Uses the same inference/export implementation as the RoboTwin stream experiment.
RGB comes from the native videos. Recorded poses are camera-to-robot-base poses
in OpenGL axes, converted to OpenCV extrinsics for descriptive comparisons only.
"""

import argparse
import hashlib
import json
import shutil
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import av
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import robotwin_omega_camera_streams as streams
import torch
from scipy.spatial.transform import Rotation

CAMERAS = ["zed_link_camera_0", "left_realsense_link_camera_0", "right_realsense_link_camera_0"]
LABELS = ["Head / ZED", "Left wrist", "Right wrist"]


def decode_images(video, timestamps, fps):
    """Decode a whole chunk in one seek; audit nearest-frame synchronization."""
    candidates = []
    with av.open(str(video)) as container:
        stream = container.streams.video[0]
        container.seek(int(min(timestamps) / stream.time_base), stream=stream, backward=True)
        for frame in container.decode(stream):
            if frame.pts is None:
                continue
            t = float(frame.pts * stream.time_base)
            if t >= min(timestamps) - 1 / fps:
                candidates.append((t, frame.to_image().convert("RGB")))
            if t > max(timestamps) + 1 / fps:
                break
    if not candidates:
        raise RuntimeError(f"No decoded frames: {video}")
    selected = [min(candidates, key=lambda item: abs(item[0] - t)) for t in timestamps]
    actual = [t for t, _ in selected]
    if (
        len(set(actual)) != len(timestamps)
        or max(abs(a - b) for a, b in zip(actual, timestamps, strict=False)) > 0.51 / fps
    ):
        raise RuntimeError(f"Unsynchronized/duplicate decoded frames: {video}")
    return [im for _, im in selected], actual


def recorded_extrinsics(poses):
    # Dataset writer: inverse(robot_world) @ camera_world, quaternion XYZW.
    # Match OmniGibson depth_to_pcd's GL->CV rotation around camera X by pi.
    camera_to_base = Rotation.from_quat(poses[:, 3:]).as_matrix() @ np.diag([1.0, -1.0, -1.0])
    base_to_camera = camera_to_base.transpose(0, 2, 1)
    translation = -np.einsum("nij,nj->ni", base_to_camera, poses[:, :3])
    return np.concatenate([base_to_camera, translation[..., None]], -1).astype(np.float32)


def report(out, manifest):
    fig, axes = plt.subplots(len(manifest["cases"]), 3, figsize=(18, 3 * len(manifest["cases"]) + 0.5), squeeze=False)
    articles = []
    text = [
        "# BEHAVIOR turning_on_radio: independent camera histories",
        "",
        (
            "**Every stream is one joint eight-image VGGT-Omega call.** The model receives only that camera's "
            "RGB history. Camera calibration, state, actions, and depth videos are not inference inputs."
        ),
        "",
        (
            "Each call returns eight camera poses and depth maps in a shared local frame. All-image geometry combines"
            " those outputs using the predicted cameras; the last-image panel is also conditioned on all eight images."
        ),
        "",
        "[Interactive gallery](index.html) · [Overview](overview.png)",
        "",
        (
            "The head-camera pose stays consistent in these examples. Wrist histories show substantial pose "
            "discrepancies in the selected motion chunks, despite locally plausible-looking depth. "
            "Use the RGB timeline and camera-motion plot alongside the geometry viewer; appearance alone "
            "does not establish that the eight outputs share the correct camera geometry."
        ),
        "",
        (
            "| Chunk | Camera | Frames | Recorded robot-relative rotation (deg) | Predicted rotation (deg) | Max"
            " rotation discrepancy (deg) |"
        ),
        "|---|---|---|---:|---:|---:|",
    ]
    for row, case in enumerate(manifest["cases"]):
        for i, cam in enumerate(CAMERAS):
            relative = f'{case["name"]}/{cam}'
            m = case["streams"][cam]
            text.append(
                f'| {case["label"]} | {LABELS[i]} | {case["frames"][0]}–{case["frames"][-1]} | '  # noqa: RUF001 - intentional mathematical/visual labels
                f'{max(m["recorded_rotation_from_first_deg"]):.3f} | '
                f'{max(m["predicted_rotation_from_first_deg"]):.3f} | '
                f'{m["max_relative_rotation_error_deg"]:.3f} |'
            )
            axes[row, i].imshow(plt.imread(out / relative / "geometry_preview.png"))
            axes[row, i].axis("off")
            axes[row, i].set_title(f'{case["label"]} / {LABELS[i]}', fontsize=10)
            articles.append(
                f'<article><h2>{case["label"]} / {LABELS[i]} — frames {case["frames"][0]}–{case["frames"][-1]}</h2>'  # noqa: RUF001 - intentional mathematical/visual labels
                f'<a href="{relative}/viewer.html">Interactive joint eight-image geometry</a> · '
                f'<a href="{relative}/temporal_preview.png">RGB / depth / confidence / normals</a> · '
                f'<a href="{relative}/camera_consistency.png">Camera-motion comparison</a> · '
                f'<a href="{relative}/metrics.json">Measurements</a>'
                f'<a href="{relative}/viewer.html"><img src="{relative}/geometry_preview.png"></a></article>'
            )
    fig.suptitle(
        f'BEHAVIOR turning_on_radio / episode {manifest["args"]["episode"]}\n'
        "One joint eight-image call per camera; independent local frames and arbitrary scales",
        fontsize=15,
    )
    fig.tight_layout()
    fig.savefig(out / "overview.png", dpi=130)
    plt.close(fig)
    text += [
        "",
        "## Pose comparison scope",
        "",
        (
            "Recorded poses describe the camera relative to the robot base, not a fixed world frame. They are converted"
            " from OpenGL camera-to-base XYZW poses to OpenCV base-to-camera extrinsics. We compare against these under"
            " an approximately stationary-base assumption. The selected chunks have small recorded base velocities;"
            " per-chunk integrated velocity bounds are in the manifest. No simulator replay verification was performed."
            " Depth videos exist in this dataset but were not evaluated here."
        ),
        "",
        "Head-camera motion is estimated rather than assumed fixed: BEHAVIOR has a mobile robot and articulated trunk.",
        "",
        (
            "Confidence filtering removes the bottom 10% of pixels independently for each image; "
            "the raw sampled cloud and full depth arrays are also saved. Moving objects may leave multiple copies "
            "in the combined output; this is not a motion-compensated surface model."
        ),
        "",
        "## Reproduce",
        "",
        "```bash",
        manifest["reproduce_command"],
        "```",
        "",
        (
            "Inputs are eight consecutive native video frames, spanning 7/30 seconds at stride 1. "
            "Selection includes an earlier reference frame plus deliberately selected wrist-motion stress cases. "
            "These examples are not a benchmark-wide accuracy estimate."
        ),
    ]
    (out / "README.md").write_text("\n".join(text) + "\n")
    (out / "index.html").write_text(
        '<!doctype html><meta charset="utf-8"><title>BEHAVIOR radio camera streams</title><style>body{font:16px'
        " system-ui;max-width:1400px;margin:auto;padding:24px;background:#f5f7fa}img{width:100%}article{background:white;padding:18px;margin:18px"  # noqa: E501 - preserve embedded commands or diagnostic text
        " 0}a{color:#1763a6}</style><h1>BEHAVIOR: turning on radio</h1><p>Each camera history is processed jointly in"
        " one eight-image Omega call. The all-image and last-image displays come from the same call. Frames and scales"
        " differ between independent streams.</p><p>Pose comparisons use robot-base-relative recorded poses under an"
        " approximately stationary-base assumption.</p>" + "".join(articles)
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset", type=Path, default=Path("/export/home/haoliang/datasets/turning-on-radio/behavior-1k-2026")
    )
    parser.add_argument("--episode", type=int, default=197)
    parser.add_argument("--end-frames", nargs="+", type=int, default=[1491, 950, 941, 989])
    parser.add_argument("--chunk-length", type=int, default=8)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--omega-root", type=Path, default=streams.DEFAULT_OMEGA)
    parser.add_argument("--checkpoint", type=Path, default=streams.DEFAULT_OMEGA / "ckpt/vggt_omega_1b_512.pt")
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out = args.out.resolve()
    args.out.mkdir(parents=True, exist_ok=False)
    for source in [Path(__file__), Path(streams.__file__), streams.ROOT / "tools/robotwin_omega_stream_viewer.html"]:
        shutil.copy2(source, args.out / source.name)
    info = json.loads((args.dataset / "meta/info.json").read_text())
    episodes = pd.concat([pd.read_parquet(p) for p in sorted((args.dataset / "meta/episodes").rglob("*.parquet"))])
    episode = episodes[episodes.episode_index == args.episode].iloc[0]
    data_path = args.dataset / info["data_path"].format(
        chunk_index=int(episode["data/chunk_index"]), file_index=int(episode["data/file_index"])
    )
    rows = pd.read_parquet(data_path, filters=[("episode_index", "=", args.episode)]).set_index("frame_index")
    labels = {
        1491: "Earlier reference scene",
        950: "Near annotated button press",
        941: "Larger right-wrist motion",
        989: "Larger left-wrist motion",
    }
    manifest = dict(
        status="running",
        started_unix=time.time(),
        args={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        cases=[],
        python=sys.executable,
        torch=torch.__version__,
        fps=info["fps"],
        raw_episode_id=int(episode.raw_episode_id),
        inference_contract=(
            "One joint call per camera history; exactly eight RGB images; shared frozen Omega512; no calibration input."
        ),
        pose_convention=(
            "Recorded camera-to-robot-base OpenGL pose [XYZ, quaternion XYZW]; CV axes via diag(1,-1,-1), then invert."
        ),
        source_sha256={
            p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__), Path(streams.__file__)]
        },
        reproduce_command=f"{sys.executable} tools/behavior_omega_camera_streams.py --episode {args.episode} "
        f'--end-frames {" ".join(map(str,args.end_frames))} --chunk-length {args.chunk_length} --stride {args.stride} '
        f"--out results/omega_camera_streams/behavior_radio_rerun",
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
        for end in args.end_frames:
            frames = list(range(end - (args.chunk_length - 1) * args.stride, end + 1, args.stride))
            chunk = rows.loc[frames]
            name = f"turning_on_radio_ep{args.episode:03d}_f{frames[0]:04d}_{frames[-1]:04d}"
            velocities = np.stack(chunk["observation.state"])[:, :3]
            case = dict(
                name=name,
                label=labels.get(end, f"Chunk ending {end}"),
                frames=frames,
                streams={},
                input_records={},
                episode_timestamps=chunk["timestamp"].tolist(),
                base_motion_check=dict(
                    proprio_indices=[0, 1, 2],
                    semantic="R1Pro proprio base_qvel [vx,vy,yaw_rate]",
                    integrated_translation_upper_bound_m=float(
                        np.linalg.norm(velocities[:, :2], axis=-1).sum() / info["fps"]
                    ),
                    integrated_yaw_upper_bound_deg=float(np.abs(velocities[:, 2]).sum() / info["fps"] * 180 / np.pi),
                ),
            )
            manifest["cases"].append(case)
            for cam in CAMERAS:
                out = args.out / name / cam
                (out / "inputs").mkdir(parents=True)
                key = f"observation.rgb.{cam}"
                video = args.dataset / info["video_path"].format(
                    video_key=key,
                    chunk_index=int(episode[f"videos/{key}/chunk_index"]),
                    file_index=int(episode[f"videos/{key}/file_index"]),
                )
                requested = [float(episode[f"videos/{key}/from_timestamp"]) + float(t) for t in chunk["timestamp"]]
                native_images, actual = decode_images(video, requested, info["fps"])
                paths = []
                for frame, image in zip(frames, native_images, strict=False):
                    path = out / "inputs" / f"frame_{frame:05d}.png"
                    image.save(path)
                    paths.append(path)
                case["input_records"][cam] = dict(
                    source_video=str(video),
                    requested_timestamps=requested,
                    decoded_timestamps=actual,
                    native_size_wh=list(native_images[0].size),
                    sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
                )
                poses = np.stack(chunk[f"observation.robot2cam_pose.{cam}"]).astype(float)
                e = recorded_extrinsics(poses)
                # Algebraic convention check against the directly recorded camera centers.
                _, centers = streams.relative_cameras(e)
                expected = (poses[:, :3] - poses[0, :3]) @ e[0, :, :3].T
                np.testing.assert_allclose(centers, expected, atol=1e-6)
                np.savez(out / "recorded_camera_poses.npz", robot2cam_pose=poses, base_to_camera_cv=e, frames=frames)
                print(f"INFER {name} {cam}: one joint call, {len(paths)} images", flush=True)
                stats = streams.infer_stream(model, paths, e, frames, out, SimpleNamespace(resolution=args.resolution))
                stats["interpretation"] = (
                    "Pose discrepancy against robot-base-relative recorded camera poses, assuming approximately"
                    " stationary base. No replay verification."
                )
                case["streams"][cam] = stats
                streams.write_json(out / "metrics.json", stats)
                streams.write_json(args.out / "manifest.json", manifest)
                print(
                    f'COMPLETE {cam}: {stats["inference_seconds"]:.2f}s; rotation discrepancy'
                    f' {stats["max_relative_rotation_error_deg"]:.2f} deg',
                    flush=True,
                )
        report(args.out, manifest)
        manifest.update(status="completed", completed_unix=time.time())
        streams.write_json(args.out / "manifest.json", manifest)
    except Exception as exc:
        manifest.update(status="failed", error=repr(exc))
        streams.write_json(args.out / "manifest.json", manifest)
        raise
    print(f"COMPLETED {args.out}", flush=True)


if __name__ == "__main__":
    main()
