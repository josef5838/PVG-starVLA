"""Capture synchronized simulator RGB and object/EE labels on registered scenes."""

import argparse
import io
import json
import sys
import time
from pathlib import Path

import h5py
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from launch_omega512_protocol import (  # noqa: E402
    SIM,
    phases,
    write,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    out = Path(manifest["visual_probe"]["output_dir"]) / "labels"
    out.mkdir(parents=True, exist_ok=True)
    rows = [row for group in manifest["visual_probe"]["splits"].values() for row in group]
    if args.limit:
        rows = rows[: args.limit]
    from robotwin_omega_intervention_scenes import setup_runtime

    _, context, make_env, config = setup_runtime(SIM, 0)
    config.update(eval_mode=True, save_data=False, need_plan=True)
    status = dict(status="running", stage="simulator_labels", completed=[], target_episodes=len(rows))
    write(out / "status.json", status)
    try:
        for row in rows:
            ep = row["training_episode"]
            target = out / f"episode_{ep:03d}.npz"
            audit = out / f"episode_{ep:03d}.json"
            if target.exists() and audit.exists() and json.loads(audit.read_text()).get("verified"):
                status["completed"].append(ep)
                continue
            source_ep = row["source_episode"]
            env = make_env("rotate_qrcode")
            started = time.monotonic()
            try:
                with h5py.File(row["source_hdf5"]) as f:
                    recorded_q = f["joint_action/vector"][0]
                    initial_rgb = np.stack(
                        [
                            np.asarray(Image.open(io.BytesIO(bytes(f[f"observation/{cam}/rgb"][0]))).convert("RGB"))
                            for cam in ["head_camera", "left_camera", "right_camera"]
                        ]
                    )
                    intrinsics = np.stack(
                        [
                            f[f"observation/{cam}/intrinsic_cv"][0]
                            for cam in ["head_camera", "left_camera", "right_camera"]
                        ]
                    )
                    extrinsics = np.stack(
                        [
                            f[f"observation/{cam}/extrinsic_cv"][0]
                            for cam in ["head_camera", "left_camera", "right_camera"]
                        ]
                    )
                env.setup_demo(now_ep_num=0, seed=row["simulator_seed"], is_test=True, **config)
                if int(env.model_id) != row["model_id"]:
                    raise RuntimeError("Object model differs from recorded source")
                initial_obs = env.get_obs()
                actual_initial = np.stack(
                    [initial_obs["observation"][cam]["rgb"] for cam in ["head_camera", "left_camera", "right_camera"]]
                )
                initial_image_error = np.abs(actual_initial.astype(float) - initial_rgb).mean((1, 2, 3))
                initial_state_error = float(np.max(np.abs(initial_obs["joint_action"]["vector"] - recorded_q)))
                initial_object_error = float(
                    np.linalg.norm(env.qrcode.get_pose().p - np.asarray(row["initial_position"]))
                )
                initial_audit = dict(
                    training_episode=ep,
                    initial_rgb_mae=initial_image_error.tolist(),
                    initial_state_max_error=initial_state_error,
                    initial_object_error_m=initial_object_error,
                    expected_object_xyz=row["initial_position"],
                    object_xyz=env.qrcode.get_pose().p.tolist(),
                    model_id=int(env.model_id),
                    expected_model_id=row["model_id"],
                    acceptance=(
                        "Matching model and robot state, object position within 2mm, head RGB MAE <=6/255. Wrist RGB"
                        " errors audited; synchronized regenerated image/pose labels do not rely on matching recorded"
                        " wrist pixels."
                    ),
                )
                write(out / f"episode_{ep:03d}_initial_scene.json", initial_audit)
                if initial_image_error[0] > 6 or initial_state_error > 1e-6 or initial_object_error > 0.002:
                    raise RuntimeError(f"Initial scene differs from source recording: {initial_audit}")
                samples = []
                images = {}

                def capture(env=env, images=images, samples=samples):
                    t = len(samples)
                    q = np.asarray(env.robot.get_left_arm_jointState() + env.robot.get_right_arm_jointState())
                    ee = np.asarray(env.get_arm_pose("right"))
                    pose = env.qrcode.get_pose()
                    samples.append((q, ee, np.r_[pose.p, pose.q]))
                    observation = env.get_obs()
                    im = np.stack(
                        [
                            observation["observation"][cam]["rgb"]
                            for cam in ["head_camera", "left_camera", "right_camera"]
                        ]
                    )
                    encoded = []
                    for view in im:
                        stream = io.BytesIO()
                        Image.fromarray(view).save(stream, format="PNG", compress_level=1)
                        encoded.append(stream.getvalue())
                    images[t] = encoded

                env._take_picture = capture
                env.play_once()
                q = np.stack([x[0] for x in samples])
                ee = np.stack([x[1] for x in samples])
                obj = np.stack([x[2] for x in samples])
                phase_records = []
                for phase, ix in phases(q[:, 13]):
                    if len(ix) < 5:
                        raise RuntimeError(f"Insufficient regenerated phase {phase}: {len(ix)}")
                    phase_records += [
                        dict(frame=int(t), phase=phase) for t in ix[np.rint(np.linspace(0, len(ix) - 1, 5)).astype(int)]
                    ]
                frames = [r["frame"] for r in phase_records]
                verified = bool(np.isfinite(obj).all() and np.isfinite(ee).all())
                record = dict(
                    training_episode=ep,
                    source_episode=source_ep,
                    simulator_seed=row["simulator_seed"],
                    verified=verified,
                    frames=len(q),
                    selected_frames=frames,
                    phase_records=phase_records,
                    initial_scene_rgb_mae=initial_image_error.tolist(),
                    initial_state_max_error=initial_state_error,
                    initial_object_error_m=initial_object_error,
                    initial_scene_acceptance=initial_audit["acceptance"],
                    label_alignment=(
                        "RGB and object/EE poses captured at the same simulator callback; regenerated trajectories, not"
                        " labels for recorded later frames."
                    ),
                    expert_success=bool(env.check_success()),
                    elapsed_seconds=time.monotonic() - started,
                    right_robot_base_to_world=env.robot.right_entity_origion_pose.to_transformation_matrix().tolist(),
                    coordinates=(
                        "Simulator world metres, actor origin and right EE reference. Displacement = object_xyz -"
                        " EE_xyz. No estimated depth scale."
                    ),
                    plan_method=(
                        "Current simulator expert, need_plan=True, saved source seed; expert success recorded"
                        " separately."
                    ),
                )
                write(audit, record)
                if not verified:
                    raise RuntimeError(f"Label alignment rejected: {record}")
                temp = target.with_suffix(".tmp.npz")
                selected_rgb = np.stack(
                    [
                        np.stack([np.asarray(Image.open(io.BytesIO(encoded)).convert("RGB")) for encoded in images[t]])
                        for t in frames
                    ]
                )
                np.savez_compressed(
                    temp,
                    frames=np.asarray(frames),
                    object_pose=obj,
                    right_ee=ee,
                    regenerated_commands=q,
                    images=selected_rgb,
                    intrinsics=intrinsics,
                    initial_extrinsics=extrinsics,
                )
                temp.replace(target)
                status["completed"].append(ep)
                write(out / "status.json", status)
                print(json.dumps(record), flush=True)
            finally:
                env.close_env(clear_cache=True)
        status.update(status="completed", stage="verified_labels")
        write(out / "status.json", status)
    except Exception as error:
        status.update(status="failed", error=repr(error))
        write(out / "status.json", status)
        raise


if __name__ == "__main__":
    main()
