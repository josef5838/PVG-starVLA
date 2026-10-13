"""Matched seen, heldout and fresh-scene closed-loop camera-stream evaluation."""

import argparse
import fcntl
import io
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import h5py
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--variant", required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--limit-scenes", type=int)
    parser.add_argument("--expert", action="store_true")
    parser.add_argument("--verify-scenes-only", action="store_true")
    parser.add_argument("--execute-horizon", type=int, choices=[4, 8])
    parser.add_argument(
        "--expert-data", type=Path, help="Prepared inverse-normalized expert actions, keyed by dataset episode"
    )
    parser.add_argument("--robotwin-root", type=Path, default=Path("/export/home/haoliang/projects/RoboTwin_PVG"))
    args = parser.parse_args()
    args.manifest = args.manifest.resolve()
    args.out = args.out.resolve()
    args.robotwin_root = args.robotwin_root.resolve()
    manifest = json.loads(args.manifest.read_text())
    expert_data = np.load(args.expert_data.resolve()) if args.expert_data else None
    if expert_data is not None and not args.expert:
        raise ValueError("--expert-data requires --expert")
    args.out.mkdir(parents=True, exist_ok=True)
    # Serialize resumptions targeting the same result/video directory.
    output_lock = (args.out / ".worker.lock").open("a")
    fcntl.flock(output_lock, fcntl.LOCK_EX)
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(ROOT / "tools"))
    from robotwin_omega_intervention_scenes import setup_runtime
    from rotate_qrcode_metrics import quaternion_error_degrees, rotation_diagnostics, summarize_phases, task_geometry

    _, renderer, make_env, config = setup_runtime(
        args.robotwin_root, 0, task=manifest["task"],
        native_camera_contract=manifest.get("native_camera_contract", False)
    )
    from examples.simBenchmarks.Robotwin.eval_files import model2robotwin_interface as adapter

    run = next(r for r in manifest["runs"] if r["variant"] == args.variant)
    checkpoint = run.get("checkpoint") or run["checkpoints"]["best"]
    source = Path(manifest["source"])
    references = np.load(manifest["scene_reference_data"]) if manifest.get("scene_reference_data") else None
    if references is not None and args.expert:
        raise ValueError("Local RGB/state references do not contain expert action trajectories")
    info = None if references is not None else json.loads((source / "scene_info.json").read_text())
    registration = (
        json.loads(Path(manifest["fresh_scene_registration"]).read_text())
        if manifest.get("fresh_scene_registration")
        else []
    )
    fresh_rows = registration.get("fresh_scenes", []) if isinstance(registration, dict) else registration
    fresh = {r["seed"]: r for r in fresh_rows}
    execute_horizon = args.execute_horizon or manifest.get("execute_horizon", 8)
    scenes = manifest["rollout_scenes"]
    if args.expert:
        scenes = [s for s in scenes if s["source_episode"] is not None]
    if args.limit_scenes:
        scenes = scenes[: args.limit_scenes]
    expert_actions = None

    class ExpertTransport:
        def __init__(self, *args):
            self.index = 0

        def get_server_metadata(self):
            return dict(
                action_chunk_size=16,
                include_state=True,
                execute_horizon=execute_horizon,
                num_ddim_steps=20,
                camera_history_length=8,
                camera_history_stride=1,
                training_obs_image_size=[640, 480],
                preserve_native_images=False,
                instruction_override=manifest["task"],
            )

        def predict_action(self, request):
            start = self.index * execute_horizon
            self.index += 1
            ids = np.minimum(np.arange(start, start + 16), len(expert_actions) - 1)
            return {"actions": expert_actions[ids][None]}

        def close(self):
            pass

    if args.expert:
        adapter.WebsocketClientPolicy = ExpertTransport
    records = []
    for _index, scene in enumerate(scenes):
        # Fresh process runs can resume after interruption, preserving completed cases.
        case = args.out / f"{scene['split']}_seed_{scene['seed']}"
        result_path = case.with_suffix(".json")
        if result_path.exists():
            records.append(json.loads(result_path.read_text()))
            continue
        model = None
        env = make_env(manifest["task"])
        pipe = None
        trace = []
        try:
            env.setup_demo(now_ep_num=0, seed=scene["seed"], is_test=True, **config)
            actor, arm = task_geometry(env)
            obs = env.get_obs()
            frame = obs["observation"]["head_camera"]["rgb"]
            image_errors = {}
            gray_errors = {}
            state_error = None
            source_verified = False
            registered_verified = False
            local_audit = {}
            if references is not None:
                from robotwin_stream_scene_reference import verify_scene

                local_audit = verify_scene(
                    scene, references, obs, env.model_id, arm, actor.get_pose().p, actor.get_pose().q,
                    manifest["camera_order"], manifest["initial_image_validation"]["grayscale_mae_limit"]
                )
                state_error = local_audit["initial_state_max_error"]
                image_errors = local_audit["initial_image_mae"]
                gray_errors = local_audit["initial_grayscale_mae"]
                source_verified = True
                registered_verified = True
            elif scene["source_episode"] is not None:
                sid = scene["source_episode"]
                with h5py.File(source / f"data/episode{sid}.hdf5") as file:
                    state_error = float(np.max(np.abs(obs["joint_action"]["vector"] - file["joint_action/vector"][0])))
                    for cam in manifest["camera_order"]:
                        image = Image.open(io.BytesIO(bytes(file[f"observation/{cam}/rgb"][0]))).convert("RGB")
                        actual = np.asarray(Image.fromarray(obs["observation"][cam]["rgb"]).resize(image.size))
                        image_errors[cam] = float(np.abs(actual.astype(float) - np.asarray(image)).mean())
                        gray_errors[cam] = float(
                            np.abs(
                                np.asarray(Image.fromarray(actual).convert("L")).astype(float)
                                - np.asarray(image.convert("L"))
                            ).mean()
                        )
                    order = [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12, 6, 13]
                    expert_actions = file["joint_action/vector"][1:][:, order]
                    if expert_data is not None:
                        expert_actions = expert_data[f"episode_{scene['episode']}"]
                expected = int(info[f"episode_{sid}"]["info"]["{A}"].split("base")[-1])
                appearance_ok = (
                    max(gray_errors.values()) <= 6
                    if manifest.get("initial_image_validation")
                    else max(image_errors.values()) <= 6
                )
                if int(env.model_id) != expected or state_error > 1e-6 or not appearance_ok:
                    raise RuntimeError(
                        f"Scene mismatch: {scene}, model={env.model_id}/{expected}, state={state_error},"
                        f" images={image_errors}"
                    )
                source_verified = True
                if "position" in scene and "quaternion" in scene:
                    position_error = float(np.linalg.norm(actor.get_pose().p - np.asarray(scene["position"])))
                    orientation_error = quaternion_error_degrees(actor.get_pose().q, scene["quaternion"])
                    if position_error > 1e-4 or orientation_error > 0.1:
                        raise RuntimeError(
                            f"Registered source object pose mismatch: {position_error}, {orientation_error}"
                        )
                    registered_verified = True
            elif scene["seed"] in fresh:
                reference = fresh[scene["seed"]]
                state_error = float(
                    np.max(np.abs(obs["joint_action"]["vector"] - np.asarray(reference["initial_state"])))
                )
                pose_error = float(np.linalg.norm(actor.get_pose().p - np.asarray(reference["position"])))
                quaternion_error = (
                    quaternion_error_degrees(actor.get_pose().q, reference["quaternion"])
                    if "quaternion" in reference
                    else 0.0
                )
                if (
                    state_error > 1e-6
                    or pose_error > 1e-5
                    or quaternion_error > 1e-3
                    or int(env.model_id) != reference["model_id"]
                    or arm != reference["arm"]
                ):
                    raise RuntimeError(f'Registered fresh scene mismatch at seed {scene["seed"]}')
                registered_verified = True
            if args.verify_scenes_only:
                record = dict(scene)
                record.update(**local_audit, status="passed",
                              source_scene_verified=source_verified, registered_scene_verified=registered_verified)
                result_path.write_text(json.dumps(record, indent=2)+"\n")
                records.append(record)
                (args.out / "results.json").write_text(json.dumps(records, indent=2)+"\n")
                print(json.dumps(record), flush=True)
                continue
            model = adapter.ModelClient(checkpoint, port=args.port, execute_horizon=execute_horizon)
            model.diffusion_seed = manifest["diffusion_seed"]
            if model.camera_history_length != 8 or model.preserve_native_images or list(model.image_size) != [640, 480]:
                raise RuntimeError("Server/client history or image-size contract mismatch")
            instruction = manifest["task"]
            env.set_instruction(instruction)
            model.reset(instruction)
            video = case.with_suffix(".mp4")
            ffmpeg = shutil.which("ffmpeg") or str(Path(sys.executable).parent / "ffmpeg")
            pipe = subprocess.Popen(
                [
                    ffmpeg,
                    "-y",
                    "-loglevel",
                    "error",
                    "-f",
                    "rawvideo",
                    "-pixel_format",
                    "rgb24",
                    "-video_size",
                    f"{frame.shape[1]}x{frame.shape[0]}",
                    "-framerate",
                    "10",
                    "-i",
                    "-",
                    "-pix_fmt",
                    "yuv420p",
                    "-vcodec",
                    "libx264",
                    "-threads",
                    "2",
                    str(video),
                ],
                stdin=subprocess.PIPE,
            )
            env.eval_video_path = str(args.out)
            env._set_eval_video_ffmpeg(pipe)
            initial = actor.get_pose().p.copy()
            initial_quaternion = actor.get_pose().q.copy()
            started = time.monotonic()
            limit = env.step_lim if not args.expert else len(expert_actions)
            for step in range(limit):
                obs = env.get_obs()
                env.now_obs = obs
                example = dict(
                    image=[obs["observation"][cam]["rgb"] for cam in manifest["camera_order"]],
                    state=obs["joint_action"]["vector"],
                    lang=instruction,
                )
                inference = time.monotonic()
                action = model.step(example, step=step)
                inference = time.monotonic() - inference
                env.take_action(action, action_type="qpos")
                contacts = set()
                for contact in env.scene.get_contacts():
                    names = [body.entity.name for body in contact.bodies]
                    if actor.actor.name in names:
                        contacts.update(n for n in names if n in env.robot.gripper_name)
                command = np.asarray(env.robot.get_left_arm_jointState() + env.robot.get_right_arm_jointState())
                trace.append(
                    dict(
                        step=step,
                        action=action.tolist(),
                        state_before=np.asarray(example["state"]).tolist(),
                        command_after=command.tolist(),
                        object_pose=np.r_[actor.get_pose().p, actor.get_pose().q].tolist(),
                        functional_point=np.asarray(
                            actor.get_pose().p if hasattr(env, "qrcode") else actor.get_functional_point(0)[:3]
                        ).tolist(),
                        object_gripper_contacts=sorted(contacts),
                        closed_two_finger_contact=bool(
                            len(contacts) >= 2 and command[13 if arm == "right" else 6] < 0.5
                        ),
                        object_lift_m=float(actor.get_pose().p[2] - initial[2]),
                        history_observations=len(model.camera_history),
                        policy_step_seconds=inference,
                        **rotation_diagnostics(env, actor, command),
                    )
                )
                if env.eval_success:
                    break
            record = dict(scene)
            record.update(
                variant=args.variant,
                checkpoint=checkpoint,
                mode=("expert_roundtrip" if expert_data is not None else "expert") if args.expert else "policy",
                source_scene_verified=source_verified,
                registered_scene_verified=registered_verified,
                initial_state_max_error=state_error,
                initial_image_mae=image_errors,
                initial_grayscale_mae=gray_errors,
                model_id=int(env.model_id),
                arm=arm,
                success=bool(env.eval_success or env.check_success()),
                steps=len(trace),
                grasp_contact=any(x["closed_two_finger_contact"] for x in trace),
                lift_4cm=any(x["object_lift_m"] >= 0.04 for x in trace),
                max_lift_m=max(x["object_lift_m"] for x in trace),
                elapsed_seconds=time.monotonic() - started,
                video=str(video),
                trace=str(case.with_suffix(".trace.json")),
                history_length=8,
                history_stride=1,
                image_size=list(model.image_size),
                num_ddim_steps=20,
                execute_horizon=execute_horizon,
                diffusion_seed=manifest["diffusion_seed"],
                initial_object_position=initial.tolist(),
                initial_object_quaternion=initial_quaternion.tolist(),
                final_object_position=actor.get_pose().p.tolist(),
                step_limit=env.step_lim,
            )
            if hasattr(env, "qrcode"):
                record.update(
                    final_orientation_error_deg=quaternion_error_degrees(actor.get_pose().q),
                    minimum_orientation_error_deg=min(x["orientation_error_deg"] for x in trace),
                    final_placement_xy_error_m=trace[-1]["placement_xy_error_m"],
                    phase_diagnostics=summarize_phases(trace, record["success"]),
                )
            case.with_suffix(".trace.json").write_text(json.dumps(trace) + "\n")
            result_path.write_text(json.dumps(record, indent=2) + "\n")
            records.append(record)
            (args.out / "results.json").write_text(json.dumps(records, indent=2) + "\n")
            print(json.dumps(record), flush=True)
        finally:
            if pipe is not None:
                env._del_eval_video_ffmpeg()
                if pipe.wait(timeout=30):
                    raise RuntimeError("Video encoding failed")
                env.eval_video_path = None
            env.close_env(clear_cache=True)
            if model is not None:
                model.client.close()
    (args.out / "results.json").write_text(json.dumps(records, indent=2) + "\n")


if __name__ == "__main__":
    main()
