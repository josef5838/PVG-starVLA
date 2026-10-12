"""Verified training-scene rollouts and recorded-expert replay through ModelClient.

The oracle replaces model inference/transport only. Expert chunks pass through
the production normalization transform (prepared separately) and the unchanged
ModelClient chunk scheduling/order conversion and simulator take_action path.
"""

import argparse
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
SCENES = [(0, 51, 131), (1, 60, 162), (2, 38, 91)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["expert", "rollout"], required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--variant", required=True)
    parser.add_argument("--port", type=int, default=5750)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expert-input", type=Path)
    parser.add_argument("--episodes", default="0,1,2")
    parser.add_argument("--execute-horizon", type=int, default=8)
    parser.add_argument("--raw-expert", action="store_true")
    parser.add_argument("--scene-manifest", type=Path, help="Protocol manifest supplying matched training scenes")
    parser.add_argument("--diffusion-seed", type=int)
    parser.add_argument(
        "--match-training-image-size",
        action="store_true",
        help="Resize simulator RGB to the checkpoint training image size before Omega preprocessing",
    )
    parser.add_argument("--robotwin-root", type=Path, default=Path("/export/home/haoliang/projects/RoboTwin_PVG"))
    args = parser.parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(ROOT / "tools"))
    from robotwin_omega_intervention_scenes import setup_runtime

    _, render_context, make_env, config = setup_runtime(args.robotwin_root, 0)
    from examples.simBenchmarks.Robotwin.eval_files import model2robotwin_interface as adapter

    source = Path("/export/home/haoliang/projects/RoboTwin/data/rotate_qrcode/demo_clean")
    info = json.loads((source / "scene_info.json").read_text())
    expert_data = np.load(args.expert_input) if args.mode == "expert" else None
    metadata = json.loads(str(expert_data["metadata"])) if expert_data is not None else None
    oracle_episode = None

    class OracleTransport:
        def __init__(self, *unused):
            self.chunk_index = 0

        def get_server_metadata(self):
            return metadata

        def predict_action(self, request):
            index = self.chunk_index * args.execute_horizon
            key = f"episode_{oracle_episode}_{'raw' if args.raw_expert else 'roundtrip'}"
            actions = expert_data[key]
            indices = np.minimum(np.arange(index, index + 16), len(actions) - 1)
            self.chunk_index += 1
            return {"actions": actions[indices][None]}

        def close(self):
            pass

    if args.mode == "expert":
        adapter.WebsocketClientPolicy = OracleTransport
    records = []
    wanted = {int(x) for x in args.episodes.split(",")}
    scenes = SCENES
    if args.scene_manifest:
        registered = json.loads(args.scene_manifest.read_text())
        scenes = [
            (r["training_episode"], r["source_episode"], r["simulator_seed"])
            for r in registered["policy"]["selected_episodes"]
        ]
    for episode, source_episode, seed in scenes:
        if episode not in wanted:
            continue
        oracle_episode = episode
        model = adapter.ModelClient(args.checkpoint, port=args.port, execute_horizon=args.execute_horizon)
        if args.match_training_image_size:
            model.preserve_native_images = False
        if args.diffusion_seed is not None:
            model.diffusion_seed = args.diffusion_seed
        env = make_env("rotate_qrcode")
        pipe = None
        trace = []
        try:
            env.setup_demo(now_ep_num=0, seed=seed, is_test=True, **config)
            obs = env.get_obs()
            frame = obs["observation"]["head_camera"]["rgb"]
            image_errors = {}
            with h5py.File(source / f"data/episode{source_episode}.hdf5") as file:
                for cam in ("head_camera", "left_camera", "right_camera"):
                    image = Image.open(io.BytesIO(bytes(file[f"observation/{cam}/rgb"][0]))).convert("RGB")
                    actual = np.asarray(Image.fromarray(obs["observation"][cam]["rgb"]).resize(image.size))
                    image_errors[cam] = float(np.abs(actual.astype(float) - np.asarray(image)).mean())
                state_error = float(np.max(np.abs(obs["joint_action"]["vector"] - file["joint_action/vector"][0])))
                source_q = file["joint_action/vector"][:]
                closed = np.flatnonzero(source_q[:, 13] < 0.5)
                expected_close_step = int(closed[0]) - 1 if len(closed) else None
                expert_grasp_ee = file["endpose/right_endpose"][int(closed[0]), :3] if len(closed) else None
            expected = int(info[f"episode_{source_episode}"]["info"]["{A}"].split("base")[-1])
            if int(env.model_id) != expected or state_error > 1e-6 or image_errors["head_camera"] > 6:
                raise RuntimeError(
                    f"Scene mismatch: episode={episode}, model={env.model_id}/{expected}, state={state_error},"
                    f" RGB={image_errors}"
                )
            env.set_instruction("rotate_qrcode")
            model.reset("rotate_qrcode")
            video = output / f"{args.variant}_episode_{episode}_seed_{seed}.mp4"
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
                    str(video),
                ],
                stdin=subprocess.PIPE,
            )
            env.eval_video_path = str(output)
            env._set_eval_video_ffmpeg(pipe)
            initial = env.qrcode.get_pose().p.tolist()
            topp_lengths = []
            for planner in (env.robot.left_mplib_planner, env.robot.right_mplib_planner):
                original = planner.TOPP

                def wrap(*values, _original=original, topp_lengths=topp_lengths, **kwargs):
                    result = _original(*values, **kwargs)
                    topp_lengths.append(len(result[1]))
                    return result

                planner.TOPP = wrap
            limit = len(expert_data[f"episode_{episode}_raw"]) if args.mode == "expert" else env.step_lim
            started = time.monotonic()
            for step in range(limit):
                obs = env.get_obs()
                env.now_obs = obs
                example = dict(
                    image=[obs["observation"][cam]["rgb"] for cam in ("head_camera", "left_camera", "right_camera")],
                    state=obs["joint_action"]["vector"],
                    lang="rotate_qrcode",
                )
                topp_lengths.clear()
                action = model.step(example, step=step)
                before = np.asarray(env.get_arm_pose("right")).tolist()
                env.take_action(action, action_type="qpos")
                commanded = np.asarray(env.robot.get_left_arm_jointState() + env.robot.get_right_arm_jointState())
                real = np.asarray(env.robot.get_left_arm_real_jointState() + env.robot.get_right_arm_real_jointState())
                contacts = set()
                for contact in env.scene.get_contacts():
                    names = [body.entity.name for body in contact.bodies]
                    if env.qrcode.actor.name in names:
                        contacts.update(name for name in names if name in env.robot.gripper_name)
                trace.append(
                    dict(
                        step=step,
                        action=action.tolist(),
                        state_before=np.asarray(example["state"]).tolist(),
                        command_after=commanded.tolist(),
                        real_after=real.tolist(),
                        ee_before=before,
                        ee_after=np.asarray(env.get_arm_pose("right")).tolist(),
                        object_pose=np.r_[env.qrcode.get_pose().p, env.qrcode.get_pose().q].tolist(),
                        object_gripper_contacts=sorted(contacts),
                        closed_two_finger_contact=bool(len(contacts) >= 2 and commanded[13] < 0.5),
                        object_lift_m=float(env.qrcode.get_pose().p[2] - initial[2]),
                        ee_error_to_expert_grasp_m=(
                            float(np.linalg.norm(np.asarray(env.get_arm_pose("right"))[:3] - expert_grasp_ee))
                            if expert_grasp_ee is not None
                            else None
                        ),
                        right_tracking_rmse=float(np.sqrt(np.mean((real[7:13] - action[7:13]) ** 2))),
                        controller_duration_seconds=max(topp_lengths, default=50) / 250,
                    )
                )
                if env.eval_success:
                    break
            first_close = next((x["step"] for x in trace if x["action"][13] < 0.5), None)
            record = dict(
                mode=args.mode,
                variant=args.variant,
                checkpoint=args.checkpoint,
                training_episode=episode,
                source_episode=source_episode,
                seed=seed,
                scene_verified=True,
                initial_image_mae=image_errors,
                initial_state_max_error=state_error,
                policy_image_size=(
                    list(model.image_size) if args.match_training_image_size else list(frame.shape[:2][::-1])
                ),
                match_training_image_size=args.match_training_image_size,
                object_model_id=int(env.model_id),
                initial_object_position=initial,
                final_object_position=env.qrcode.get_pose().p.tolist(),
                steps=len(trace),
                success=bool(env.eval_success or env.check_success()),
                instruction="rotate_qrcode",
                execute_horizon=args.execute_horizon,
                num_ddim_steps=20,
                raw_expert=args.raw_expert,
                diffusion_seed=args.diffusion_seed,
                grasp_contact=any(x["closed_two_finger_contact"] for x in trace),
                lift_4cm=any(x["object_lift_m"] >= 0.04 for x in trace),
                max_object_lift_m=max(x["object_lift_m"] for x in trace),
                first_close_step=first_close,
                expert_first_close_step=expected_close_step,
                gripper_timing_error_steps=(
                    first_close - expected_close_step
                    if first_close is not None and expected_close_step is not None
                    else None
                ),
                ee_error_at_first_close_m=(
                    trace[first_close]["ee_error_to_expert_grasp_m"] if first_close is not None else None
                ),
                outcome_definitions=(
                    "Grasp contact: closed gripper with object contacting two distinct gripper bodies. Lift: object"
                    " actor origin rises >=4cm. Full success: simulator task criterion. Timing uses source command t+1;"
                    " EE reference is recorded first closure."
                ),
                mean_right_tracking_rmse=float(np.mean([x["right_tracking_rmse"] for x in trace])),
                mean_controller_duration_seconds=float(np.mean([x["controller_duration_seconds"] for x in trace])),
                elapsed_seconds=time.monotonic() - started,
                video=str(video),
            )
            records.append(record)
            (output / f"{args.variant}_results.json").write_text(json.dumps(records, indent=2) + "\n")
            (output / f"{args.variant}_episode_{episode}_trace.json").write_text(json.dumps(trace) + "\n")
            print(json.dumps(record), flush=True)
        finally:
            if pipe is not None:
                env._del_eval_video_ffmpeg()
                if pipe.wait(timeout=30):
                    raise RuntimeError("Video encoding failed")
                env.eval_video_path = None
            env.close_env(clear_cache=True)
            model.client.close()


if __name__ == "__main__":
    main()
