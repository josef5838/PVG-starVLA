"""Learned-policy rollouts using verified original training-scene seeds."""

import argparse
import io
import json
import subprocess
import sys
from pathlib import Path

import h5py
import numpy as np
from PIL import Image

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--variant", required=True)
parser.add_argument("--port", type=int, default=5694)
parser.add_argument("--output-dir", required=True)
args = parser.parse_args()
output = Path(args.output_dir).resolve()
output.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, "/export/home/haoliang/projects/RoboTwin_PVG")
from scripts.eval_policy_xpolicylab import (  # noqa: E402
    class_decorator,
    load_task_args,
)

from examples.simBenchmarks.Robotwin.eval_files.model2robotwin_interface import (  # noqa: E402
    ModelClient,
)

source_root = Path("/export/home/haoliang/projects/RoboTwin/data/rotate_qrcode/demo_clean")
scene_info = json.loads((source_root / "scene_info.json").read_text())
config, _ = load_task_args(
    {"task_name": "rotate_qrcode", "task_config": "demo_clean", "policy_name": "starVLA", "ckpt_name": args.variant}
)
config.update(render_freq=0, save_data=False, eval_mode=True, eval_video_log=False)
model = ModelClient(args.checkpoint, port=args.port)
records = []
for episode, source_episode, seed in [(0, 51, 131), (1, 60, 162), (2, 38, 91)]:
    env = class_decorator("rotate_qrcode")
    env.setup_demo(now_ep_num=0, seed=seed, is_test=True, **config)
    observation = env.get_obs()
    frame = observation["observation"]["head_camera"]["rgb"]
    with h5py.File(source_root / f"data/episode{source_episode}.hdf5") as f:
        source = Image.open(io.BytesIO(bytes(f["observation/head_camera/rgb"][0]))).convert("RGB")
        image_error = float(
            np.abs(np.asarray(Image.fromarray(frame).resize(source.size)).astype(float) - np.asarray(source)).mean()
        )
        state_error = float(np.max(np.abs(observation["joint_action"]["vector"] - f["joint_action/vector"][0])))
    expected_model = int(scene_info[f"episode_{source_episode}"]["info"]["{A}"].split("base")[-1])
    if env.model_id != expected_model or state_error > 1e-6 or image_error > 6:
        raise RuntimeError(
            "Training scene verification failed:"
            f" episode={episode} model={env.model_id}/{expected_model} state={state_error} image={image_error}"
        )
    env.set_instruction("rotate_qrcode")
    model.reset("rotate_qrcode")
    video = output / f"{args.variant}_training_episode_{episode}_seed_{seed}.mp4"
    ffmpeg = subprocess.Popen(
        [
            str(Path(sys.executable).parent / "ffmpeg"),
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
    env._set_eval_video_ffmpeg(ffmpeg)
    action_trace = []
    initial_pose = env.qrcode.get_pose().p.tolist()
    try:
        for step in range(env.step_lim):
            observation = env.get_obs()
            env.now_obs = observation
            example = {
                "image": [
                    observation["observation"][name]["rgb"] for name in ["head_camera", "left_camera", "right_camera"]
                ],
                "state": observation["joint_action"]["vector"],
                "lang": "rotate_qrcode",
            }
            action = model.step(example, step=step)
            action_trace.append(action.tolist())
            env.take_action(action, action_type="qpos")
            if env.eval_success:
                break
        record = {
            "variant": args.variant,
            "checkpoint": args.checkpoint,
            "training_episode": episode,
            "source_episode": source_episode,
            "seed": seed,
            "scene_verified": True,
            "initial_image_mae": image_error,
            "initial_robot_state_max_error": state_error,
            "object_model_id": int(env.model_id),
            "initial_object_position": initial_pose,
            "final_object_position": env.qrcode.get_pose().p.tolist(),
            "instruction": "rotate_qrcode",
            "steps": len(action_trace),
            "success": bool(env.eval_success or env.check_success()),
            "video": str(video),
        }
        records.append(record)
        (output / f"{args.variant}_results.json").write_text(json.dumps(records, indent=2) + "\n")
        (output / f"{args.variant}_episode_{episode}_actions.json").write_text(json.dumps(action_trace) + "\n")
        print(json.dumps(record), flush=True)
    finally:
        env._del_eval_video_ffmpeg()
        env.eval_video_path = None
        env.close_env(clear_cache=True)
model.client.close()
