"""Recover initial object labels from the exact recorded RoboTwin scene seeds."""

import argparse
import io
import json
import sys
from pathlib import Path

import h5py
import numpy as np
from PIL import Image

parser = argparse.ArgumentParser()
parser.add_argument("--output-dir", required=True)
parser.add_argument("--limit", type=int, default=100)
args = parser.parse_args()
out = Path(args.output_dir)
out.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, "/export/home/haoliang/projects/RoboTwin_PVG")
from scripts.eval_policy_xpolicylab import (  # noqa: E402
    class_decorator,
    load_task_args,
)

src = Path("/export/home/haoliang/projects/RoboTwin/data/rotate_qrcode/demo_clean")
seeds = [int(s) for s in (src / "seed.txt").read_text().split()]
info = json.loads((src / "scene_info.json").read_text())
config, _ = load_task_args(
    {
        "task_name": "rotate_qrcode",
        "task_config": "demo_clean",
        "policy_name": "diagnostic",
        "ckpt_name": "feature_probe",
    }
)
config.update(render_freq=0, save_data=False, eval_mode=True, eval_video_log=False)
rows = []
for ep in range(min(len(seeds), args.limit)):
    if (out / f"scene_{ep:03d}.npz").exists():
        rows.append(json.loads((out / f"scene_{ep:03d}.json").read_text()))
        continue
    env = class_decorator("rotate_qrcode")
    try:
        env.setup_demo(now_ep_num=0, seed=seeds[ep], is_test=True, **config)
        obs = env.get_obs()
        rgb = []
        with h5py.File(src / f"data/episode{ep}.hdf5") as f:
            for name in ["head_camera", "left_camera", "right_camera"]:
                im = Image.open(io.BytesIO(bytes(f[f"observation/{name}/rgb"][0]))).convert("RGB")
                rgb.append(np.asarray(im))
            state = f["joint_action/vector"][0]
        sim = Image.fromarray(obs["observation"]["head_camera"]["rgb"]).resize((rgb[0].shape[1], rgb[0].shape[0]))
        mae = float(np.abs(np.asarray(sim).astype(float) - rgb[0]).mean())
        expected = int(info[f"episode_{ep}"]["info"]["{A}"].split("base")[-1])
        state_error = float(np.max(np.abs(state - obs["joint_action"]["vector"])))
        verified = env.model_id == expected and state_error < 1e-6 and mae < 6
        p = env.qrcode.get_pose()
        row = {
            "episode": ep,
            "seed": seeds[ep],
            "verified": bool(verified),
            "image_mae": mae,
            "state_error": state_error,
            "model_id": int(env.model_id),
            "position": p.p.tolist(),
            "quaternion": p.q.tolist(),
        }
        np.savez_compressed(out / f"scene_{ep:03d}.npz", images=np.stack(rgb), position=p.p, state=state, quaternion=p.q)
        (out / f"scene_{ep:03d}.json").write_text(json.dumps(row))
        rows.append(row)
        print(json.dumps(row), flush=True)
    finally:
        env.close_env(clear_cache=True)
(out / "scenes.json").write_text(json.dumps(rows, indent=2))
print("COMPLETED", len(rows), "verified", sum(r["verified"] for r in rows), flush=True)
