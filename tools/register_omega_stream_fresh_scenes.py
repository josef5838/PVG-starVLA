"""Reject unstable simulator setups, without using policy/expert task outcomes."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    path = args.manifest.resolve()
    m = json.loads(path.read_text())
    sys.path.insert(0, str(ROOT / "tools"))
    from robotwin_omega_intervention_scenes import setup_runtime

    _, renderer, make_env, config = setup_runtime(Path("/export/home/haoliang/projects/RoboTwin_PVG"), 0, task=m["task"])
    from envs.utils.create_actor import UnStableError

    accepted = []
    for scene in m["rollout_scenes"]:
        if scene["split"] != "unseen_eval":
            continue
        start = scene.get("seed_start", scene["seed"])
        rejected = []
        for seed in range(start, start + 100):
            env = make_env(m["task"])
            try:
                env.setup_demo(now_ep_num=0, seed=seed, is_test=True, **config)
                obs = env.get_obs()
                row = dict(
                    seed_start=start,
                    seed=seed,
                    arm="right" if env.qpose_tag else "left",
                    model_id=int(env.model_id),
                    position=env.bottle.get_pose().p.tolist(),
                    initial_state=np.asarray(obs["joint_action"]["vector"]).tolist(),
                    rejected_setups=rejected,
                )
                accepted.append(row)
                scene.update(seed_start=start, seed=seed, setup_rejections=len(rejected))
                print(json.dumps(row), flush=True)
                break
            except UnStableError as error:
                rejected.append(dict(seed=seed, reason=str(error)))
                print(f"Rejected setup {seed}: {error}", flush=True)
            finally:
                env.close_env(clear_cache=True)
        else:
            raise RuntimeError(f"No stable setup after {start}")
        (path.parent / "fresh_scene_registration.json").write_text(json.dumps(accepted, indent=2) + "\n")
    m["fresh_scene_registration"] = str(path.parent / "fresh_scene_registration.json")
    m["fresh_evaluation_scope"] = (
        "Five benchmark seed starts, reject unstable setup then accept first stable seed; no expert-feasibility"
        " rejection; paired unseen-scene diagnostic."
    )
    path.write_text(json.dumps(m, indent=2) + "\n")


if __name__ == "__main__":
    main()
