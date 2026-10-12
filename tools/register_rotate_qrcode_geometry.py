"""Audit all demo identities/initial poses and preregister 50 unseen right-arm setups."""

import argparse
import io
import json
import sys
from collections import Counter
from pathlib import Path

import h5py
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from rotate_qrcode_metrics import (  # noqa: E402
    orientation_bin,
    quaternion_error_degrees,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--mapping-only", action="store_true")
    args = parser.parse_args()
    path = args.manifest.resolve()
    m = json.loads(path.read_text())
    source = Path(m["source"])
    dataset = Path(m["dataset"])
    seeds = list(map(int, (source / "seed.txt").read_text().split()))
    metadata = json.loads((source / "scene_info.json").read_text())
    mapping = m.get("trajectory_mapping")
    if not mapping:
        import pyarrow.parquet as pq

        raw = {}
        for file in source.glob("data/episode*.hdf5"):
            with h5py.File(file) as h:
                raw[int(file.stem[7:])] = h["joint_action/vector"][:].astype(np.float32)
        mapping = []
        all_actions = []
        for file in sorted((dataset / "data").rglob("*.parquet")):
            episode = int(file.stem.split("_")[-1])
            d = pq.read_table(file, columns=["observation.state", "action"]).to_pydict()
            state = np.asarray(d["observation.state"], np.float32)
            actions = np.asarray(d["action"], np.float32)
            all_actions.append(actions)
            matches = [
                sid
                for sid, v in raw.items()
                if len(v) == len(state) + 1
                and np.allclose(v[:-1], state, rtol=0, atol=1e-6)
                and np.allclose(v[1:], actions, rtol=0, atol=1e-6)
            ]
            if len(matches) != 1:
                raise RuntimeError(f"Unverified source mapping for {episode}: {matches}")
            sid = matches[0]
            split = (
                "train"
                if episode in m["train_episode_ids"]
                else "validation" if episode in m["validation_episode_ids"] else "test"
            )
            mapping.append(
                dict(
                    episode=episode,
                    source_episode=sid,
                    seed=seeds[sid],
                    length=len(state),
                    split=split,
                    arm=metadata[f"episode_{sid}"]["info"]["{a}"],
                    model_id=int(metadata[f"episode_{sid}"]["info"]["{A}"].split("base")[-1]),
                    observation_matches_source_t=True,
                    action_matches_source_t_plus_1=True,
                )
            )
        actions = np.concatenate(all_actions)
        m["trajectory_mapping"] = mapping
        m["data_audit"] = dict(
            left_action_min=actions[:, :7].min(0).tolist(), left_action_max=actions[:, :7].max(0).tolist()
        )
        path.write_text(json.dumps(m, indent=2) + "\n")
    if len(mapping) != 100 or any(r["arm"] != "right" for r in mapping):
        raise RuntimeError("Unexpected dataset/arm coverage")
    if args.mapping_only:
        print(f"Verified all {len(mapping)} observation/action source mappings.")
        return
    from robotwin_omega_intervention_scenes import setup_runtime

    _, renderer, make_env, config = setup_runtime(Path("/export/home/haoliang/projects/RoboTwin_PVG"), 0, task=m["task"])
    from envs.utils.create_actor import UnStableError

    demo_poses = []
    fresh = []
    rejections = []
    cache = path.parent / "geometry_registration.json"
    if cache.exists():
        previous = json.loads(cache.read_text())
        demo_poses = previous.get("demo_initial_geometry", [])
        fresh = previous.get("fresh_scenes", [])
        rejections = previous.get("fresh_setup_rejections", [])

    def save():
        result = dict(
            status="registering",
            trajectory_mapping=mapping,
            demo_initial_geometry=demo_poses,
            fresh_scenes=fresh,
            fresh_setup_rejections=rejections,
            **m["data_audit"],
            selection=(
                "Fresh seeds start at 100000 in increasing order. Reject unstable setups and left-arm scenes; no"
                " expert/policy outcome filtering."
            ),
        )
        cache.write_text(json.dumps(result, indent=2) + "\n")
        return result

    done = {x["episode"] for x in demo_poses}
    # RoboTwin supports resetting the same task instance, reusing its expensive planners.
    # Initial RGB/state/model checks below still verify every reset against source recordings.
    env = make_env(m["task"])
    for row in mapping:
        if row["episode"] in done:
            continue
        try:
            env.setup_demo(now_ep_num=0, seed=row["seed"], is_test=True, **config)
            obs = env.get_obs()
            pose = env.qrcode.get_pose()
            errors = {}
            gray_errors = {}
            with h5py.File(source / f"data/episode{row['source_episode']}.hdf5") as h:
                state_error = float(
                    np.max(np.abs(np.asarray(obs["joint_action"]["vector"]) - h["joint_action/vector"][0]))
                )
                for cam in m["camera_order"]:
                    expected = Image.open(io.BytesIO(bytes(h[f"observation/{cam}/rgb"][0]))).convert("RGB")
                    actual = np.asarray(Image.fromarray(obs["observation"][cam]["rgb"]).resize(expected.size))
                    errors[cam] = float(np.abs(actual.astype(float) - np.asarray(expected)).mean())
                    gray_errors[cam] = float(
                        np.abs(
                            np.asarray(Image.fromarray(actual).convert("L")).astype(float)
                            - np.asarray(expected.convert("L"))
                        ).mean()
                    )
            if state_error > 1e-6 or max(gray_errors.values()) > 6 or int(env.model_id) != row["model_id"]:
                raise RuntimeError(f"Initial source scene mismatch {row}: state={state_error},RGB={errors}")
            offset = quaternion_error_degrees(pose.q, [0, 0, 0.707, 0.707])
            demo_poses.append(
                dict(
                    **row,
                    position=pose.p.tolist(),
                    quaternion=pose.q.tolist(),
                    initial_orientation_error_deg=quaternion_error_degrees(pose.q),
                    rotation_offset_deg=offset,
                    orientation_bin=orientation_bin(offset),
                    initial_image_mae=errors,
                    initial_grayscale_mae=gray_errors,
                    photometric_shift_flag=max(errors.values()) > 6,
                    initial_state_max_error=state_error,
                )
            )
            save()
            print(f"Verified demo {len(demo_poses)}/100: {row['episode']}", flush=True)
        finally:
            env.close_env(clear_cache=True)
    source_seeds = set(seeds)
    candidate = max([99999] + [r["seed"] for r in fresh] + [r["seed"] for r in rejections]) + 1
    while len(fresh) < m["fresh_scene_count"]:
        if candidate in source_seeds:
            candidate += 1
            continue
        try:
            env.setup_demo(now_ep_num=0, seed=candidate, is_test=True, **config)
            pose = env.qrcode.get_pose()
            if pose.p[0] < 0:
                rejections.append(dict(seed=candidate, reason="left_arm_outside_training_support"))
            else:
                obs = env.get_obs()
                offset = quaternion_error_degrees(pose.q, [0, 0, 0.707, 0.707])
                fresh.append(
                    dict(
                        split="unseen_eval",
                        episode=None,
                        source_episode=None,
                        seed=candidate,
                        arm="right",
                        model_id=int(env.model_id),
                        position=pose.p.tolist(),
                        quaternion=pose.q.tolist(),
                        initial_state=np.asarray(obs["joint_action"]["vector"]).tolist(),
                        initial_orientation_error_deg=quaternion_error_degrees(pose.q),
                        rotation_offset_deg=offset,
                        orientation_bin=orientation_bin(offset),
                    )
                )
                print(f"Registered fresh right-arm scene {len(fresh)}/50: {candidate}", flush=True)
        except UnStableError as error:
            rejections.append(dict(seed=candidate, reason="unstable_setup", detail=str(error)))
        finally:
            env.close_env(clear_cache=True)
        save()
        candidate += 1
        if candidate > 105000:
            raise RuntimeError("Fresh setup search exhausted")
    result = save()
    result["status"] = "completed"
    result["coverage"] = {}
    for split in ["train", "validation", "test", "unseen_eval"]:
        rows = fresh if split == "unseen_eval" else [x for x in demo_poses if x["split"] == split]
        positions = np.array([r["position"] for r in rows])
        angles = [r["rotation_offset_deg"] for r in rows]
        result["coverage"][split] = dict(
            episodes=len(rows),
            model_counts=dict(Counter(r["model_id"] for r in rows)),
            orientation_bins=dict(Counter(r["orientation_bin"] for r in rows)),
            position_min=positions.min(0).tolist(),
            position_max=positions.max(0).tolist(),
            rotation_offset_min_deg=min(angles),
            rotation_offset_max_deg=max(angles),
        )
    cache.write_text(json.dumps(result, indent=2) + "\n")
    m["geometry_registration"] = str(cache)
    m["trajectory_mapping"] = mapping
    m["initial_image_validation"] = dict(
        grayscale_mae_limit=6,
        rgb_mae_soft_limit=6,
        note=(
            "Source recordings have a blue tint relative to the current renderer. Fresh seed9 reproduces the same"
            " geometry/state/model with RGB MAE 6.027; RGB errors are reported, grayscale MAE checks spatial agreement."
            " No policy outcomes used."
        ),
    )
    # Ten uniformly selected training scenes plus every validation/test scene and all fresh scenes.
    chosen = set(np.asarray(m["train_episode_ids"])[np.linspace(0, 79, 10).round().astype(int)].tolist())
    m["rollout_scenes"] = [r for r in demo_poses if r["split"] != "train" or r["episode"] in chosen] + fresh
    m["fresh_scene_registration"] = str(cache)
    path.write_text(json.dumps(m, indent=2) + "\n")
    print(json.dumps(result["coverage"], indent=2), flush=True)


if __name__ == "__main__":
    main()
