"""Capture matched three-camera object translations and score predicted TCP responses."""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
from robotwin_omega_intervention_scenes import setup_runtime
from robotwin_stream_scene_reference import verify_scene
from rotate_qrcode_metrics import task_geometry

ORDER = [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12, 6, 13]


def tcp_fk(env, actions, initial_qpos):
    """World-frame TCP positions from commands; no scene/physics step."""
    robot = env.robot
    entity = robot.right_entity
    saved = entity.get_qpos().copy()
    indices = [robot.right_active_joints.index(j) for j in robot.right_arm_joints]
    poses = []
    try:
        for action in actions:
            qpos = initial_qpos.copy()
            qpos[indices] = action[6:12]
            entity.set_qpos(qpos)
            poses.append(robot.get_right_tcp_pose())
    finally:
        entity.set_qpos(saved)
    return np.asarray(poses, dtype=float)


def capture(manifest, out, runtime):
    sapien, renderer, make_env, config = runtime
    out.mkdir(exist_ok=False)
    records, contact_rows = [], []
    with np.load(manifest["scene_reference_data"]) as references:
        for scene in manifest["rollout_scenes"]:
            base_state = base_left = base_right = base_images = None
            tiles = []
            for case in manifest["position_cases"]:
                env = make_env(manifest["task"])
                try:
                    env.setup_demo(now_ep_num=0, seed=scene["seed"], is_test=True, **config)
                    actor, arm = task_geometry(env)
                    obs = env.get_obs()
                    audit = verify_scene(scene, references, obs, env.model_id, arm,
                                         actor.get_pose().p, actor.get_pose().q,
                                         manifest["camera_order"],
                                         manifest["initial_image_validation"]["grayscale_mae_limit"])
                    original = actor.get_pose()
                    offset = np.asarray(case["offset_m"], dtype=float)
                    position = original.p.astype(float) + offset
                    assert 0.05 <= position[0] <= 0.25 and -0.2 <= position[1] <= 0
                    left = env.robot.left_entity.get_qpos().copy()
                    right = env.robot.right_entity.get_qpos().copy()
                    state = np.asarray(obs["joint_action"]["vector"]).copy()
                    if base_state is None:
                        base_state, base_left, base_right = state.copy(), left.copy(), right.copy()
                    np.testing.assert_array_equal(state, base_state)
                    np.testing.assert_array_equal(left, base_left)
                    np.testing.assert_array_equal(right, base_right)
                    actor.actor.set_pose(sapien.Pose(position, original.q))
                    obs = env.get_obs()  # Rerender; no robot dynamics advance.
                    np.testing.assert_array_equal(obs["joint_action"]["vector"], base_state)
                    np.testing.assert_array_equal(env.robot.left_entity.get_qpos(), base_left)
                    np.testing.assert_array_equal(env.robot.right_entity.get_qpos(), base_right)
                    np.testing.assert_allclose(actor.get_pose().p, position, rtol=0, atol=1e-7)
                    np.testing.assert_allclose(actor.get_pose().q, original.q, rtol=0, atol=1e-7)
                    images = np.stack([obs["observation"][c]["rgb"] for c in manifest["camera_order"]])
                    assert images.shape == (3, 480, 640, 3) and images.dtype == np.uint8
                    if base_images is None:
                        base_images = images.copy()
                    pixel_mae = np.abs(images.astype(float) - base_images).mean(axis=(1, 2, 3))
                    if np.any(offset):
                        assert pixel_mae.max() > 0, "Object movement did not change rendered images"
                    directory = out / f"episode_{scene['episode']}" / case["name"]
                    directory.mkdir(parents=True)
                    for index, image in enumerate(images):
                        Image.fromarray(image).save(directory / f"camera_{index}.png")
                    tile = Image.fromarray(images[0]).resize((320, 240))
                    ImageDraw.Draw(tile).text((5, 5), f"ep {scene['episode']} {case['name']}", fill="red")
                    tiles.append(tile)
                    sampled = []

                    def record(env=env, sampled=sampled):
                        sampled.append(np.asarray(env.robot.get_left_arm_jointState()
                                                  + env.robot.get_right_arm_jointState(), dtype=float))

                    env._take_picture = record
                    expert_error = None
                    try:
                        env.play_once()
                    except Exception as error:
                        # Keep registered failed expert cases visible, without outcome selection.
                        expert_error = repr(error)
                    expert = np.asarray(sampled)
                    target = expert[1:17, ORDER] if len(expert) >= 17 else np.empty((0, 14))
                    expert_tcp = tcp_fk(env, target, right) if len(target) else np.empty((0, 7))
                    file = directory / "scene.npz"
                    np.savez_compressed(file, images=images, raw_state=state, initial_right_qpos=right,
                                        target=target, expert_tcp=expert_tcp)
                    row = dict(episode=scene["episode"], split=scene["split"], seed=scene["seed"],
                               name=case["name"], offset_m=offset.tolist(), position=position.tolist(),
                               quaternion=original.q.tolist(), model_id=int(env.model_id), arm=str(arm),
                               state_max_difference=0.0, physical_qpos_max_difference=0.0,
                               image_mae_from_base=pixel_mae.tolist(), base_scene_audit=audit,
                               expert_success=bool(env.check_success()) if expert_error is None else False,
                               expert_plan_success=bool(env.plan_success), expert_error=expert_error,
                               expert_frames=len(expert), valid_expert_chunk=len(target) == 16,
                               scene_file=str(file), scene_sha256=hashlib.sha256(file.read_bytes()).hexdigest())
                    records.append(row)
                    (out / "scenes.json").write_text(json.dumps(records, indent=2) + "\n")
                    print(json.dumps(row), flush=True)
                finally:
                    env.close_env(clear_cache=True)
            contact_rows.append(tiles)
    canvas = Image.new("RGB", (320 * len(manifest["position_cases"]), 240 * len(contact_rows)))
    for y, tiles in enumerate(contact_rows):
        for x, tile in enumerate(tiles):
            canvas.paste(tile, (x * 320, y * 240))
    canvas.save(out / "object_position_contact_sheet.jpg")


def score(manifest, out, runtime):
    from omega_object_position_metrics import response_metrics

    _, renderer, make_env, config = runtime
    scenes = json.loads((out / "scenes/scenes.json").read_text())
    lookup = {(s["episode"], s["name"]): s for s in scenes}
    report = json.loads((out / "predictions.json").read_text())
    groups = {}
    for row in report["records"]:
        groups.setdefault(row["episode"], []).append(row)
    for episode, rows in groups.items():
        env = make_env(manifest["task"])
        try:
            env.setup_demo(now_ep_num=0, seed=lookup[(episode, "base")]["seed"], is_test=True, **config)
            baselines = {}
            for row in rows:
                scene = lookup[(episode, row["case"])]
                with np.load(scene["scene_file"]) as data:
                    predicted = tcp_fk(env, np.asarray(row["actions"]), data["initial_right_qpos"])
                    expert = data["expert_tcp"].copy()
                row["predicted_tcp_world"] = predicted.tolist()
                row["expert_success"] = scene["expert_success"]
                key = (row["condition"], row["noise_seed"])
                if row["case"] == "base":
                    baselines[key] = (predicted, expert, scene["expert_success"])
                else:
                    base_pred, base_expert, base_success = baselines[key]
                    row["responses"] = {}
                    for horizon in (8, 16):
                        actual = predicted[horizon - 1, :3] - base_pred[horizon - 1, :3]
                        metrics = response_metrics(actual, np.asarray(scene["offset_m"]))
                        metrics["predicted_tcp_delta_m"] = actual.tolist()
                        if len(expert) == 16 and len(base_expert) == 16:
                            expected = expert[horizon - 1, :3] - base_expert[horizon - 1, :3]
                            metrics["expert_tcp_delta_m"] = expected.tolist()
                            metrics["vs_expert"] = response_metrics(actual, expected)
                            metrics["expert_pair_successful"] = bool(base_success and scene["expert_success"])
                        row["responses"][str(horizon)] = metrics
                if len(expert) == 16:
                    row["tcp_expert_rmse_m"] = float(np.sqrt(np.mean(np.sum((predicted[:, :3] - expert[:, :3]) ** 2, axis=1))))
            print(f"FK scoring episode {episode} complete", flush=True)
        finally:
            env.close_env(clear_cache=True)
    report["status"] = "completed"
    report["expert_successes"] = sum(s["expert_success"] for s in scenes)
    report["total_captured_cases"] = len(scenes)
    report["fk_protocol"] = "World-frame TCP static forward kinematics at command 8 and 16; no policy physics rollout. Expert t+1..t+16 commands; successful and failed expert cases retained."
    summary = {}
    for condition in manifest["conditions"]:
        name = condition["name"]
        summary[name] = {}
        for horizon in (8, 16):
            metrics = [r["responses"][str(horizon)] for r in report["records"]
                       if r["condition"] == name and "responses" in r]
            summary[name][str(horizon)] = dict(comparisons=len(metrics),
                mean_tcp_change_m=float(np.mean([m["actual_norm_m"] for m in metrics])),
                mean_object_direction_gain=float(np.mean([m["gain"] for m in metrics])),
                opposing_object_direction=sum(m["gain"] < 0 for m in metrics),
                negligible_tcp_response=sum(m["actual_norm_m"] < 1e-4 for m in metrics),
                mean_response_error_to_object_shift_m=float(np.mean([m["response_error_m"] for m in metrics])))
            expert = [m["vs_expert"] for m in metrics if m.get("expert_pair_successful")
                      and m["vs_expert"]["expected_norm_m"] >= 0.001]
            summary[name][str(horizon)]["successful_expert_pairs_with_response_1mm"] = len(expert)
            summary[name][str(horizon)]["opposing_expert_direction"] = sum(m["gain"] < 0 for m in expert)
            summary[name][str(horizon)]["mean_error_to_expert_response_m"] = (
                float(np.mean([m["response_error_m"] for m in expert])) if expert else None)
    report["summary"] = summary
    (out / "object_position_comparison.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(dict(status="completed", expert_successes=report["expert_successes"], summary=summary)), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, required=True)
    parser.add_argument("--phase", choices=("capture", "score"), required=True)
    args = parser.parse_args()
    manifest = json.loads((args.study / "manifest.json").read_text())
    out = Path(manifest["result_root"])
    runtime = setup_runtime(Path(manifest["simulator_root"]), 0, native_camera_contract=True)
    capture(manifest, out / "scenes", runtime) if args.phase == "capture" else score(manifest, out, runtime)


if __name__ == "__main__":
    main()
