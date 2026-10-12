"""Secondary geometry diagnostics; official task success is always reported separately."""

import numpy as np

TARGET_QUATERNION = np.array([0.707, 0.707, 0.0, 0.0])


def quaternion_error_degrees(actual, target=TARGET_QUATERNION):
    a, b = np.asarray(actual, dtype=float), np.asarray(target, dtype=float)
    denominator = np.linalg.norm(a) * np.linalg.norm(b)
    if denominator <= 0:
        raise ValueError("Quaternion must have nonzero norm")
    return float(np.degrees(2 * np.arccos(np.clip(abs(np.dot(a, b)) / denominator, 0, 1))))


def orientation_bin(offset_degrees):
    return "0-15deg" if offset_degrees < 15 else "15-30deg" if offset_degrees < 30 else "30deg-plus"


def task_geometry(env):
    if hasattr(env, "qrcode"):
        actor = env.qrcode
        arm = "right" if actor.get_pose().p[0] >= 0 else "left"
    else:
        actor = env.bottle
        arm = "right" if env.qpose_tag else "left"
    return actor, arm


def rotation_diagnostics(env, actor, command):
    if not hasattr(env, "qrcode"):
        return {}
    pose = actor.get_pose()
    error = quaternion_error_degrees(pose.q)
    height_ok = bool(pose.p[2] < 0.75 + env.table_z_bias)
    release = bool(env.is_left_gripper_open() and env.is_right_gripper_open())
    target = np.asarray(env.target_pose[:3])
    return dict(
        orientation_error_deg=error,
        placement_xy_error_m=float(np.linalg.norm(pose.p[:2] - target[:2])),
        placed_within_15deg=bool(height_ok and error < 15),
        both_grippers_open=release,
    )


def summarize_phases(trace, success):
    grasp = any(x["closed_two_finger_contact"] for x in trace)
    lift = any(x["object_lift_m"] >= 0.04 for x in trace)
    rotation = any(x.get("orientation_error_deg", float("inf")) < 15 and x["object_lift_m"] >= 0.02 for x in trace)
    placement = any(x.get("placed_within_15deg", False) for x in trace)
    release = any(x.get("placed_within_15deg", False) and x.get("both_grippers_open", False) for x in trace)
    failure = (
        "success"
        if success
        else (
            "grasp"
            if not grasp
            else (
                "lift"
                if not lift
                else (
                    "rotation"
                    if not rotation
                    else "placement" if not placement else "release" if not release else "official_success_tolerance"
                )
            )
        )
    )
    return dict(
        grasp=grasp,
        lift=lift,
        rotation=rotation,
        placement=placement,
        release=release,
        diagnostic_failure_phase=failure,
        rotation_threshold_deg=15,
        note="Secondary thresholds; do not replace official success or prove a stable grasp.",
    )
