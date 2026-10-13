"""Reject reset/data mismatches before labeling a rollout as a seen/test scene."""

import numpy as np
import pytest

from tools.robotwin_stream_scene_reference import verify_scene


def inputs():
    cameras = ["head_camera", "left_camera", "right_camera"]
    images = np.stack([np.full((480,640,3), level, np.uint8) for level in (20,80,140)])
    state = np.arange(14, dtype=np.float32)/10
    scene = dict(episode=2, seed=91, arm="right", model_id=1,
                 position=[0.1,-0.05,0.74], quaternion=[1,0,0,0])
    references = {"episode_2_state":state.copy(), "episode_2_images":images.copy()}
    obs = {"joint_action":{"vector":state.copy()}, "observation":{
        name:{"rgb":images[index].copy()} for index,name in enumerate(cameras)}}
    return dict(scene=scene, references=references, observation=obs, model_id=1, arm="right",
                position=scene["position"], quaternion=[-1,0,0,0], camera_order=cameras)


def test_exact_reset_and_equivalent_quaternion_sign_pass():
    report = verify_scene(**inputs())
    assert report["initial_state_max_error"] == 0
    assert report["initial_quaternion_error_deg"] == 0
    assert max(report["initial_grayscale_mae"].values()) == 0


@pytest.mark.parametrize("mismatch", ["state", "camera_order", "position", "orientation", "model", "arm"])
def test_wrong_scene_or_data_contract_is_rejected(mismatch):
    data = inputs()
    if mismatch == "state":
        data["observation"]["joint_action"]["vector"][8] += 0.01
    elif mismatch == "camera_order":
        data["camera_order"] = data["camera_order"][::-1]
    elif mismatch == "position":
        data["position"] = [0.15,-0.05,0.74]
    elif mismatch == "orientation":
        data["quaternion"] = [0,0,0,1]
    elif mismatch == "model":
        data["model_id"] = 0
    elif mismatch == "arm":
        data["arm"] = "left"
    with pytest.raises(ValueError, match="Scene mismatch"):
        verify_scene(**data)


def test_lower_resolution_render_is_rejected():
    data = inputs()
    data["observation"]["observation"]["head_camera"]["rgb"] = np.zeros((240,320,3), np.uint8)
    with pytest.raises(ValueError, match="RGB contract"):
        verify_scene(**data)
