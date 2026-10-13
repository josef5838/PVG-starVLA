"""Verify a simulator reset against registered geometry and local dataset RGB/state."""

import numpy as np
from PIL import Image

from tools.rotate_qrcode_metrics import quaternion_error_degrees


def verify_scene(scene, references, observation, model_id, arm, position, quaternion,
                 camera_order, grayscale_mae_limit=6):
    prefix = f"episode_{scene['episode']}"
    expected_state = references[f"{prefix}_state"]
    expected_images = references[f"{prefix}_images"]
    state_error = float(np.max(np.abs(np.asarray(observation['joint_action']['vector'])-expected_state)))
    position_error = float(np.linalg.norm(np.asarray(position)-scene['position']))
    orientation_error = quaternion_error_degrees(quaternion, scene['quaternion'])
    rgb_errors, gray_errors = {}, {}
    for index, camera in enumerate(camera_order):
        actual = np.asarray(observation['observation'][camera]['rgb'])
        expected = expected_images[index]
        if actual.shape != (480, 640, 3) or actual.dtype != np.uint8 or expected.shape != actual.shape:
            raise ValueError(f"Scene RGB contract mismatch: {camera}, {actual.shape}, {actual.dtype}")
        rgb_errors[camera] = float(np.abs(actual.astype(float)-expected).mean())
        gray_errors[camera] = float(np.abs(np.asarray(Image.fromarray(actual).convert('L')).astype(float)
                                         -np.asarray(Image.fromarray(expected).convert('L'))).mean())
    report = dict(initial_state_max_error=state_error, initial_image_mae=rgb_errors,
                  initial_grayscale_mae=gray_errors, initial_position_error_m=position_error,
                  initial_quaternion_error_deg=orientation_error,
                  photometric_shift_flag=max(rgb_errors.values()) > 6)
    if (int(model_id) != scene['model_id'] or arm != scene['arm'] or state_error > 1e-6
            or position_error > 1e-4 or orientation_error > 0.1
            or max(gray_errors.values()) > grayscale_mae_limit):
        raise ValueError(f"Scene mismatch at episode {scene['episode']}, seed {scene['seed']}: "
                         f"model={model_id}/{scene['model_id']}, arm={arm}/{scene['arm']}, {report}")
    return report
