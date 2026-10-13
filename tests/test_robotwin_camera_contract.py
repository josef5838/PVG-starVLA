import unittest
from types import SimpleNamespace
from tools.robotwin_camera_contract import configure_native_cameras


class CameraContractTests(unittest.TestCase):
    def configure(self, small, large):
        evaluator = SimpleNamespace(get_camera_config=lambda name: {'D435': small, 'Large_D435': large}[name])
        config = {'camera': {'head_camera_type': 'D435', 'wrist_camera_type': 'D435'}}
        return configure_native_cameras(evaluator, config)

    def test_selects_full_resolution_preserving_fov(self):
        result = self.configure({'w': 320, 'h': 240, 'fovy': 37}, {'w': 640, 'h': 480, 'fovy': 37})
        self.assertEqual(set(result['camera'].values()), {'Large_D435'})
        self.assertEqual((result['head_camera_w'], result['head_camera_h']), (640, 480))

    def test_preserves_existing_full_resolution_preset(self):
        result = self.configure({'w': 640, 'h': 480, 'fovy': 37}, None)
        self.assertEqual(set(result['camera'].values()), {'D435'})

    def test_rejects_changed_geometry(self):
        with self.assertRaisesRegex(ValueError, 'matching native'):
            self.configure({'w': 320, 'h': 240, 'fovy': 37}, {'w': 640, 'h': 480, 'fovy': 45})
