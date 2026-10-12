"""Causal stream histories must advance while an action chunk is being executed."""

import unittest
from unittest.mock import patch

import numpy as np

from examples.simBenchmarks.Robotwin.eval_files.model2robotwin_interface import ModelClient


class HistoryTests(unittest.TestCase):
    def test_stream_metadata_requests_training_resolution_and_history(self):
        from deployment.model_server.policy_wrapper import PolicyServerWrapper

        wrapper = PolicyServerWrapper.__new__(PolicyServerWrapper)
        wrapper._model_cfg = {
            "framework": {"name": "OmegaCameraStreamDiT", "streams": {"history_length": 8}},
            "datasets": {
                "vla_data": {"obs_image_size": [640, 480], "preserve_native_images": True, "include_state": True}
            },
            "evaluation": {"execute_horizon": 8, "diffusion_seed": 4000},
        }
        wrapper._ckpt_path = "unused"
        wrapper._action_chunk_size = 16
        wrapper._available_unnorm_keys = []
        wrapper._default_unnorm_key = None
        metadata = wrapper.metadata
        self.assertEqual(metadata["camera_history_length"], 8)
        self.assertEqual(metadata["training_obs_image_size"], [640, 480])
        self.assertFalse(metadata["preserve_native_images"])
        self.assertEqual(metadata["evaluation_diffusion_seed"], 4000)

    def test_history_observes_every_step_pads_and_resets(self):
        class Transport:
            def __init__(self, *args):
                self.requests = []

            def get_server_metadata(self):
                return dict(
                    action_chunk_size=16,
                    include_state=True,
                    execute_horizon=8,
                    camera_history_length=8,
                    camera_history_stride=1,
                    training_obs_image_size=[640, 480],
                    preserve_native_images=False,
                )

            def predict_action(self, request):
                self.requests.append(request)
                return {"actions": np.zeros((1, 16, 14))}

        with patch(
            "examples.simBenchmarks.Robotwin.eval_files.model2robotwin_interface.WebsocketClientPolicy", Transport
        ):
            model = ModelClient("unused")
            for step in range(9):
                example = dict(
                    image=[np.full((24, 32, 3), step + c * 20, np.uint8) for c in range(3)],
                    state=np.zeros(14),
                    lang="adjust_bottle",
                )
                model.step(example, step)
            self.assertEqual(len(model.client.requests), 2)
            first = model.client.requests[0]["examples"][0]["image_history"]
            second = model.client.requests[1]["examples"][0]["image_history"]
            for c in range(3):
                self.assertEqual([int(x[0, 0, 0]) for x in first[c]], [c * 20] * 8)
                self.assertEqual([int(x[0, 0, 0]) for x in second[c]], list(range(1 + c * 20, 9 + c * 20)))
                self.assertEqual(second[c][0].shape, (480, 640, 3))
            model.reset("adjust_bottle")
            self.assertEqual(len(model.camera_history), 0)
            model.step(example, 0)
            self.assertTrue(
                all(int(x[0, 0, 0]) == 8 for x in model.client.requests[-1]["examples"][0]["image_history"][0])
            )


if __name__ == "__main__":
    unittest.main()
