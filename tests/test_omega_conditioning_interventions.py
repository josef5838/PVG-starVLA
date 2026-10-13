"""Meaningful route-isolation checks with a nonzero decoder output head."""

import copy
from unittest.mock import patch

import numpy as np
import torch

from test_omega_camera_stream_policy import example, policy, prepare_images


@patch("starVLA.model.framework.VLM4A.Omega512DiT.Omega512DiT._prepare_images", prepare_images)
def test_four_condition_routes_preserve_task_and_mask_only_intended_inputs():
    torch.set_num_threads(2)
    torch.manual_seed(42)
    model = policy("direct", "joint24").eval()
    torch.nn.init.normal_(model.action_model.net.final_layer.linear.weight, std=0.02)
    original = example()
    different_state = copy.deepcopy(original)
    different_state["state"] += 0.1
    different_image = example(25)
    for state, vision in (("full",True),("none",True),("full",False),("none",False)):
        model.config.framework.state_conditioning = state
        model.config.framework.visual_conditioning = vision
        z, memory = model._condition([original])
        assert torch.count_nonzero(z[:,4:]) > 0
        if state == "none":
            assert torch.count_nonzero(z[:,:4]) == 0
        if not vision:
            assert isinstance(memory,torch.Tensor) and torch.count_nonzero(memory) == 0
        before = len(model.omega.aggregator.calls)
        def sample(row):
            result = model.predict_action([row],num_ddim_steps=2,diffusion_seed=4000)["normalized_actions"]
            assert result.shape == (1,16,14) and np.isfinite(result).all()
            return result
        a, b, c = sample(original), sample(different_state), sample(different_image)
        np.testing.assert_array_equal(a,sample(original))
        assert (not np.array_equal(a,b)) == (state == "full")
        assert (not np.array_equal(a,c)) == vision
        if not vision:
            assert len(model.omega.aggregator.calls) == before
