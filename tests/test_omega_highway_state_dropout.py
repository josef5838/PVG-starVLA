"""Training removes complete state tokens; vision and task survive, eval keeps state."""

from unittest.mock import patch

import torch
from test_omega_spatial_highway import make_pair, prepare_patch_images
from test_omega_camera_stream_policy import example


@patch("starVLA.model.framework.VLM4A.Omega512DiT.Omega512DiT._prepare_images", prepare_patch_images)
def test_twenty_percent_state_dropout_preserves_task_and_both_visual_routes():
    _, model = make_pair()
    model.config.framework.state_dropout_prob = 0.2
    rows = [example(), example(5)]
    normal, visual = model._condition(rows, apply_dropout=False)
    model.train()
    # Draws on each side of p=.2 exercise one dropped and one retained example.
    with patch("torch.rand", return_value=torch.tensor([0.1, 0.9])):
        masked, masked_visual = model._condition(rows, apply_dropout=True)
    assert torch.count_nonzero(masked[0, :4]) == 0
    torch.testing.assert_close(masked[1, :4], normal[1, :4], rtol=0, atol=0)
    torch.testing.assert_close(masked[:, 4:], normal[:, 4:], rtol=0, atol=0)
    for route in ("registers", "spatial"):
        torch.testing.assert_close(masked_visual[route], visual[route], rtol=0, atol=0)
    # Masked examples cannot backpropagate through state projections/embeddings.
    masked[0, :4].sum().backward(retain_graph=True)
    assert all(projection.weight.grad.abs().sum() == 0 for projection in model.state_proj)
    model.zero_grad(set_to_none=True)
    masked[1, :4].sum().backward()
    assert all(projection.weight.grad.abs().sum() > 0 for projection in model.state_proj)
    model.eval()
    evaluated, _ = model._condition(rows, apply_dropout=False)
    torch.testing.assert_close(evaluated, normal, rtol=0, atol=0)
