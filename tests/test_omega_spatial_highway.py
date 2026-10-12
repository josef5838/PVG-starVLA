"""Baseline equivalence, spatial ordering, useful gradients and inference KV reuse."""

from unittest.mock import patch

import torch
from test_omega_camera_stream_policy import example, policy, prepare_images


def prepare_patch_images(model, rows):
    small = prepare_images(model, rows)
    return small[:, :, :, :1, :1].expand(-1, -1, -1, 16, 16).contiguous()


def make_pair():
    torch.manual_seed(42)
    baseline = policy("direct", "joint24")
    rng = torch.get_rng_state()
    torch.manual_seed(42)
    highway = policy("direct", "joint24", highway=True)
    torch.testing.assert_close(rng, torch.get_rng_state(), rtol=0, atol=0)
    for name, value in baseline.state_dict().items():
        torch.testing.assert_close(value, highway.state_dict()[name], rtol=0, atol=0)
    # A nonzero shared output head makes parity and sensitivity checks meaningful.
    torch.nn.init.normal_(baseline.action_model.net.final_layer.linear.weight, std=0.02)
    highway.action_model.net.final_layer.load_state_dict(baseline.action_model.net.final_layer.state_dict())
    return baseline.eval(), highway.eval()


@patch("starVLA.model.framework.VLM4A.Omega512DiT.Omega512DiT._prepare_images", prepare_patch_images)
def test_current_patch_order_batch_isolation_and_zero_gate_baseline_parity():
    torch.set_num_threads(2)
    baseline, highway = make_pair()
    batch = [example(), example(5)]
    z, registers = baseline._condition(batch)
    hz, memory = highway._condition(batch)
    assert memory["registers"].shape == (2, 408, 16)
    assert memory["spatial"].shape == (2, 3, 16)
    assert highway._current_patch_tokens.shape == (2, 3, 1, 2048)
    calls = highway.omega.aggregator.calls
    assert len(calls) == 1 and calls[0].shape[:2] == (2, 24)
    layers, _ = highway.omega.aggregator(calls[0])
    expected = layers[1][:, -3:, 17:]
    torch.testing.assert_close(highway._current_patch_tokens, expected)
    torch.testing.assert_close(z, hz, rtol=0, atol=0)
    torch.testing.assert_close(registers, memory["registers"], rtol=0, atol=0)
    x, t = torch.randn(2, 16, 14), torch.tensor([20, 30])
    a = baseline.action_model.net(x, t, z, baseline.action_model.net.prepare_visual(registers))
    b = highway.action_model.net(x, t, hz, highway.action_model.net.prepare_visual(memory))
    torch.testing.assert_close(a, b, rtol=0, atol=0)
    _, alone = highway._condition([batch[0]])
    torch.testing.assert_close(memory["spatial"][0], alone["spatial"][0])
    highway.action_model.net.spatial_gates.data.fill_(0.2)
    c = highway.action_model.net(x, t, hz, highway.action_model.net.prepare_visual(memory))
    assert not torch.allclose(a, c)
    highway.highway_active = False
    dz, dm = highway._condition(batch)
    disabled = highway.action_model.net(x, t, dz, highway.action_model.net.prepare_visual(dm))
    torch.testing.assert_close(a, disabled, rtol=0, atol=0)


@patch("starVLA.model.framework.VLM4A.Omega512DiT.Omega512DiT._prepare_images", prepare_patch_images)
def test_gate_opens_and_spatial_gradients_flow_while_omega_stays_frozen():
    _, model = make_pair()
    model.train()
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=3e-4)
    first = model([example()])["action_loss"]
    first.backward()
    assert torch.isfinite(first)
    assert model.action_model.net.spatial_gates.grad.abs().sum() > 0
    assert model.spatial_proj.weight.grad.abs().sum() == 0
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    second = model([example()])["action_loss"]
    second.backward()
    assert torch.isfinite(second)
    assert model.spatial_proj.weight.grad.abs().sum() > 0
    assert model.spatial_position_proj[0].weight.grad.abs().sum() > 0
    assert model.action_model.net.spatial_attentions[0].key_value.weight.grad.abs().sum() > 0
    assert all(p.grad is None for p in model.omega.parameters())


@patch("starVLA.model.framework.VLM4A.Omega512DiT.Omega512DiT._prepare_images", prepare_patch_images)
def test_sampling_reuses_spatial_kv_and_all_visual_mask_removes_highway():
    _, model = make_pair()
    calls = [0] * len(model.action_model.net.spatial_attentions)
    hooks = []
    for slot, attention in enumerate(model.action_model.net.spatial_attentions):
        def count(module, inputs, output, slot=slot):
            calls[slot] += 1
        hooks.append(attention.key_value.register_forward_hook(count))
    actions = model.predict_action([example()], num_ddim_steps=2, diffusion_seed=4000)["normalized_actions"]
    for hook in hooks:
        hook.remove()
    assert actions.shape == (1, 16, 14)
    assert calls == [1] * len(calls)
    before = len(model.omega.aggregator.calls)
    _, memory = model._condition([example()], visual_conditioning=False)
    assert isinstance(memory, torch.Tensor) and torch.count_nonzero(memory) == 0
    assert len(model.omega.aggregator.calls) == before
