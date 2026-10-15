"""Check route masking and deterministic sampling using the real final checkpoint."""

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import torch

from deployment.model_server.policy_wrapper import PolicyServerWrapper


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, required=True)
    parser.add_argument("--variant", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads((args.study / "manifest.json").read_text())
    run = next(r for r in manifest["runs"] if r["variant"] == args.variant)
    overrides = [f"framework.state_conditioning={run['state_conditioning']}",
                 f"framework.visual_conditioning={str(run['visual_conditioning']).lower()}"]
    torch.set_num_threads(4)
    wrapper = PolicyServerWrapper(run["checkpoint"], use_bf16=True, config_overrides=overrides)
    model = wrapper._framework
    assert wrapper.metadata["state_conditioning"] == run["state_conditioning"]
    assert wrapper.metadata["visual_conditioning"] == run["visual_conditioning"]
    with np.load(manifest["scene_reference_data"]) as reference:
        prefix = f"episode_{manifest['rollout_scenes'][0]['episode']}"
        images = reference[prefix + "_images"].copy()
        state = reference[prefix + "_state"].copy()
    # Same simulator -> training order conversion as the deployment adapter.
    state = state[[0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12, 6, 13]]
    example = dict(image=list(images), image_history=[[im.copy() for _ in range(8)] for im in images],
                   state=state[None], lang=manifest["task"])
    prepared = dict(example, state=wrapper.get_norm_processor().apply_state(example["state"]))
    calls = []
    hook = model.omega.aggregator.register_forward_pre_hook(lambda module, inputs: calls.append(1))
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        condition, memory = model._condition([prepared])
    assert condition.shape == (1, 5, 1024) and torch.isfinite(condition).all()
    if run["state_conditioning"] == "none":
        assert torch.count_nonzero(condition[:, :4]) == 0
    else:
        assert torch.count_nonzero(condition[:, :4]) > 0, "Full-state inference masked every state token"
    assert torch.count_nonzero(condition[:, 4:]) > 0
    if not run["visual_conditioning"]:
        assert isinstance(memory, torch.Tensor) and torch.count_nonzero(memory) == 0
        assert not calls, "Vision-disabled inference called Omega"
    else:
        assert len(calls) == 1
    del condition, memory

    def infer(row):
        result = wrapper.predict_action([row], num_ddim_steps=20, diffusion_seed=4000)["actions"]
        assert result.shape == (1, 16, 14) and np.isfinite(result).all()
        return result

    original = infer(example)
    repeated = infer(example)
    np.testing.assert_allclose(original, repeated, rtol=0, atol=1e-6)
    state_error = image_error = None
    if run["state_conditioning"] == "none":
        changed = copy.deepcopy(example)
        changed["state"][0, 6:12] += 0.03
        altered = infer(changed)
        state_error = float(np.max(np.abs(original - altered)))
        np.testing.assert_allclose(original, altered, rtol=0, atol=1e-6)
    if not run["visual_conditioning"]:
        changed = copy.deepcopy(example)
        changed["image"] = [255 - im for im in changed["image"]]
        changed["image_history"] = [[255 - im for im in view] for view in changed["image_history"]]
        altered = infer(changed)
        image_error = float(np.max(np.abs(original - altered)))
        np.testing.assert_allclose(original, altered, rtol=0, atol=1e-6)
        assert not calls, "Vision-disabled sampling called Omega"
    hook.remove()
    report = dict(status="passed", variant=args.variant, checkpoint=run["checkpoint"], checkpoint_step=run["checkpoint_step"],
                  state_conditioning=run["state_conditioning"], visual_conditioning=run["visual_conditioning"],
                  diffusion_seed=4000, num_ddim_steps=20, input_episode=manifest["rollout_scenes"][0]["episode"],
                  repeat_max_action_difference=float(np.max(np.abs(original - repeated))),
                  masked_state_max_action_difference=state_error, masked_images_max_action_difference=image_error,
                  omega_calls=len(calls), first_action_chunk=original[0].tolist(),
                  note="Perturbations verify masking, not useful grounding or rollout quality.")
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
