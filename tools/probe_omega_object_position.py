"""Matched static history/noise interventions with real A20K deployment normalization."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from deployment.model_server.policy_wrapper import PolicyServerWrapper

ORDER = [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12, 6, 13]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads((args.study / "manifest.json").read_text())
    out = Path(manifest["result_root"])
    scenes = json.loads((out / "scenes/scenes.json").read_text())
    assert len(scenes) == len(manifest["rollout_scenes"]) * len(manifest["position_cases"])
    torch.set_num_threads(4)
    report = dict(status="running", checkpoint=manifest["run"]["checkpoint"], checkpoint_step=20000,
                  source_commit=manifest["source_commit"], records=[], reproducibility_checks=[],
                  limitations="Initial-frame static interventions; no dynamics rollout. First 8 commands match execution horizon. State masking is out of training distribution. Full 16-command targeting need not equal object displacement.")
    wrapper = PolicyServerWrapper(manifest["run"]["checkpoint"], use_bf16=True)
    model = wrapper._framework
    calls = []
    hook = model.omega.aggregator.register_forward_pre_hook(lambda module, inputs: calls.append(1))
    for condition in manifest["conditions"]:
        # Supported per-call overrides leave checkpoint parameters untouched.
        for seed in manifest["noise_seeds"]:
            bases = {}
            for scene in scenes:
                file = Path(scene["scene_file"])
                assert hashlib.sha256(file.read_bytes()).hexdigest() == scene["scene_sha256"]
                with np.load(file) as data:
                    images = data["images"].copy()
                    state = data["raw_state"][ORDER].copy()
                example = dict(image=list(images), image_history=[[im.copy() for _ in range(8)] for im in images],
                               state=state[None], lang=manifest["task"])

                def infer():
                    before = len(calls)
                    value = wrapper.predict_action([example], num_ddim_steps=20, diffusion_seed=seed,
                                state_conditioning=condition["state_conditioning"],
                                visual_conditioning=condition["visual_conditioning"])["actions"][0]
                    assert value.shape == (16, 14) and np.isfinite(value).all()
                    assert len(calls) - before == int(condition["visual_conditioning"])
                    return value

                actions = infer()
                key = scene["episode"]
                if scene["name"] == "base":
                    repeated = infer()
                    np.testing.assert_allclose(actions, repeated, rtol=0, atol=1e-6)
                    bases[key] = actions
                    report["reproducibility_checks"].append(dict(condition=condition["name"], episode=key,
                        noise_seed=seed, max_action_difference=float(np.max(np.abs(actions - repeated)))))
                if not condition["visual_conditioning"]:
                    np.testing.assert_allclose(actions, bases[key], rtol=0, atol=1e-6)
                report["records"].append(dict(episode=key, split=scene["split"], seed=scene["seed"],
                    case=scene["name"], offset_m=scene["offset_m"], condition=condition["name"], noise_seed=seed,
                    actions=actions.tolist(), first_chunk_joint_change_rmse_rad=float(np.sqrt(
                        np.mean((actions[:8, 6:12] - bases[key][:8, 6:12]) ** 2)))))
                (out / "predictions.json").write_text(json.dumps(report, indent=2) + "\n")
                print(f"{condition['name']} noise {seed} episode {key} {scene['name']} complete", flush=True)
    hook.remove()
    assert len(report["records"]) == manifest["total_policy_predictions"]
    report["status"] = "predictions_completed"
    (out / "predictions.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
