"""Production dense training with fixed phase probes and exact sample exposure."""

import argparse
import json
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
from launch_omega512_protocol import PYTHON, write  # noqa: E402 - runtime environment/path bootstrap precedes imports


def main():
    from starVLA.dataloader.gr00t_lerobot.datasets import LeRobotMixtureDataset
    from starVLA.training import train_starvla as training

    class RecordedMixture(LeRobotMixtureDataset):
        def sample_step(self, index):
            dataset, episode, frame = super().sample_step(index)
            self.last_key = (int(episode), int(frame))
            return dataset, episode, frame

        def __getitem__(self, index):
            result = super().__getitem__(index)
            result["sample_episode"], result["sample_frame"] = self.last_key
            return result

    class ReferenceTrainer(training.VLATrainer):
        def prepare_training(self):
            self.exposure = Counter()
            super().prepare_training()
            self.study = Path(self.config.diagnostic.manifest).parent
            self.protocol = json.loads(Path(self.config.diagnostic.manifest).read_text())
            mixture = self.vla_train_dataloader.dataset
            # Accelerator's loader retains the original dataset object.
            mixture.__class__ = RecordedMixture
            single = mixture.datasets[0]
            expected = sorted(r["training_episode"] for r in self.protocol["policy"]["selected_episodes"])
            assert sorted(single.trajectory_ids.tolist()) == expected and len(expected) == 8
            baseline = json.loads(Path(self.config.diagnostic.baseline_statistics).read_text())
            actual = json.loads((Path(self.config.output_dir) / "dataset_statistics.json").read_text())
            for key in baseline:
                for modality in ["action", "state"]:
                    for name, values in baseline[key][modality].items():
                        np.testing.assert_allclose(actual[key][modality][name], values, rtol=0, atol=1e-7)
            self.keys = [(int(ep), int(frame)) for ep, frame in single.all_steps]
            self.positions = {key: i for i, key in enumerate(self.keys)}
            original_mode = single.transforms.training
            single.transforms.eval()
            try:
                self.full_examples = [single[i] for i in range(len(single))]
            finally:
                if original_mode:
                    single.transforms.train()
                else:
                    single.transforms.eval()
            self.probe_records = self.protocol["policy"]["fixed_probe"]
            self.probe_examples = [
                self.full_examples[self.positions[(r["training_episode"], r["frame"])]] for r in self.probe_records
            ]
            rows = {r["training_episode"]: r for r in self.protocol["policy"]["selected_episodes"]}
            phase_maps = {ep: {r["frame"]: r["phase"] for r in row["phase_frames"]} for ep, row in rows.items()}
            from launch_omega512_protocol import phases

            for ep, row in rows.items():
                phase_maps[ep] = {int(t): name for name, ix in phases(np.asarray(row["right_gripper"])) for t in ix}
            self.full_records = [
                dict(training_episode=ep, frame=frame, phase=phase_maps[ep][frame]) for ep, frame in self.keys
            ]
            audit = dict(
                episodes=expected,
                full_frames=len(self.full_examples),
                fixed_probe_frames=len(self.probe_examples),
                action_shapes=sorted({tuple(x["action"].shape) for x in self.full_examples}),
                instructions=sorted({x["lang"] for x in self.full_examples}),
                original_normalization_verified=True,
                omega_frozen=all(not p.requires_grad for p in self.model.omega.parameters()),
                probe="Training-trajectory sampled-action error; not held-out validation.",
            )
            assert audit["instructions"] == ["rotate_qrcode"] and audit["omega_frozen"]
            write(self.study / "reference_data_audit.json", audit)
            self.update_status("training")

        def state_dict(self):
            result = super().state_dict()
            result["frame_exposure"] = {f"{ep}:{frame}": count for (ep, frame), count in self.exposure.items()}
            return result

        def load_state_dict(self, state):
            super().load_state_dict(state)
            self.exposure = Counter(
                {tuple(int(x) for x in key.split(":")): value for key, value in state.get("frame_exposure", {}).items()}
            )

        def _get_next_batch(self):
            examples = super()._get_next_batch()
            for row in examples:
                self.exposure[(row["sample_episode"], row["sample_frame"])] += 1
            return examples

        def update_status(self, stage, **extra):
            write(
                self.study / "reference_status.json",
                dict(
                    status="completed" if stage == "completed" else "running",
                    stage=stage,
                    completed_updates=self.completed_steps,
                    target_updates=20000,
                    consumed_observations=sum(self.exposure.values()),
                    updated_unix=time.time(),
                    **extra,
                ),
            )

        def probe(self, examples, records, name):
            model = getattr(self.model, "module", self.model)
            size = int(self.config.datasets.vla_data.per_device_batch_size)
            modes = [(module, module.training) for module in model.modules()]
            predictions = []
            try:
                model.eval()
                with torch.inference_mode(), torch.random.fork_rng(devices=[torch.cuda.current_device()]):
                    torch.manual_seed(int(self.config.seed))
                    torch.cuda.manual_seed(int(self.config.seed))
                    for start in range(0, len(examples), size):
                        batch = examples[start : start + size]
                        predictions.append(
                            np.asarray(
                                model.predict_action(examples=batch, use_ddim=True, num_ddim_steps=20)[
                                    "normalized_actions"
                                ]
                            )
                        )
            finally:
                for module, mode in modes:
                    module.training = mode
            predicted = np.concatenate(predictions)
            target = np.stack([x["action"] for x in examples]).astype(np.float32)
            assert predicted.shape == target.shape and np.isfinite(predicted).all()
            err = (predicted.astype(np.float64) - target) ** 2

            def scores(indices):
                v = err[indices]
                return dict(
                    action_mse=float(v.mean()),
                    right_joint_mse=float(v[..., 6:12].mean()),
                    right_gripper_mse=float(v[..., 13].mean()),
                    first_command_mse=float(v[:, 0].mean()),
                )

            result = dict(
                step=self.completed_steps,
                probe=name,
                samples=len(examples),
                **scores(np.arange(len(examples))),
                by_phase={
                    phase: scores([i for i, r in enumerate(records) if r["phase"] == phase])
                    for phase in ["approach", "grasp", "transport", "release"]
                },
                by_episode={
                    str(ep): scores([i for i, r in enumerate(records) if r["training_episode"] == ep])
                    for ep in sorted({r["training_episode"] for r in records})
                },
                diffusion_seed=int(self.config.seed),
                num_ddim_steps=20,
            )
            with (self.study / "reference_probe_metrics.jsonl").open("a") as stream:
                stream.write(json.dumps(result) + "\n")
            (self.study / "action_probes").mkdir(exist_ok=True)
            np.savez_compressed(
                self.study / "action_probes" / f"{name}_step_{self.completed_steps:05d}.npz",
                predicted=predicted,
                target=target,
                episodes=np.array([r["training_episode"] for r in records]),
                frames=np.array([r["frame"] for r in records]),
            )
            write(
                self.study / "frame_exposure.json",
                dict(
                    step=self.completed_steps,
                    total=sum(self.exposure.values()),
                    frames=[
                        dict(training_episode=ep, frame=frame, count=self.exposure[(ep, frame)])
                        for ep, frame in self.keys
                    ],
                ),
            )
            self.update_status("training", latest_probe=result)
            print("REFERENCE_PROBE " + json.dumps(result), flush=True)
            return result

        def eval_action_model(self, step_metrics=None):
            result = self.probe(self.probe_examples, self.probe_records, "fixed128")
            return {**(step_metrics or {}), "training_probe_action_mse": result["action_mse"]}

        def _save_checkpoint(self):
            super()._save_checkpoint()
            self.probe(self.full_examples, self.full_records, "full")
            if self.completed_steps in [10000, 20000]:
                self.update_status("matched_scene_rollouts")
                torch.cuda.empty_cache()
                command = [
                    str(PYTHON),
                    str(ROOT / "tools/evaluate_omega512_reference_fit.py"),
                    "--manifest",
                    str(self.study / "manifest.json"),
                    "--step",
                    str(self.completed_steps),
                ]
                with (self.study / f"rollout_worker_{self.completed_steps}.log").open("a") as stream:
                    result = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT)
                if result.returncode:
                    # Retain the saved state and continue fitting; expose evaluation
                    # errors explicitly rather than losing the remaining budget.
                    write(
                        self.study / f"rollout_error_{self.completed_steps}.json",
                        dict(exit_code=result.returncode, command=command),
                    )
                self.update_status("training", rollout_exit_code=result.returncode)

        def _log_metrics(self, metrics):
            super()._log_metrics(metrics)
            if self.completed_steps % int(self.config.trainer.logging_frequency) == 0:
                self.update_status("training")

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config_yaml", required=True)
    args = parser.parse_args()
    cfg = training.apply_config_compat(OmegaConf.load(args.config_yaml))
    cfg.config_yaml = args.config_yaml
    training.VLATrainer = ReferenceTrainer
    study = Path(cfg.diagnostic.manifest).parent
    try:
        training.main(cfg)
        path = study / "reference_status.json"
        status = json.loads(path.read_text())
        status.update(status="completed", stage="completed")
        write(path, status)
    except Exception as error:
        path = study / "reference_status.json"
        status = json.loads(path.read_text()) if path.exists() else {}
        status.update(status="failed", error=repr(error))
        write(path, status)
        raise


if __name__ == "__main__":
    main()
