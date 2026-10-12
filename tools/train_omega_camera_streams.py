"""Standard StarVLA training with local progress and matched-sample audits."""

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def write(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def main():
    import torch
    from omegaconf import OmegaConf

    from starVLA.training import train_starvla as training

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config_yaml", required=True)
    args = parser.parse_args()
    cfg = training.apply_config_compat(OmegaConf.load(args.config_yaml))
    cfg.config_yaml = args.config_yaml
    directory = Path(cfg.run_root_dir) / cfg.run_id
    directory.mkdir(parents=True, exist_ok=True)
    status_path = directory / "stream_training_status.json"
    write(
        status_path,
        dict(
            status="starting",
            completed_updates=0,
            target_updates=int(cfg.trainer.max_train_steps),
            updated_unix=time.time(),
        ),
    )

    class StreamTrainer(training.VLATrainer):
        def prepare_training(self):
            super().prepare_training()
            self.progress("running")

        def _get_next_batch(self):
            batch = super()._get_next_batch()
            if self.consumed_batches <= 16:
                with (directory / "initial_sample_order.jsonl").open("a") as stream:
                    stream.write(
                        json.dumps(
                            dict(
                                batch=self.consumed_batches,
                                samples=[dict(episode=x["sample_episode"], frame=x["sample_frame"]) for x in batch],
                            )
                        )
                        + "\n"
                    )
            return batch

        def progress(self, status, metrics=None):
            model = getattr(self.model, "module", self.model)
            write(
                status_path,
                dict(
                    status=status,
                    completed_updates=self.completed_steps,
                    target_updates=int(self.config.trainer.max_train_steps),
                    consumed_batches=self.consumed_batches,
                    consumed_observations=self.consumed_batches
                    * int(self.config.datasets.vla_data.per_device_batch_size),
                    visual_info=model.last_visual_info,
                    latest_metrics=metrics,
                    peak_allocated_gpu_gib=torch.cuda.max_memory_allocated() / 2**30,
                    updated_unix=time.time(),
                ),
            )

        def _log_metrics(self, metrics):
            super()._log_metrics(metrics)
            if self.completed_steps % int(self.config.trainer.logging_frequency) == 0:
                self.progress("running", metrics)
                with (directory / "stream_training_metrics.jsonl").open("a") as stream:
                    stream.write(json.dumps(dict(step=self.completed_steps, **metrics)) + "\n")

        def _save_checkpoint(self):
            if not self.config.trainer.get("compact_checkpoints", False):
                super()._save_checkpoint()
                self.progress("running")
                return
            if self.accelerator.is_main_process and self.completed_steps in self.config.trainer.milestone_updates:
                torch.save(
                    self._model_state_dict(),
                    Path(self.checkpoint_dir) / f"steps_{self.completed_steps}_pytorch_model.pt",
                )
            self._save_training_state(Path(self.checkpoint_dir) / f"steps_{self.completed_steps}")
            self.progress("running")

        def _save_training_state(self, target):
            root = self.config.trainer.get("training_state_root")
            if not root:
                return super()._save_training_state(target)
            # Best weights are retained separately. Resume the latest periodic/final state.
            if Path(target).parent.name == "best_model":
                return
            root = Path(root)
            root.mkdir(parents=True, exist_ok=True)
            staging = root / "staging"
            super()._save_training_state(staging)
            if self.accelerator.is_main_process:
                latest, previous = root / "latest", root / "previous"
                if previous.exists():
                    shutil.rmtree(previous)
                if latest.exists():
                    latest.rename(previous)
                staging.rename(latest)
                if previous.exists():
                    shutil.rmtree(previous)
                link = directory / "latest_training_state"
                if not link.is_symlink():
                    link.symlink_to(latest, target_is_directory=True)
            self.accelerator.wait_for_everyone()

        def _finalize_training(self):
            super()._finalize_training()
            self.progress("completed")

    training.VLATrainer = StreamTrainer
    try:
        training.main(cfg)
    except Exception as exc:
        status = json.loads(status_path.read_text())
        status.update(status="failed", error=repr(exc), updated_unix=time.time())
        write(status_path, status)
        raise


if __name__ == "__main__":
    main()
