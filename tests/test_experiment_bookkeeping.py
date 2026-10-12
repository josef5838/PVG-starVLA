"""Check cancellation, stale process state, and historical checkpoint evidence."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tools.launch_rotate_qrcode_streams import worker
from tools.omega_camera_stream_status import read_status
from tools.prepare_rotate_qrcode_evaluation import completed_runs
from tools.snapshot_machine_experiments import training_record


class BookkeepingTests(unittest.TestCase):
    def test_stale_running_status_and_cancelled_controls_are_distinguished(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "primary"
            output.mkdir()
            (output / "stream_training_status.json").write_text(
                json.dumps(dict(status="running", completed_updates=100))
            )
            path = root / "manifest.json"
            runs = [
                dict(
                    variant=name,
                    gpu="0",
                    pid=None,
                    config="config.yaml",
                    log="train.log",
                    output_dir=str(output if name == "direct" else root / name),
                    status=state,
                )
                for name, state in [("direct", "running"), ("state_only", "cancelled")]
            ]
            path.write_text(json.dumps(dict(status="running", runs=runs)))
            with patch("tools.omega_camera_stream_status.process_alive", return_value=False):
                status = read_status(path)
            self.assertEqual(status["status"], "interrupted")
            self.assertEqual([row["status"] for row in status["runs"]], ["interrupted", "cancelled"])
            (output / "stream_training_status.json").write_text(
                json.dumps(dict(status="completed", completed_updates=200))
            )
            self.assertEqual(read_status(path)["status"], "completed_with_cancellations")

    def test_queue_never_launches_cancelled_or_completed_jobs(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.json"
            path.write_text(
                json.dumps(
                    dict(
                        status="running",
                        runs=[
                            dict(variant=name, gpu="0", status=state)
                            for name, state in [("direct", "completed"), ("state_only", "cancelled")]
                        ],
                    )
                )
            )
            with patch("tools.launch_rotate_qrcode_streams.subprocess.Popen") as launch:
                worker(path, "0")
            launch.assert_not_called()

    def test_old_final_weights_do_not_prove_a_continuation_finished(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            (path / "config.full.yaml").write_text("trainer:\n  max_train_steps: 60000\n")
            (path / "summary.jsonl").write_text('{"steps": 39000}\n')
            (path / "final_model").mkdir()
            (path / "final_model/pytorch_model.pt").touch()
            # Weight paths are relative to the project, so use a fixture root.
            with patch("tools.snapshot_machine_experiments.ROOT", path.parent):
                status = training_record(path)
            self.assertEqual(status["completed_updates"], 39000)
            self.assertEqual(status["status"], "final_weights_present_budget_unverified")

    def test_evaluation_omits_cancelled_controls_but_requires_primary_completion(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            status = output / "stream_training_status.json"
            primary = dict(variant="direct", status="running", output_dir=str(output))
            cancelled = dict(variant="state_only", status="cancelled")
            training = dict(optimizer_updates_per_run=20000, runs=[primary, cancelled])
            status.write_text(json.dumps(dict(status="completed", completed_updates=20000)))
            self.assertEqual(completed_runs(training), [primary])
            status.write_text(json.dumps(dict(status="running", completed_updates=19990)))
            with self.assertRaisesRegex(RuntimeError, "Run not completed: direct"):
                completed_runs(training)


if __name__ == "__main__":
    unittest.main()
