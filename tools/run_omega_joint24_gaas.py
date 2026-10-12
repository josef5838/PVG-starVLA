"""Run the queued batch benchmark, paired preflight and fresh training in order."""

import argparse
import datetime
import json
import os
import shutil
import signal
import subprocess
import sys
from pathlib import Path

from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, required=True)
    args = parser.parse_args()
    study = args.study
    manifest_path = study / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if Path(manifest["output_dir"]).exists():
        raise FileExistsError(f"Fresh training output already exists: {manifest['output_dir']}")
    production = study / "production.yaml"
    child = None

    def update(status, **kwargs):
        manifest.update(status=status, updated_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), **kwargs)
        temporary = manifest_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(manifest, indent=2)+"\n")
        temporary.replace(manifest_path)

    def terminate(signum, frame):
        if child is not None and child.poll() is None:
            child.terminate()
            child.wait(timeout=45)
        raise SystemExit(128+signum)

    signal.signal(signal.SIGTERM, terminate)
    signal.signal(signal.SIGINT, terminate)

    def run(phase, command):
        nonlocal child
        update(phase)
        with (study / f"{phase}.log").open("w") as stream:
            child = subprocess.Popen(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
            result = child.wait()
        child = None
        if result:
            raise RuntimeError(f"{phase} exited {result}; see {study / (phase+'.log')}")

    try:
        update("starting", pbs_job_id=os.environ.get("PBS_JOBID"), hostname=os.uname().nodename)
        if manifest.get("fixed_batch", False):
            shutil.copy2(manifest["config"], production)
            cfg = OmegaConf.load(production)
            cfg.config_yaml = str(production)
            OmegaConf.save(cfg, production, resolve=True)
            update("preflight", production_config=str(production), fixed_batch=True)
        else:
            run("batch_tuning", [sys.executable, str(ROOT / "tools/tune_omega_joint24_batch.py"),
                                 "--config", manifest["config"], "--out", str(production),
                                 "--report", str(study / "batch_tuning.json")])
            tuning = json.loads((study / "batch_tuning.json").read_text())
            update("preflight", production_config=str(production), batch_tuning=tuning)
        preflight_tool = manifest.get("preflight_tool", "preflight_omega_camera_streams.py")
        run("preflight", [sys.executable, str(ROOT / "tools" / preflight_tool),
                          "--config", str(production), "--out", str(study / "preflight.json"),
                          "--reference", manifest["reference_preflight"],
                          "--reference-statistics", manifest["reference_statistics"]])
        preflight = json.loads((study / "preflight.json").read_text())
        assert preflight["trainable_parameters"] == manifest["expected_trainable_parameters"]
        assert preflight["reference_parity_verified"] and preflight["reference_statistics_verified"]
        update("training", preflight_passed=True)
        run("training", [sys.executable, str(ROOT / "tools/train_omega_camera_streams.py"),
                         "--config_yaml", str(production)])
        status = json.loads((Path(manifest["output_dir"]) / "stream_training_status.json").read_text())
        cfg = OmegaConf.load(production)
        assert status["status"] == "completed" and status["completed_updates"] == cfg.trainer.max_train_steps
        update("completed", completed_updates=status["completed_updates"],
               consumed_observations=status["consumed_observations"])
    except BaseException as exc:
        update("failed", failed_phase=manifest["status"], error=repr(exc))
        raise
    finally:
        # Preserve the latest complete optimizer/RNG/cursor state from node-local /tmp.
        if production.exists():
            cfg = OmegaConf.load(production)
            latest = Path(cfg.trainer.training_state_root) / "latest"
            if (latest / "complete.json").is_file():
                if manifest.get("durable_state", False):
                    update(manifest["status"], retained_training_state=str(latest))
                else:
                    destination = Path(manifest["output_dir"]) / "retained_training_state"
                    shutil.copytree(latest, destination)
                    link = Path(manifest["output_dir"]) / "latest_training_state"
                    if link.is_symlink():
                        link.unlink()
                    link.symlink_to(destination, target_is_directory=True)
                    update(manifest["status"], retained_training_state=str(destination))


if __name__ == "__main__":
    main()
