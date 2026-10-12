"""Run fresh one-trajectory training on GPU 1 and verified scene controls on GPU 0."""

import argparse
import datetime
import hashlib
import json
import os
import shutil
import socket
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
PYTHON = Path("/export/home/haoliang/miniconda3/envs/starVLA/bin/python")
SIM_PYTHON = Path("/export/home/haoliang/miniconda3/envs/RoboTwin/bin/python")
ROBOTWIN = Path("/export/home/haoliang/projects/RoboTwin_PVG")
BASE = ROOT / "results/Checkpoints/omega512_robotwin_rotate_qrcode_dense_fresh_10000steps_20261005_053805"
QUERY = ROOT / "results/Checkpoints/omega512_robotwin_rotate_qrcode_state_query_local_fresh_10000steps_20261005_053805"
SCENES = [(0, 51, 131), (1, 60, 162), (2, 38, 91)]


def write(path, data):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(data, indent=2) + "\n")
    temporary.replace(path)


def environment(gpu):
    return dict(
        os.environ,
        CUDA_VISIBLE_DEVICES=str(gpu),
        PYTHONUNBUFFERED="1",
        OMP_NUM_THREADS="4",
        STARVLA_DISABLE_DEEPSPEED="1",
        STARVLA_FRAMEWORK_MODULES="starVLA.model.framework.VLM4A.Omega512DiT",
        WANDB_MODE="offline",
        NO_ALBUMENTATIONS_UPDATE="1",
        ROBOTWIN_PATH=str(ROBOTWIN),
        ROBOTWIN_SUPPRESS_EVAL_CONFIG="1",
        PYTHONPATH=str(ROOT) + ":" + str(ROBOTWIN),
        PATH=str(SIM_PYTHON.parent) + ":" + os.environ["PATH"],
    )


def prepare():
    import h5py
    import numpy as np
    import pandas as pd
    from omegaconf import OmegaConf

    from deployment.model_server.policy_norm_processor import PolicyNormProcessor

    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d_%H%M%S")
    directory = ROOT / "results/robotwin_omega512_minimum_case" / stamp
    directory.mkdir(parents=True, exist_ok=False)
    cfg = OmegaConf.load(BASE / "config.full.yaml")
    run_id = f"omega512_overfit_episode0_fresh_{stamp}"
    checkpoint_dir = ROOT / "results/Checkpoints" / run_id
    assert not checkpoint_dir.exists()
    cfg.run_id = run_id
    cfg.run_root_dir = str(ROOT / "results/Checkpoints")
    cfg.output_dir = str(checkpoint_dir)
    cfg.datasets.vla_data.validation_fraction = 0.0
    cfg.datasets.vla_data.episode_ids = [0]
    split = json.loads((BASE / "data_split/episode_split.json").read_text())
    cfg.datasets.vla_data.statistics_episode_ids = split["rotate_qrcode"]["train_episode_ids"]
    cfg.datasets.vla_data.statistics_cache_dir = str(directory / "normalization_cache")
    shutil.copytree(BASE / "data_split/rotate_qrcode", directory / "normalization_cache/rotate_qrcode")
    cfg.framework.state_conditioning = "full"
    cfg.framework.visual_conditioning = True
    cfg.evaluation.diffusion_seed = 4000
    cfg.diagnostic = dict(
        baseline_statistics=str(BASE / "dataset_statistics.json"),
        training_episode=0,
        source_episode=51,
        simulator_seed=131,
    )
    for key in ("pretrained_checkpoint", "resume_checkpoint", "is_resume"):
        cfg.trainer.pop(key, None)
    cfg.trainer.save_training_state = True
    cfg.config_yaml = str(directory / "overfit.yaml")
    OmegaConf.save(cfg, cfg.config_yaml, resolve=True)
    processor = PolicyNormProcessor(str(BASE / "checkpoints/steps_10000_pytorch_model.pt"))
    order = [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12, 6, 13]
    payload, audits = {}, []
    data_root = Path(str(cfg.datasets.vla_data.data_root_dir)) / "rotate_qrcode"
    source = Path("/export/home/haoliang/projects/RoboTwin/data/rotate_qrcode/demo_clean")
    for episode, source_episode, seed in SCENES:
        df = pd.read_parquet(
            data_root / f"data/chunk-000/episode_{episode:06d}.parquet", columns=["action", "observation.state"]
        )
        raw_sim = np.stack(df["action"]).astype(np.float32)
        states = np.stack(df["observation.state"]).astype(np.float32)
        raw = raw_sim[:, order]
        pieces, cursor = {}, 0
        for key in processor.action_keys:
            width = processor.action_key_dims[key]
            pieces[key] = raw[:, cursor : cursor + width].copy()
            cursor += width
        normalized = processor.transform(pieces)
        normalized = np.concatenate([np.asarray(normalized[key]) for key in processor.action_keys], axis=-1)
        roundtrip = processor.unapply_actions(normalized)
        with h5py.File(source / f"data/episode{source_episode}.hdf5") as file:
            source_states = file["joint_action/vector"][:].astype(np.float32)
        np.testing.assert_allclose(states, source_states[:-1], rtol=0, atol=1e-6)
        np.testing.assert_allclose(raw_sim, source_states[1:], rtol=0, atol=1e-6)
        payload[f"episode_{episode}_raw"] = raw
        payload[f"episode_{episode}_roundtrip"] = roundtrip
        audits.append(
            dict(
                training_episode=episode,
                source_episode=source_episode,
                simulator_seed=seed,
                frames=len(raw),
                observation_matches_source_t=True,
                action_matches_source_t_plus_1=True,
                joint_roundtrip_max_error=float(np.max(np.abs(roundtrip[:, :12] - raw[:, :12]))),
                gripper_roundtrip_max_error=float(np.max(np.abs(roundtrip[:, 12:] - raw[:, 12:]))),
                temporal_contract="Converted observation t predicts recorded command t+1",
            )
        )
        np.testing.assert_allclose(roundtrip[:, :12], raw[:, :12], rtol=0, atol=2e-6)
    metadata = dict(
        action_chunk_size=16,
        include_state=True,
        execute_horizon=8,
        num_ddim_steps=20,
        preserve_native_images=True,
        training_obs_image_size=[640, 480],
        instruction_override="rotate_qrcode",
        evaluation_diffusion_seed=4000,
        framework_name="RecordedExpertOracle",
        state_normalization="server",
    )
    payload["metadata"] = np.array(json.dumps(metadata))
    np.savez_compressed(directory / "expert_actions.npz", **payload)
    write(
        directory / "expert_data_audit.json",
        dict(
            scenes=audits,
            adapter="Production ModelClient; inference transport replaced by recorded action oracle",
            normalization=(
                "Production PolicyNormProcessor transform and unapply_actions; raw comparator scheduled for failed"
                " replays."
            ),
        ),
    )
    sources = [
        "tools/train_omega512_overfit.py",
        "tools/robotwin_omega_seen_scene.py",
        "tools/launch_omega512_minimum_case.py",
        "starVLA/model/framework/VLM4A/Omega512DiT.py",
        "starVLA/model/modules/action_model/OmegaCrossAttentionHeader.py",
        "starVLA/model/modules/projector/omega_readout.py",
        "starVLA/training/train_starvla.py",
        "examples/simBenchmarks/Robotwin/eval_files/model2robotwin_interface.py",
        "deployment/model_server/policy_norm_processor.py",
    ]
    hashes = {}
    for name in sources:
        destination = directory / "code" / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / name, destination)
        hashes[name] = hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
    manifest = dict(
        status="prepared",
        created_utc=stamp,
        directory=str(directory),
        code_sha256=hashes,
        training=dict(
            status="prepared",
            gpu=1,
            config=cfg.config_yaml,
            checkpoint_dir=str(checkpoint_dir),
            max_updates=10000,
            save_updates=[5000, 10000],
            fresh_initialization=True,
            frozen_backbone=True,
            conditioning_dropout_unchanged=True,
            normalization="Original 90-episode training statistics; samples only from episode 0",
            log=str(directory / "overfit_training.log"),
        ),
        diagnostics=dict(
            status="prepared",
            gpu=0,
            expert_replay="3 verified recorded training trajectories; raw comparators if necessary",
            rollouts="Dense and state-query/local 10000 checkpoints, 3 scenes each",
            scenes=SCENES,
            log=str(directory / "diagnostics_worker.log"),
        ),
        overfit_evaluation=dict(status="queued_after_training", gpu=1, episode=0, execute_horizons=[1, 2, 8]),
    )
    write(directory / "manifest.json", manifest)
    (directory / "README.md").write_text(
        """Fresh Omega512 single-trajectory diagnostic

GPU 1: original dense architecture, fresh head/readout, frozen Omega reference,
episode 0 only (source 51, seed 131), original normalization/dropout/optimizer,
10,000 updates, complete resumable saves at 5,000 and 10,000.
Every recorded observation is evaluated with fixed-noise 20-step DDIM every 250 updates.
The metric is training-trajectory MSE, not held-out validation.
After training, matched-scene videos are queued for execution horizons 1, 2 and 8.

GPU 0: recorded expert actions through production normalization and ModelClient,
then dense and state-query/local original checkpoints on verified seeds 131, 162, 91.
Failed normalized expert replays receive a raw-action comparator.
All rollouts save videos and traces of actions, commanded/physical states,
end-effector/object poses and TOPP execution durations.

Status: [manifest.json](manifest.json). Logs and results are in this directory.
"""
    )
    return directory / "manifest.json"


def child(command, log, gpu, cwd=ROOT):
    with Path(log).open("w") as stream:
        return subprocess.Popen(
            command, cwd=cwd, env=environment(gpu), stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT
        )


def wait_server(server, port):
    deadline = time.monotonic() + 300
    while time.monotonic() < deadline:
        if server.poll() is not None:
            raise RuntimeError(f"Policy server exited {server.returncode}")
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=1):
                return
        except OSError:
            time.sleep(1)
    raise TimeoutError("Server readiness timeout")


def finish(process):
    if process is not None and process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=20)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()


def diagnostic_worker(path, overfit=False):
    directory = path.parent
    state_path = directory / ("overfit_evaluation_status.json" if overfit else "diagnostics_status.json")
    status = dict(status="running", worker_pid=os.getpid(), completed=[], gpu=1 if overfit else 0)
    write(state_path, status)
    checkpoint = BASE / "checkpoints/steps_10000_pytorch_model.pt"
    gpu = status["gpu"]
    server = None
    simulator = None
    try:
        if not overfit:
            out = directory / "expert_replay"
            command = [
                str(SIM_PYTHON),
                str(ROOT / "tools/robotwin_omega_seen_scene.py"),
                "--mode",
                "expert",
                "--checkpoint",
                str(checkpoint),
                "--variant",
                "expert_roundtrip",
                "--output-dir",
                str(out),
                "--expert-input",
                str(directory / "expert_actions.npz"),
            ]
            simulator = child(command, directory / "expert_replay.log", gpu)
            status.update(phase="expert_replay", simulator_pid=simulator.pid)
            write(state_path, status)
            if simulator.wait():
                raise RuntimeError("Expert replay process failed; inspect expert_replay.log")
            records = json.loads((out / "expert_roundtrip_results.json").read_text())
            validate_videos(records)
            status["completed"].append(
                dict(phase="expert_replay", success_count=sum(r["success"] for r in records), episodes=len(records))
            )
            failed = [str(r["training_episode"]) for r in records if not r["success"]]
            if failed:
                raw = command.copy()
                raw[raw.index("expert_roundtrip")] = "expert_raw"
                raw += ["--raw-expert", "--episodes", ",".join(failed)]
                simulator = child(raw, directory / "expert_raw_replay.log", gpu)
                status.update(phase="expert_raw_comparator", simulator_pid=simulator.pid)
                write(state_path, status)
                if simulator.wait():
                    raise RuntimeError("Raw expert replay process failed")
                records = json.loads((out / "expert_raw_results.json").read_text())
                validate_videos(records)
                status["completed"].append(
                    dict(
                        phase="expert_raw_comparator",
                        success_count=sum(r["success"] for r in records),
                        episodes=len(records),
                    )
                )
        if overfit:
            manifest = json.loads(path.read_text())
            models = [
                ("overfit", Path(manifest["training"]["checkpoint_dir"]) / "checkpoints/steps_10000_pytorch_model.pt")
            ]
            horizons, episodes = [1, 2, 8], "0"
        else:
            models = [("dense", checkpoint), ("state_query_local", QUERY / "checkpoints/steps_10000_pytorch_model.pt")]
            horizons, episodes = [8], "0,1,2"
        port = 5751 if overfit else 5750
        for variant, checkpoint in models:
            server_command = [
                str(PYTHON),
                str(ROOT / "deployment/model_server/server_policy.py"),
                "--ckpt_path",
                str(checkpoint),
                "--port",
                str(port),
                "--config_override",
                "evaluation.diffusion_seed=4000",
            ]
            server = child(server_command, directory / f"{variant}_server.log", gpu)
            status.update(phase=f"{variant}_loading", server_pid=server.pid)
            write(state_path, status)
            wait_server(server, port)
            for horizon in horizons:
                name = f"{variant}_horizon{horizon}"
                out = directory / ("overfit_evaluation" if overfit else "training_scene_rollouts") / name
                command = [
                    str(SIM_PYTHON),
                    str(ROOT / "tools/robotwin_omega_seen_scene.py"),
                    "--mode",
                    "rollout",
                    "--checkpoint",
                    str(checkpoint),
                    "--variant",
                    name,
                    "--port",
                    str(port),
                    "--output-dir",
                    str(out),
                    "--episodes",
                    episodes,
                    "--execute-horizon",
                    str(horizon),
                ]
                simulator = child(command, directory / f"{name}.log", gpu)
                status.update(phase=name, simulator_pid=simulator.pid)
                write(state_path, status)
                if simulator.wait():
                    raise RuntimeError(f"{name} failed; inspect its log")
                records = json.loads((out / f"{name}_results.json").read_text())
                validate_videos(records)
                status["completed"].append(
                    dict(phase=name, success_count=sum(r["success"] for r in records), episodes=len(records))
                )
                write(state_path, status)
            finish(server)
            server = None
        status.update(status="completed", finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
    except Exception as error:
        status.update(status="failed", error=str(error))
        raise
    finally:
        finish(simulator)
        finish(server)
        write(state_path, status)


def validate_videos(records):
    import av

    for record in records:
        with av.open(record["video"]) as container:
            frames = sum(1 for unused in container.decode(video=0))
        if frames < 2:
            raise RuntimeError(f"Truncated video: {record['video']}")


def worker(path):
    manifest = json.loads(path.read_text())
    assert manifest["status"] == "prepared"
    for name, sha in manifest["code_sha256"].items():
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == sha, name
    for gpu in (0, 1):
        used = subprocess.check_output(
            ["nvidia-smi", f"--id={gpu}", "--query-gpu=memory.used", "--format=csv,noheader,nounits"], text=True
        )
        assert int(used.strip()) < 1000, f"GPU {gpu} is occupied"
    assert shutil.disk_usage(ROOT).free > 35 * 1024**3
    training = child(
        [str(PYTHON), str(ROOT / "tools/train_omega512_overfit.py"), "--config_yaml", manifest["training"]["config"]],
        manifest["training"]["log"],
        1,
    )
    diagnostics = child(
        [str(PYTHON), str(Path(__file__).resolve()), "--diagnostic-worker", str(path)], manifest["diagnostics"]["log"], 0
    )
    manifest.update(status="running", worker_pid=os.getpid())
    manifest["training"].update(status="running", pid=training.pid)
    manifest["diagnostics"].update(status="running", pid=diagnostics.pid)
    write(path, manifest)
    overfit_eval = None
    while True:
        if training.poll() is not None and manifest["training"]["status"] == "running":
            code = training.returncode
            marker = Path(manifest["training"]["checkpoint_dir"]) / "checkpoints/steps_10000/complete.json"
            manifest["training"].update(
                status="completed" if code == 0 and marker.is_file() else "failed", exit_code=code
            )
            if manifest["training"]["status"] == "completed":
                overfit_eval = child(
                    [str(PYTHON), str(Path(__file__).resolve()), "--overfit-evaluation-worker", str(path)],
                    path.parent / "overfit_evaluation_worker.log",
                    1,
                )
                manifest["overfit_evaluation"].update(status="running", pid=overfit_eval.pid)
            else:
                manifest["overfit_evaluation"]["status"] = "not_started_training_failed"
        if diagnostics.poll() is not None:
            manifest["diagnostics"].update(
                status="completed" if diagnostics.returncode == 0 else "failed", exit_code=diagnostics.returncode
            )
        if overfit_eval is not None and overfit_eval.poll() is not None:
            manifest["overfit_evaluation"].update(
                status="completed" if overfit_eval.returncode == 0 else "failed", exit_code=overfit_eval.returncode
            )
        write(path, manifest)
        if (
            manifest["training"]["status"] != "running"
            and diagnostics.poll() is not None
            and (overfit_eval is None or overfit_eval.poll() is not None)
        ):
            break
        time.sleep(5)
    # Poll completion one final time even when both initial children end together.
    manifest["diagnostics"].update(
        status="completed" if diagnostics.returncode == 0 else "failed", exit_code=diagnostics.returncode
    )
    if overfit_eval is not None:
        manifest["overfit_evaluation"].update(
            status="completed" if overfit_eval.returncode == 0 else "failed", exit_code=overfit_eval.returncode
        )
    manifest["status"] = (
        "completed"
        if all(manifest[key]["status"] == "completed" for key in ("training", "diagnostics", "overfit_evaluation"))
        else "failed"
    )
    write(path, manifest)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", type=Path)
    parser.add_argument("--diagnostic-worker", type=Path)
    parser.add_argument("--overfit-evaluation-worker", type=Path)
    args = parser.parse_args()
    if args.worker:
        worker(args.worker.resolve())
    elif args.diagnostic_worker:
        diagnostic_worker(args.diagnostic_worker.resolve())
    elif args.overfit_evaluation_worker:
        diagnostic_worker(args.overfit_evaluation_worker.resolve(), overfit=True)
    else:
        path = prepare()
        with (path.parent / "worker.log").open("w") as stream:
            process = subprocess.Popen(
                [str(PYTHON), str(Path(__file__).resolve()), "--worker", str(path)],
                cwd=ROOT,
                stdin=subprocess.DEVNULL,
                stdout=stream,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        print(json.dumps(dict(manifest=str(path), worker_pid=process.pid)), flush=True)


if __name__ == "__main__":
    main()
