"""Prepare and launch matched rotate_qrcode histories and conditioning controls."""

import argparse
import datetime
import fcntl
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from starVLA.dataloader.episode_split import (  # noqa: E402
    split_train_validation_test,
)

PYTHON = Path(os.environ.get("STARVLA_PYTHON", sys.executable))
RUNTIME = Path(os.environ.get("ROBOTWIN_NVIDIA_RUNTIME", "/export/home/haoliang/.cache/robotwin-nvidia-595.71.05"))
PROTOCOL = ROOT / "docs/omega_rotate_qrcode_protocol_20261010.md"


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write(path, value):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(value, indent=2) + "\n")
    tmp.replace(path)


def update(path, callback):
    with path.with_suffix(".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        value = json.loads(path.read_text())
        result = callback(value)
        write(path, value)
        return result


def environment(gpu):
    env = dict(
        os.environ,
        CUDA_VISIBLE_DEVICES=str(gpu),
        STARVLA_DISABLE_DEEPSPEED="1",
        STARVLA_FRAMEWORK_MODULES="starVLA.model.framework.VLM4A.OmegaCameraStreamDiT",
        WANDB_MODE="offline",
        PYTHONUNBUFFERED="1",
        OMP_NUM_THREADS="4",
        NO_ALBUMENTATIONS_UPDATE="1",
        PYTHONPATH=str(ROOT),
        ACCELERATE_MIXED_PRECISION="bf16",
    )
    if (RUNTIME / "runtime_lib").is_dir():
        env["LD_LIBRARY_PATH"] = str(RUNTIME / "runtime_lib") + (
            ":" + env["LD_LIBRARY_PATH"] if env.get("LD_LIBRARY_PATH") else ""
        )
    return env


def prepare(args):
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d_%H%M%S")
    directory = ROOT / "results/omega_camera_stream_training" / f"rotate_qrcode_{stamp}"
    directory.mkdir(parents=True)
    cfg = OmegaConf.load(ROOT / "examples/simBenchmarks/Robotwin/train_files/omega512_robotwin.yaml")
    cfg.framework.name = "OmegaCameraStreamDiT"
    cfg.framework.omega.source_root = os.environ.get("OMEGA_ROOT", cfg.framework.omega.source_root)
    cfg.framework.omega.checkpoint_path = os.environ.get(
        "OMEGA_CHECKPOINT", str(Path(cfg.framework.omega.source_root) / "ckpt/vggt_omega_1b_512.pt")
    )
    cfg.framework.state_dropout_prob = 0.0
    cfg.framework.condition_dropout_prob = 0.0
    cfg.framework.state_conditioning = "full"
    cfg.framework.visual_conditioning = True
    cfg.framework.streams = dict(
        history_length=8,
        conditioning="direct",
        width=512,
        depth=2,
        heads=8,
        output_queries=32,
        temporal_embedding="learned",
        temporal_rope=False,
        observation_mode="history",
    )
    data = cfg.datasets.vla_data
    data.data_root_dir = os.environ.get("ROBOTWIN_DATA_ROOT", data.data_root_dir)
    data.data_mix = "robotwin_omega_stream8_rotate_qrcode"
    data.pack_image_history = True
    data.num_workers = 2
    data.prefetch_factor = 1
    data.test_fraction = 0.1
    data.validation_fraction = 0.1
    data.validation_seed = 42
    dataset = Path(data.data_root_dir) / "rotate_qrcode"
    train, val, test = split_train_validation_test(dataset, 0.1, 0.1, 42)
    episodes = [json.loads(x) for x in (dataset / "meta/episodes.jsonl").read_text().splitlines()]
    lengths = {x["episode_index"]: x["length"] for x in episodes}

    def anchors(ids):
        return [
            dict(episode=e, frame=int(t))
            for e in ids
            for t in np.unique(np.linspace(0, lengths[e] - 1, 16).round().astype(int))
        ]

    data.validation_samples = anchors(val)
    data.test_samples = anchors(test)
    cfg.trainer.max_train_steps = args.steps
    cfg.trainer.step_limit = args.steps
    cfg.trainer.sample_budget = args.steps * 8
    cfg.trainer.planned_training_samples = args.steps * 8
    cfg.trainer.save_interval = 1000
    cfg.trainer.eval_interval = 250
    cfg.trainer.validation_max_batches = len(data.validation_samples) // 4
    cfg.trainer.validation_action_indices = [6, 7, 8, 9, 10, 11, 13]
    cfg.trainer.validation_valid_future_only = True
    cfg.trainer.learning_rate.memory_proj = 3e-4
    cfg.trainer.compact_checkpoints = True
    cfg.trainer.milestone_updates = [10000]
    cfg.run_root_dir = str(ROOT / "results/Checkpoints")
    cfg.wandb_project = "omega-camera-streams-robotwin"
    manifest = dict(
        status="prepared",
        task="rotate_qrcode",
        seed=42,
        created_utc=stamp,
        protocol=str(PROTOCOL),
        dataset=str(dataset),
        source=os.environ.get(
            "ROBOTWIN_SOURCE_ROOT", "/export/home/haoliang/projects/RoboTwin/data/rotate_qrcode/demo_clean"
        ),
        train_episode_ids=train,
        validation_episode_ids=val,
        test_episode_ids=test,
        optimizer_updates_per_run=args.steps,
        effective_batch_per_run=8,
        sample_budget_per_run=args.steps * 8,
        history_offsets=list(range(-7, 1)),
        temporal_rope=False,
        temporal_embedding="learned",
        rollout_execute_horizons=[8, 4],
        diffusion_seed=4000,
        num_ddim_steps=20,
        rollout_checkpoint="best",
        camera_order=["head_camera", "left_camera", "right_camera"],
        evaluation_arm="right",
        fresh_scene_count=50,
        queue_policy="primary then control on each GPU",
        runs=[],
    )
    for variant, kind, mode, use_visual, gpu in [
        ("direct", "direct", "history", True, "0"),
        ("fused", "fused", "history", True, "1"),
        ("state_only", "direct", "history", False, "0"),
        ("current_only", "direct", "current_repeat", True, "1"),
    ]:
        item = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
        item.framework.streams.conditioning = kind
        item.framework.streams.observation_mode = mode
        item.framework.visual_conditioning = use_visual
        item.run_id = f"omega_stream8_rotate_qrcode_{variant}_{args.steps}steps_{stamp}"
        item.datasets.vla_data.statistics_cache_dir = str(Path(item.run_root_dir) / item.run_id / "data_split")
        item.trainer.training_state_root = str(Path("/tmp/omega_rotate_qrcode_training_states") / stamp / variant)
        path = directory / f"{variant}.yaml"
        item.config_yaml = str(path)
        OmegaConf.save(item, path, resolve=True)
        manifest["runs"].append(
            dict(
                variant=variant,
                gpu=gpu,
                config=str(path),
                config_sha256=digest(path),
                output_dir=str(Path(item.run_root_dir) / item.run_id),
                log=str(directory / f"{variant}.log"),
                preflight=str(directory / f"{variant}_preflight.json"),
                pid=None,
                status="prepared",
                command=[str(PYTHON), str(ROOT / "tools/train_omega_camera_streams.py"), "--config_yaml", str(path)],
            )
        )
    path = directory / "manifest.json"
    write(path, manifest)
    return path


def worker(path, gpu):
    manifest = json.loads(path.read_text())
    runs = [r for r in manifest["runs"] if r["gpu"] == gpu]
    for run in runs:

        def starting(m, run=run):
            r = next(x for x in m["runs"] if x["variant"] == run["variant"])
            if r["status"] in {"cancelled", "completed"}:
                return False
            r.update(status="starting", started_unix=time.time())
            return True

        if not update(path, starting):
            continue
        with Path(run["log"]).open("w") as stream:
            child = subprocess.Popen(
                run["command"],
                cwd=ROOT,
                env=environment(gpu),
                stdin=subprocess.DEVNULL,
                stdout=stream,
                stderr=subprocess.STDOUT,
            )

        def running(m, run=run, child=child):
            next(x for x in m["runs"] if x["variant"] == run["variant"]).update(pid=child.pid, status="running")

        update(path, running)
        code = child.wait()
        progress = Path(run["output_dir"]) / "stream_training_status.json"
        status = json.loads(progress.read_text()) if progress.exists() else {}
        success = (
            code == 0
            and status.get("status") == "completed"
            and status.get("completed_updates") == manifest["optimizer_updates_per_run"]
        )

        def finished(m, run=run, success=success, code=code):
            next(x for x in m["runs"] if x["variant"] == run["variant"]).update(
                status="completed" if success else "failed", exit_code=code, finished_unix=time.time()
            )
            if not success:
                m["status"] = "failed"
            elif all(r["status"] in {"completed", "cancelled"} for r in m["runs"]):
                m["status"] = (
                    "completed_with_cancellations" if any(r["status"] == "cancelled" for r in m["runs"]) else "completed"
                )

        update(path, finished)
        if not success:
            raise RuntimeError(f"Training failed: {run['variant']}; see {run['log']}")


def launch(path):
    m = json.loads(path.read_text())
    if m["status"] != "prepared":
        raise ValueError("Refuse duplicate launch")
    if not Path(m.get("geometry_registration", "")).is_file():
        raise ValueError("Geometry/scenes must be registered first")
    geometry = json.loads(Path(m["geometry_registration"]).read_text())
    if (
        geometry.get("status") != "completed"
        or len(geometry["demo_initial_geometry"]) != 100
        or len(geometry["fresh_scenes"]) != 50
    ):
        raise ValueError("Require completed 100-demo audit and 50-scene registration")
    for audit_name in ["trainer_integration_audit.json", "split_validation_audit.json"]:
        if json.loads((path.parent / audit_name).read_text()).get("status") != "passed":
            raise ValueError(f"Failed integration audit: {audit_name}")
    audits = []
    normalized = []
    for r in m["runs"]:
        if digest(r["config"]) != r["config_sha256"]:
            raise ValueError("Config changed after registration")
        if Path(r["output_dir"]).exists():
            raise ValueError("Fresh output directory already exists")
        audit = json.loads(Path(r["preflight"]).read_text())
        assert audit["status"] == "passed"
        audits.append(audit)
        cfg = OmegaConf.to_container(OmegaConf.load(r["config"]), resolve=True)
        if r["variant"] in {"direct", "fused"}:
            for key in ["run_id", "config_yaml"]:
                cfg.pop(key)
            cfg["datasets"]["vla_data"].pop("statistics_cache_dir")
            cfg["trainer"].pop("training_state_root")
            cfg["framework"]["streams"].pop("conditioning")
            normalized.append(cfg)
    assert normalized[0] == normalized[1]
    for key in ["initial_common_weights_sha256", "batch_image_sha256", "batch_action_sha256"]:
        assert len({a[key] for a in audits}) == 1, f"Unmatched {key}"
    m["preflight_input_and_common_initialization_matched"] = True
    m["geometry_registration_sha256"] = digest(m["geometry_registration"])
    from launch_omega_camera_streams import SOURCES

    sources = [
        *list(SOURCES),
        "starVLA/dataloader/episode_split.py",
        "tools/launch_rotate_qrcode_streams.py",
        "tools/register_rotate_qrcode_geometry.py",
        "tools/omega_camera_stream_status.py",
        "tools/rotate_qrcode_metrics.py",
        "tools/rollout_omega_camera_streams.py",
        "tools/evaluate_omega_camera_stream_actions.py",
        "tools/prepare_rotate_qrcode_evaluation.py",
        "tools/run_omega_stream_evaluation.py",
        "tests/test_rotate_qrcode_experiment.py",
        str(PROTOCOL.relative_to(ROOT)),
    ]
    m["code_sha256"] = {}
    for name in sources:
        source = ROOT / name
        dest = path.parent / "code" / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, dest)
        m["code_sha256"][name] = digest(source)
    m["status"] = "running"
    m["queue_workers"] = []
    for r in m["runs"]:
        r["status"] = "queued"
    write(path, m)
    for gpu in ["0", "1"]:
        with (path.parent / f"queue_gpu{gpu}.log").open("w") as stream:
            child = subprocess.Popen(
                [str(PYTHON), str(Path(__file__).resolve()), "--worker-manifest", str(path), "--gpu", gpu],
                cwd=ROOT,
                env=environment(gpu),
                stdin=subprocess.DEVNULL,
                stdout=stream,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        update(path, lambda value, child=child, gpu=gpu: value["queue_workers"].append(dict(gpu=gpu, pid=child.pid)))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--steps", type=int, default=20000)
    p.add_argument("--launch-manifest", type=Path)
    p.add_argument("--worker-manifest", type=Path)
    p.add_argument("--gpu")
    args = p.parse_args()
    if args.worker_manifest:
        worker(args.worker_manifest.resolve(), args.gpu)
    elif args.launch_manifest:
        launch(args.launch_manifest.resolve())
        print(args.launch_manifest.resolve())
    else:
        if args.steps < 10000:
            raise ValueError("Budget must preserve the 10000-update milestone")
        print(prepare(args))


if __name__ == "__main__":
    main()
