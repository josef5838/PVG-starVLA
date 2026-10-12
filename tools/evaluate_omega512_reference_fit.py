"""Evaluate each registered training scene at all three fixed diffusion seeds."""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from launch_omega512_minimum_case import (  # noqa: E402
    finish,
    validate_videos,
    wait_server,
)
from launch_omega512_protocol import (  # noqa: E402
    PYTHON,
    SIM_PYTHON,
    write,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--step", required=True, type=int)
    parser.add_argument("--episodes", help="Comma-separated registered training episodes; defaults to all eight")
    parser.add_argument("--diffusion-seeds", help="Comma-separated seeds; defaults to the protocol seeds")
    parser.add_argument("--output-dir", type=Path, help="Separate destination for an additional evaluation")
    parser.add_argument("--port", type=int, default=5780)
    parser.add_argument("--match-training-image-size", action="store_true")
    args = parser.parse_args()
    args.manifest = args.manifest.resolve()
    manifest = json.loads(args.manifest.read_text())
    cfg = manifest["policy"]
    registered = [r["training_episode"] for r in cfg["selected_episodes"]]
    chosen = [int(x) for x in args.episodes.split(",")] if args.episodes else registered
    if not chosen or len(chosen) != len(set(chosen)) or not set(chosen).issubset(registered):
        parser.error("--episodes must contain unique registered training episodes")
    seeds = [int(x) for x in args.diffusion_seeds.split(",")] if args.diffusion_seeds else cfg["rollout_noise_seeds"]
    if not seeds or len(seeds) != len(set(seeds)):
        parser.error("Diffusion seeds must be nonempty and unique")
    out = args.output_dir.resolve() if args.output_dir else args.manifest.parent / "rollouts" / f"step_{args.step}"
    if (out / "status.json").exists():
        raise FileExistsError(f"Evaluation destination already used: {out}")
    out.mkdir(parents=True, exist_ok=True)
    checkpoint = Path(cfg["checkpoint_dir"]) / "checkpoints" / f"steps_{args.step}_pytorch_model.pt"
    state = checkpoint.parent / f"steps_{args.step}" / "complete.json"
    assert checkpoint.exists() and state.exists()
    status = dict(
        status="running",
        checkpoint=str(checkpoint),
        step=args.step,
        seeds={},
        episodes=chosen,
        match_training_image_size=args.match_training_image_size,
        started_unix=time.time(),
    )
    write(out / "status.json", status)
    server = None
    simulator = None
    port = args.port
    try:
        with (out / "policy_server.log").open("w") as stream:
            server = subprocess.Popen(
                [
                    str(PYTHON),
                    str(ROOT / "deployment/model_server/server_policy.py"),
                    "--ckpt_path",
                    str(checkpoint),
                    "--port",
                    str(port),
                ],
                cwd=ROOT,
                stdout=stream,
                stderr=subprocess.STDOUT,
            )
        wait_server(server, port)
        episodes = ",".join(map(str, chosen))
        all_records = []
        for seed in seeds:
            name = f"reference_{args.step}_seed{seed}"
            dest = out / f"seed_{seed}"
            dest.mkdir(exist_ok=True)
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
                str(dest),
                "--scene-manifest",
                str(args.manifest),
                "--episodes",
                episodes,
                "--execute-horizon",
                "8",
                "--diffusion-seed",
                str(seed),
            ]
            if args.match_training_image_size:
                command.append("--match-training-image-size")
            with (out / f"seed_{seed}.log").open("w") as stream:
                simulator = subprocess.Popen(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
            if simulator.wait():
                raise RuntimeError(f"Rollouts failed at seed {seed}; inspect its log")
            records = json.loads((dest / f"{name}_results.json").read_text())
            assert len(records) == len(chosen)
            validate_videos(records)
            all_records += records
            status["seeds"][str(seed)] = dict(
                successes=sum(r["success"] for r in records),
                grasps=sum(r["grasp_contact"] for r in records),
                lifts=sum(r["lift_4cm"] for r in records),
                episodes=len(chosen),
            )
            write(out / "status.json", status)
        write(
            out / "results.json",
            dict(
                records=all_records,
                total=len(all_records),
                successes=sum(r["success"] for r in all_records),
                grasps=sum(r["grasp_contact"] for r in all_records),
                lifts=sum(r["lift_4cm"] for r in all_records),
            ),
        )
        status.update(status="completed", finished_unix=time.time())
        write(out / "status.json", status)
    except Exception as error:
        status.update(status="failed", error=repr(error))
        write(out / "status.json", status)
        raise
    finally:
        finish(simulator)
        finish(server)


if __name__ == "__main__":
    main()
