"""Foreground Omega rollouts with videos, suitable for PBS GPU allocations.

Uses the existing evaluator worker and its fixed task instruction/renderer
contract. RoboTwin and its separate simulator environment must be installed.
"""

import argparse
import datetime
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import launch_robotwin_compressor_evaluation as evaluation  # noqa: E402 - runtime environment/path bootstrap precedes imports


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--robotwin-root", type=Path, default=os.environ.get("ROBOTWIN_PATH"))
    parser.add_argument("--robotwin-python", type=Path, default=os.environ.get("ROBOTWIN_PYTHON"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--gpu", default="0", help="GPU index within the scheduler allocation")
    parser.add_argument("--port", type=int, default=5710)
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--seed-indices", help="Explicit comma-separated simulator seed indices; overrides --samples")
    parser.add_argument("--config-override", action="append", default=[], help="Serving-only KEY=VALUE override")
    parser.add_argument("--task", default="rotate_qrcode")
    args = parser.parse_args()
    if not args.robotwin_root or not args.robotwin_python:
        parser.error("Specify ROBOTWIN_PATH and ROBOTWIN_PYTHON or the corresponding options")
    if not args.checkpoint.is_file() or args.samples < 1:
        parser.error("Checkpoint must exist and sample count must be positive")
    evaluation.ROBOTWIN = args.robotwin_root.resolve()
    evaluation.ROBOTWIN_PYTHON = args.robotwin_python.resolve()
    evaluator = evaluation.ROBOTWIN / "scripts/eval_policy_xpolicylab.py"
    if not evaluator.is_file() or not evaluation.ROBOTWIN_PYTHON.is_file():
        parser.error("Missing RoboTwin evaluator or simulator interpreter")
    directory = args.out.resolve()
    directory.mkdir(parents=True, exist_ok=False)
    (directory / "videos").mkdir()
    checkpoint = args.checkpoint.resolve()
    indices = list(range(args.samples)) if args.seed_indices is None else [int(x) for x in args.seed_indices.split(",")]
    if not indices or min(indices) < 0 or len(set(indices)) != len(indices):
        parser.error("Seed indices must be unique nonnegative integers")
    # Both variants use the same checkpoint filename; distinguish their run
    # directories so simultaneous rollouts cannot share simulator outputs.
    identity = f"{checkpoint.parents[1].name}_evaluation_{datetime.datetime.now(datetime.timezone.utc):%Y%m%d_%H%M%S}"
    state = {
        "task": args.task,
        "gpu": args.gpu,
        "port": args.port,
        "checkpoint": str(checkpoint),
        "directory": str(directory),
        "evaluation_id": identity,
        "seed_indices": indices,
        "simulator_seeds": [100000 * (i + 1) for i in indices],
        "simulator_output": str(evaluation.ROBOTWIN / "eval_result" / args.task / "starVLA/demo_clean" / identity),
        "video_folder": str(directory / "videos"),
        "server_log": str(directory / "server.log"),
        "framework_module": "starVLA.model.framework.VLM4A.Omega512DiT",
        "status": "starting",
        "episodes": [],
        "completed_episodes": 0,
        "success_count": 0,
        "server_command": [
            sys.executable,
            str(ROOT / "deployment/model_server/server_policy.py"),
            "--ckpt_path",
            str(checkpoint),
            "--port",
            str(args.port),
        ],
    }
    # The worker isolates logical GPU 0; the fixed renderer uses that same GPU.
    for override in args.config_override:
        state["server_command"].extend(["--config_override", override])
    state["config_overrides"] = args.config_override
    os.environ["ROBOTWIN_PATH"] = str(evaluation.ROBOTWIN)
    state_path = directory / "status.json"
    evaluation.write_state(state_path, state)
    evaluation.worker(state_path)
    import av

    result = json.loads(state_path.read_text())
    try:
        decoded = {}
        for video in (directory / "videos").glob("*.mp4"):
            with av.open(str(video)) as container:
                count = sum(1 for _ in container.decode(video=0))
            if count < 2:
                raise RuntimeError(f"Empty or truncated rollout video: {video}")
            decoded[str(video)] = count
        result["decoded_video_frames"] = decoded
        result["video_validation"] = "passed"
    except Exception as error:
        result.update(status="failed", video_validation="failed", error=str(error))
        raise
    finally:
        evaluation.write_state(state_path, result)
    print(state_path, flush=True)


if __name__ == "__main__":
    main()
