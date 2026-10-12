"""Check a real Omega checkpoint through the websocket deployment interface."""

import argparse
import json
import socket
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from deployment.model_server.tools.websocket_policy_client import (  # noqa: E402
    WebsocketClientPolicy,
)
from starVLA.dataloader.lerobot_datasets import (  # noqa: E402
    get_vla_dataset,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--use-bf16", action="store_true")
    args = parser.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    cfg = OmegaConf.load(args.config)
    cfg.datasets.vla_data.statistics_cache_dir = str(args.out.parent / "serving_data_split")
    batch = get_vla_dataset(cfg.datasets.vla_data, mode="train")[0]
    # Raw zero joint/gripper state checks transport/normalization, not task success.
    observation = {
        "image": [np.asarray(image) for image in batch["image"]],
        "state": np.zeros(14, dtype=np.float32),
        "lang": str(cfg.framework.task_names[0]),
    }
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    command = [
        sys.executable,
        str(ROOT / "deployment/model_server/server_policy.py"),
        "--ckpt_path",
        str(args.checkpoint),
        "--port",
        str(port),
    ]
    if args.use_bf16:
        command.append("--use_bf16")
    client = None
    with args.out.with_suffix(".server.log").open("w") as stream:
        server = subprocess.Popen(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
        try:
            deadline = time.monotonic() + 300
            while True:
                if server.poll() is not None:
                    raise RuntimeError("Server failed; inspect its log")
                try:
                    with socket.create_connection(("127.0.0.1", port), timeout=1):
                        break
                except OSError as exc:
                    if time.monotonic() > deadline:
                        raise TimeoutError("Server readiness timed out") from exc
                    time.sleep(1)
            client = WebsocketClientPolicy(port=port)
            metadata = client.get_server_metadata()
            assert metadata["framework_name"] == "Omega512DiT"
            assert metadata["preserve_native_images"] is True
            assert metadata["instruction_override"] == observation["lang"]
            assert metadata["action_chunk_size"] == 16 and metadata["execute_horizon"] == 8
            actions = np.asarray(client.predict_action({"examples": [observation], "num_ddim_steps": 20})["actions"])
            assert actions.shape == (1, 16, 14) and np.isfinite(actions).all()
            args.out.write_text(
                json.dumps(
                    {
                        "status": "passed",
                        "metadata": metadata,
                        "actions_shape": list(actions.shape),
                        "action_min": float(actions.min()),
                        "action_max": float(actions.max()),
                        "use_bf16": args.use_bf16,
                        "scope": "Interface and finite outputs; raw zero state is not a performance evaluation.",
                    },
                    indent=2,
                )
                + "\n"
            )
        finally:
            if client is not None:
                client.close()
            server.terminate()
            try:
                server.wait(timeout=20)
            except subprocess.TimeoutExpired:
                server.kill()
                server.wait()


if __name__ == "__main__":
    main()
