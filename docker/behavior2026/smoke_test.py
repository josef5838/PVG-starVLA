#!/usr/bin/env python3
"""Exercise the BEHAVIOR health and WebSocket policy contracts."""

import argparse
import json
import urllib.request

import numpy as np
import websockets.sync.client

from deployment.model_server.tools import msgpack_numpy


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", default=8000, type=int)
    parser.add_argument("--timeout", default=120.0, type=float)
    return parser.parse_args()


def main():
    args = parse_args()
    with urllib.request.urlopen(
        f"http://{args.host}:{args.port}/healthz", timeout=args.timeout
    ) as response:
        if response.status != 200:
            raise RuntimeError(f"Health check returned HTTP {response.status}")

    packer = msgpack_numpy.Packer()
    observation = {
        "robot_r1::zed_link:Camera:0::rgb": np.zeros((224, 224, 3), dtype=np.uint8),
        "robot_r1::left_realsense_link:Camera:0::rgb": np.zeros((224, 224, 3), dtype=np.uint8),
        "robot_r1::right_realsense_link:Camera:0::rgb": np.zeros((224, 224, 3), dtype=np.uint8),
        "robot_r1::proprio": np.zeros(61, dtype=np.float32),
        "task_id": np.array([0], dtype=np.int64),
    }
    with websockets.sync.client.connect(
        f"ws://{args.host}:{args.port}", open_timeout=args.timeout, max_size=None
    ) as websocket:
        metadata = msgpack_numpy.unpackb(websocket.recv(), strict_map_key=False)
        websocket.send(packer.pack(observation))
        result = msgpack_numpy.unpackb(websocket.recv(), strict_map_key=False)

    action = np.asarray(result["action"])
    if action.shape != (23,) or not np.isfinite(action).all():
        raise RuntimeError(f"Invalid action response: shape={action.shape}, action={action}")
    print(
        json.dumps(
            {
                "health": "ok",
                "metadata": metadata,
                "action_shape": list(action.shape),
                "action_finite": True,
                "server_timing": result.get("server_timing"),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

