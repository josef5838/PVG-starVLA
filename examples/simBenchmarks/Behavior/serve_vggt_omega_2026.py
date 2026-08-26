#!/usr/bin/env python3
"""Serve a StarVLA checkpoint using the official BEHAVIOR 2026 protocol."""

import argparse
import asyncio
import http
import json
import logging
import time
import traceback
from pathlib import Path

import numpy as np
import torch
import websockets
import websockets.asyncio.server
from PIL import Image

from deployment.model_server.tools import msgpack_numpy
from starVLA.model.framework.base_framework import baseframework


LOGGER = logging.getLogger("behavior2026_policy")

CAMERA_KEYS = (
    "robot_r1::zed_link:Camera:0::rgb",
    "robot_r1::left_realsense_link:Camera:0::rgb",
    "robot_r1::right_realsense_link:Camera:0::rgb",
)
STATE_KEY = "robot_r1::proprio"


def load_task_descriptions(path: Path):
    descriptions = {}
    with path.open() as f:
        for line in f:
            if line.strip():
                row = json.loads(line)
                descriptions[int(row["task_index"])] = row["task"]
    if not descriptions:
        raise ValueError(f"No task descriptions found in {path}")
    return descriptions


def resize_rgb(image, size: int):
    array = np.asarray(image)
    if array.ndim != 3 or array.shape[-1] < 3:
        raise ValueError(f"Expected an HxWx3 RGB image, got {array.shape}")
    array = array[..., :3]
    if np.issubdtype(array.dtype, np.floating) and array.size and array.max() <= 1.0:
        array = array * 255.0
    array = np.clip(array, 0, 255).astype(np.uint8)
    return np.asarray(Image.fromarray(array).resize((size, size), Image.Resampling.BILINEAR))


def normalize_min_max(values, stats):
    values = np.asarray(values, dtype=np.float32)
    low = np.asarray(stats["min"], dtype=np.float32)
    high = np.asarray(stats["max"], dtype=np.float32)
    if values.shape[-1] != low.size:
        raise ValueError(f"State has {values.shape[-1]} dimensions, expected {low.size}")
    mask = low != high
    output = np.zeros_like(values)
    output[..., mask] = 2.0 * (values[..., mask] - low[mask]) / (high[mask] - low[mask]) - 1.0
    return output


def unnormalize_actions(values, stats):
    values = np.clip(np.asarray(values, dtype=np.float32), -1.0, 1.0)
    low = np.asarray(stats["min"], dtype=np.float32)
    high = np.asarray(stats["max"], dtype=np.float32)
    mask = np.asarray(stats.get("mask", low != high), dtype=bool)
    return np.where(mask, 0.5 * (values + 1.0) * (high - low) + low, values)


class Behavior2026Policy:
    def __init__(self, model, task_descriptions, norm_key, execute_horizon, image_size):
        self.model = model
        self.task_descriptions = task_descriptions
        self.execute_horizon = execute_horizon
        self.image_size = image_size
        if norm_key not in model.norm_stats:
            raise KeyError(f"Normalization key {norm_key!r} not found; available: {list(model.norm_stats)}")
        self.stats = model.norm_stats[norm_key]
        self._actions = None
        self._action_index = 0

    def reset(self):
        self._actions = None
        self._action_index = 0

    def _predict_chunk(self, obs):
        missing = [key for key in (*CAMERA_KEYS, STATE_KEY, "task_id") if key not in obs]
        if missing:
            raise KeyError(f"BEHAVIOR observation is missing keys: {missing}")

        task_index = int(np.asarray(obs["task_id"]).reshape(-1)[0])
        if task_index not in self.task_descriptions:
            raise KeyError(f"No natural-language description for task index {task_index}")

        images = [resize_rgb(obs[key], self.image_size) for key in CAMERA_KEYS]
        state = np.asarray(obs[STATE_KEY], dtype=np.float32).reshape(-1)
        if state.size != 61:
            raise ValueError(f"Official R1Pro proprioception must have 61 dimensions, got {state.size}")
        state = normalize_min_max(state, self.stats["state"])[None, :]

        result = self.model.predict_action(
            examples=[{"image": images, "state": state, "lang": self.task_descriptions[task_index]}]
        )
        actions = unnormalize_actions(result["normalized_actions"][0], self.stats["action"])
        if actions.ndim != 2 or actions.shape[1] != 23:
            raise ValueError(f"Expected a (T, 23) action chunk, got {actions.shape}")
        return actions

    def act(self, obs):
        if self._actions is None or self._action_index >= min(self.execute_horizon, len(self._actions)):
            self._actions = self._predict_chunk(obs)
            self._action_index = 0
        action = self._actions[self._action_index]
        self._action_index += 1
        return action.astype(np.float32, copy=False)


def health_check(connection, request):
    if getattr(request, "path", None) == "/healthz":
        if hasattr(connection, "respond"):
            return connection.respond(http.HTTPStatus.OK, "OK\n")
        return http.HTTPStatus.OK, {"Content-Type": "text/plain"}, b"OK\n"
    return None


class Behavior2026Server:
    def __init__(self, policy, host, port):
        self.policy = policy
        self.host = host
        self.port = port

    def serve_forever(self):
        asyncio.run(self.run())

    async def run(self):
        LOGGER.info("Starting BEHAVIOR 2026 policy server on %s:%d", self.host, self.port)
        async with websockets.asyncio.server.serve(
            self.handle,
            self.host,
            self.port,
            compression=None,
            max_size=None,
            process_request=health_check,
        ) as server:
            await server.serve_forever()

    async def handle(self, websocket):
        packer = msgpack_numpy.Packer()
        await websocket.send(packer.pack({"policy": "VGGTOmegaOFT", "benchmark": "BEHAVIOR-1K-2026"}))
        while True:
            try:
                request = msgpack_numpy.unpackb(await websocket.recv(), strict_map_key=False)
                if request.get("reset", False):
                    self.policy.reset()
                    continue
                start = time.monotonic()
                action = self.policy.act(request)
                response = {
                    "action": action,
                    "server_timing": {"infer_ms": (time.monotonic() - start) * 1000.0},
                }
                await websocket.send(packer.pack(response))
            except websockets.ConnectionClosed:
                break
            except Exception:
                LOGGER.error("Policy request failed:\n%s", traceback.format_exc())
                await websocket.close(code=1011, reason="Policy inference failed")
                raise


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt-path", required=True)
    parser.add_argument("--tasks-jsonl", required=True, type=Path)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--norm-key", default="new_embodiment")
    parser.add_argument("--execute-horizon", type=int, default=8)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument(
        "--dtype",
        choices=("auto", "bf16", "fp16", "fp32"),
        default="auto",
        help="Inference dtype; auto uses BF16 when supported, otherwise FP16 on CUDA",
    )
    parser.add_argument(
        "--bf16",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Deprecated compatibility alias for --dtype bf16/--dtype fp32",
    )
    return parser.parse_args()


def resolve_dtype(name: str, device: str):
    device_type = torch.device(device).type
    if name == "auto":
        if device_type == "cuda":
            return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        return torch.float32
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[name]
    if device_type == "cpu" and dtype == torch.float16:
        raise ValueError("FP16 inference is not supported on CPU; use --dtype fp32")
    if device_type == "cuda" and dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
        raise ValueError("This CUDA device does not support BF16; use --dtype auto or fp16")
    return dtype


def main():
    args = parse_args()
    if args.execute_horizon < 1 or args.execute_horizon > 16:
        raise ValueError("--execute-horizon must be in [1, 16]")
    if args.bf16 is not None:
        LOGGER.warning("--bf16/--no-bf16 is deprecated; use --dtype instead")
        args.dtype = "bf16" if args.bf16 else "fp32"
    dtype = resolve_dtype(args.dtype, args.device)
    LOGGER.info("Loading policy on %s with dtype %s", args.device, dtype)
    model = baseframework.from_pretrained(args.ckpt_path)
    model = model.to(device=args.device, dtype=dtype).eval()
    policy = Behavior2026Policy(
        model=model,
        task_descriptions=load_task_descriptions(args.tasks_jsonl),
        norm_key=args.norm_key,
        execute_horizon=args.execute_horizon,
        image_size=args.image_size,
    )
    Behavior2026Server(policy, args.host, args.port).serve_forever()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main()
