"""
Precompute frozen visual-backbone token features for a RoboTwin LeRobot dataset.

Because the Agilex pipeline applies no random image augmentation, the frozen
VGGT-Omega backbone produces identical token features for a frame on every
epoch. Caching them once lets training skip the (1B-param) backbone
forward entirely — a large speedup — and the cache is reused across every head /
hierarchy level that shares the same backbone+task.

Output (co-located with the dataset, auto-discovered via the
`datasets.vla_data.visual_feature_cache: <tag>` config key):

    <dataset_path>/cache_visual/<tag>/features.npy   memmap float16 [N, N_tokens, dim]
    <dataset_path>/cache_visual/<tag>/index.json     {"key_to_row": {"<traj>|<base>": row}, ...}

Usage:
    conda run -n starVLA python examples/simBenchmarks/Robotwin/precompute_visual_features.py \
        --config_yaml examples/simBenchmarks/Robotwin/train_files/phase2_2a.yaml
    # optional: --tag vggt_omega  --batch 32
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from numpy.lib.format import open_memmap
from omegaconf import OmegaConf

from starVLA.dataloader.lerobot_datasets import get_vla_dataset
from starVLA.model.framework.share_tools import apply_config_compat
from starVLA.model.framework.VLM4A.VGGTOmegaOFT import VGGTOmega_OFT


@torch.no_grad()
def precompute_for_dataset(model, single, tag, batch_size, device, np_dtype=np.float32):
    cache_dir = single.dataset_path / "cache_visual" / str(tag)
    cache_dir.mkdir(parents=True, exist_ok=True)
    feat_path = cache_dir / "features.npy"
    index_path = cache_dir / "index.json"

    steps = list(single.all_steps)              # [(traj, base), ...] deterministic
    N = len(steps)

    # Probe one frame for token shape (N_tokens, dim).
    def frame_images(traj, base):
        data = single.transforms(single.get_step_data(traj, base))
        return single._pack_sample(data)["image"]   # list[PIL], no cache during precompute

    probe = model._pil_to_tensor_stack(frame_images(*steps[0])).unsqueeze(0).to(device)
    probe_tokens = model._encode_visual_tokens(probe)        # (1, N_tokens, dim)
    n_tokens, dim = probe_tokens.shape[1], probe_tokens.shape[2]
    itemsize = np.dtype(np_dtype).itemsize
    print(f"  {single.dataset_name}: N={N} frames, tokens={n_tokens}, dim={dim} "
          f"→ ~{N * n_tokens * dim * itemsize / 1e9:.1f} GB ({np.dtype(np_dtype).name})")

    feats = open_memmap(feat_path, mode="w+", dtype=np_dtype, shape=(N, n_tokens, dim))
    key_to_row = {}

    for start in range(0, N, batch_size):
        chunk = steps[start:start + batch_size]
        imgs = torch.stack([
            model._pil_to_tensor_stack(frame_images(traj, base)) for traj, base in chunk
        ]).to(device)                                        # (b, T, 3, H, W)
        tokens = model._encode_visual_tokens(imgs)           # (b, N_tokens, dim)
        feats[start:start + len(chunk)] = tokens.float().cpu().numpy().astype(np_dtype)
        for j, (traj, base) in enumerate(chunk):
            key_to_row[f"{traj}|{base}"] = start + j
        print(f"    {min(start + len(chunk), N)}/{N}", end="\r")

    feats.flush()
    del feats
    with open(index_path, "w") as f:
        json.dump({
            "key_to_row": key_to_row,
            "n_rows": N,
            "n_tokens": n_tokens,
            "dim": dim,
            "dtype": np.dtype(np_dtype).name,
            "backbone": tag,
        }, f)
    print(f"\n  wrote {feat_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config_yaml", required=True)
    ap.add_argument("--tag", default=None, help="cache tag (default: visual_backbone.type)")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--device", default="cuda")
    ap.add_argument(
        "--dtype", default="float32", choices=["float32", "float16"],
        help="float32 (default) avoids truncating VGGT massive-activation tokens; "
             "float16 halves disk at a small train/eval precision gap.",
    )
    args = ap.parse_args()

    cfg = OmegaConf.load(args.config_yaml)
    apply_config_compat(cfg)

    tag = args.tag or cfg.framework.visual_backbone.type
    print(f"[precompute] backbone tag = {tag}")

    # Build the framework (only the frozen backbone is exercised).
    model = VGGTOmega_OFT(cfg).to(args.device).eval()

    # Build the dataset(s); iterate each single dataset's frames.
    np_dtype = np.float32 if args.dtype == "float32" else np.float16
    mixture = get_vla_dataset(data_cfg=cfg.datasets.vla_data)
    for single in mixture.datasets:
        precompute_for_dataset(model, single, tag, args.batch, args.device, np_dtype=np_dtype)

    print("[precompute] done.")


if __name__ == "__main__":
    main()
