"""Place four reconstruction models in one image without inventing missing results."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def group(root, out, label, unfiltered=False):
    base = np.load(root / "render_frames.npz")
    raw_path = root / "raw_depth_meshes/render_frames.npz"
    raw = np.load(raw_path) if unfiltered else None
    cases = [("Surflo + guidance", base, "surflo"), ("VGGT", raw if unfiltered else base, "vggt")]
    for resolution in [256, 512]:
        path = root / f"omega_{resolution}/render_frames.npz"
        if unfiltered:
            cases.append(
                (f"VGGT Omega {resolution}", raw if f"omega{resolution}_0" in raw.files else None, f"omega{resolution}")
            )
        else:
            cases.append((f"VGGT Omega {resolution}", np.load(path) if path.exists() else None, "omega"))
    fig, axes = plt.subplots(3, 5, figsize=(23, 10), gridspec_kw={"width_ratios": [1, 2, 2, 2, 2]})
    for row, camera in enumerate(["Head", "Left wrist", "Right wrist"]):
        axes[row, 0].imshow(base[f"input_{row}"])
        axes[row, 0].set_ylabel(camera)
        axes[row, 0].set_xticks([])
        axes[row, 0].set_yticks([])
        for col, (name, frames, prefix) in enumerate(cases, start=1):
            axis = axes[row, col]
            if frames is None:
                axis.set_facecolor("#eeeeee")
                axis.text(
                    0.5,
                    0.5,
                    "Pending: checkpoint download\nrequires Hugging Face login",
                    ha="center",
                    va="center",
                    transform=axis.transAxes,
                    color="#555555",
                    fontsize=10,
                )
                axis.set_xticks([])
                axis.set_yticks([])
            else:
                rgb = frames[f"{prefix}_{row}"]
                gray = frames[f"{prefix}_shaded_{row}"]
                shaded = np.repeat(gray[..., None], 3, axis=-1)
                gap = np.ones((len(rgb), 4, 3), dtype=rgb.dtype)
                axis.imshow(np.concatenate([rgb, gap, shaded], axis=1).clip(0, 1))
                axis.axis("off")
            if row == 0:
                axis.set_title(name, fontsize=12)
        if row == 0:
            axes[row, 0].set_title("Input RGB", fontsize=12)
    fig.suptitle(
        f"{label} — same three source camera images\nEach model tile: colored mesh on the left, shaded geometry on the"
        " right",
        fontsize=14,
    )
    filter_note = (
        "VGGT/Omega depth meshes use no confidence filtering. "
        if unfiltered
        else "VGGT/Omega depth meshes use relative confidence filtering. "
    )
    fig.text(
        0.5,
        0.015,
        filter_note
        + "Each model uses its estimated cameras. Surflo includes RGB/depth/normal guidance. Geometry accuracy requires"
        " ground truth.",
        ha="center",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.04, 1, 0.94))
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=180)
    plt.close(fig)
    available = [name for name, frames, _ in cases if frames is not None]
    pending = [name for name, frames, _ in cases if frames is None]
    metadata = dict(
        image=str(out.resolve()),
        source_root=str(root.resolve()),
        available_models=available,
        pending_models=pending,
        status="complete" if not pending else "partial: missing checkpoint",
        depth_confidence_filter="none" if unfiltered else "relative",
        layout="Camera rows; input + four model columns. Each model has RGB and shaded geometry.",
    )
    out.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--unfiltered", action="store_true")
    args = parser.parse_args()
    group(args.root, args.out, args.label, args.unfiltered)


if __name__ == "__main__":
    main()
