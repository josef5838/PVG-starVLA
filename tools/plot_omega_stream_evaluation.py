"""Plot paired trajectory errors and valid-target errors by prediction horizon."""

import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    root = args.manifest.resolve().parent
    metrics = {
        name: json.loads((root / name / "offline/best_sample_metrics.json").read_text()) for name in ["direct", "fused"]
    }
    data = {name: np.load(root / name / "offline/best_predictions.npz") for name in metrics}
    np.testing.assert_array_equal(data["direct"]["episode_frame"], data["fused"]["episode_frame"])
    np.testing.assert_array_equal(data["direct"]["normalized_target"], data["fused"]["normalized_target"])
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    for split, marker in [("train", "o"), ("heldout_validation", "*")]:
        eps = sorted({r["episode"] for r in metrics["direct"] if r["split"] == split})
        values = {
            name: [np.mean([r["valid_normalized_mse"] for r in metrics[name] if r["episode"] == ep]) for ep in eps]
            for name in metrics
        }
        axes[0].scatter(
            values["direct"], values["fused"], marker=marker, label=split, s=60 if marker == "*" else 25, alpha=0.8
        )
    bound = axes[0].get_xlim() + axes[0].get_ylim()
    lo = max(0.0001, min(bound))
    hi = max(bound)
    axes[0].plot([lo, hi], [lo, hi], "k--", lw=1)
    axes[0].set_xscale("log")
    axes[0].set_yscale("log")
    axes[0].set_xlabel("Direct: trajectory mean MSE")
    axes[0].set_ylabel("Fused: trajectory mean MSE")
    axes[0].set_title("Paired valid-future errors (above line favors direct)")
    for j, split in enumerate(["train", "heldout_validation"], start=1):
        indices = [i for i, r in enumerate(metrics["direct"]) if r["split"] == split]
        mask = np.asarray([np.arange(16) < metrics["direct"][i]["valid_horizon"] for i in indices])
        for name in metrics:
            errors = (
                (data[name]["normalized_prediction"][indices] - data[name]["normalized_target"][indices]) ** 2
            ).mean(axis=-1)
            curve = (errors * mask).sum(axis=0) / mask.sum(axis=0)
            axes[j].plot(np.arange(1, 17), curve, label=name, marker=".", ms=4)
        axes[j].set_xlabel("Future action index")
        axes[j].set_ylabel("Normalized action MSE")
        axes[j].set_title(split.replace("_", " "))
    for ax in axes:
        ax.grid(alpha=0.2)
        ax.legend()
    fig.tight_layout()
    fig.savefig(root / "offline_comparison.png", dpi=180)
    fig.savefig(root / "offline_comparison.pdf")
    plt.close(fig)
    # Inspect complete physical rollouts separately from teacher forcing.
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    for ax, split in zip(axes, ["train", "heldout_validation", "unseen_eval"], strict=False):
        for name in metrics:
            path = root / name / "rollouts/results.json"
            records = json.loads(path.read_text()) if path.exists() else []
            for index, r in enumerate([r for r in records if r["split"] == split]):
                trace = json.loads(Path(r["trace"]).read_text())
                ax.plot(
                    [x["step"] for x in trace],
                    [x["functional_point"][2] for x in trace],
                    color="tab:blue" if name == "direct" else "tab:orange",
                    alpha=0.5,
                    label=name if index == 0 else None,
                )
        ax.axhline(0.9, color="black", ls="--", lw=1, label="height criterion")
        ax.set_title(split.replace("_", " "))
        ax.set_xlabel("Control steps")
        ax.grid(alpha=0.2)
        ax.legend()
    axes[0].set_ylabel("Bottle functional point height (m)")
    fig.suptitle("Bottle height in closed-loop rollouts (success also requires target side)")
    fig.tight_layout()
    fig.savefig(root / "rollout_heights.png", dpi=180)
    fig.savefig(root / "rollout_heights.pdf")
    plt.close(fig)
    print(root / "offline_comparison.png")


if __name__ == "__main__":
    main()
