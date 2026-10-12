"""Summarize intervention measurements and current matched-control launch state."""

import argparse
import datetime
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    args = parser.parse_args()
    directory = args.directory.resolve()
    reports = {
        name: json.loads((directory / f"{name}_probe.json").read_text()) for name in ("dense", "state_query_local")
    }
    scenes = json.loads((directory / "controlled_scenes/scenes.json").read_text())
    manifest = json.loads((directory / "training_controls/manifest.json").read_text())
    summary = dict(
        observed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        expert_successful_interventions=sum(row["expert_success"] for row in scenes),
        total_interventions=len(scenes),
        heldout_episodes=[18, 21, 27],
        noise_seeds=[4000, 4001, 4002],
        models={},
        training=[],
    )
    conditions = ["full", "none", "state_only", "decoder_only"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    for axis, (variant, report) in zip(axes, reports.items(), strict=False):
        groups = report["heldout_summary"]
        selected = [name for name in conditions if name in groups]
        means = [groups[name]["right_joint_target_rmse_rad"] for name in selected]
        spread = []
        for name in selected:
            values = [
                np.mean(
                    [
                        row["metrics"]["right_joint_target_rmse_rad"]
                        for row in report["heldout_records"]
                        if row["condition"] == name and row["noise_seed"] == seed
                    ]
                )
                for seed in report["noise_seeds"]
            ]
            spread.append(np.std(values))
        axis.bar(
            range(len(selected)),
            means,
            yerr=spread,
            capsize=4,
            color=["#4477aa", "#ee6677", "#bbbbbb", "#228833"][: len(selected)],
        )
        axis.set_xticks(
            range(len(selected)),
            [
                dict(
                    full="Original",
                    none="State masked",
                    state_only="Visual memory\nmasked",
                    decoder_only="Query state\nmasked",
                )[name]
                for name in selected
            ],
        )
        axis.set_ylabel("Mean right-joint chunk RMSE (rad)")
        axis.set_title(variant.replace("_", " ").capitalize())
        axis.set_ylim(0, 0.85)
        axis.grid(axis="y", alpha=0.2)
        controlled = [row for row in report["controlled_scene_records"] if row["state_conditioning"] == "full"]
        translated = [row for row in controlled if row["scene"] in ("x_minus", "x_plus", "y_plus")]
        summary["models"][variant] = dict(
            heldout=groups,
            initial_frame=report["initial_frame_summary"],
            later_frame=report["later_frame_summary"],
            mean_expert_chunk_ee_position_error_m=float(np.mean([row["ee_position_rmse_m"] for row in controlled])),
            translated_scenes_opposed_expert_response=sum(row["endpoint_response_cosine"] < 0 for row in translated),
            translated_scene_comparisons=len(translated),
        )
    fig.suptitle(
        "Six held-out observations, three diffusion noise seeds\nError bars: variation between noise-seed means; not"
        " confidence intervals",
        fontsize=10,
    )
    fig.savefig(directory / "conditioning_interventions.png", dpi=160)
    plt.close(fig)
    for run in manifest["runs"]:
        path = Path(run["log"])
        matches = re.findall(r"(\d+)/10000", path.read_text(errors="replace")[-300000:]) if path.exists() else []
        summary["training"].append(
            dict(
                variant=run["variant"],
                status=run["status"],
                gpu=run.get("gpu"),
                pid=run.get("pid"),
                step=int(matches[-1]) if matches else None,
                output=run["checkpoint_dir"],
            )
        )
    (directory / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    table = []
    for label, key in [
        ("Original", "full"),
        ("All explicit state masked", "none"),
        ("Visual memory masked", "state_only"),
        ("Query state masked; decoder state retained", "decoder_only"),
    ]:
        values = [
            (
                f"{report['heldout_summary'][key]['right_joint_target_rmse_rad']:.3f}"
                if key in report["heldout_summary"]
                else "—"
            )
            for report in reports.values()
        ]
        table.append(f"| {label} | {values[0]} | {values[1]} |")
    jobs = "\n".join(
        f"| {row['variant']} | {row['status']} | {row['gpu']} | {row['pid']} | {row['step']} |"
        for row in summary["training"]
    )
    text = f"""# Omega512 state-conditioning investigation

Updated {summary['observed_utc']}. Measurements below are completed diagnostics. Training statuses are a timestamped observation; the manifest is the live queue record.

## What was implemented

Explicit conditioning switches (`state_conditioning=full/none/decoder_only/readout_only`, `visual_conditioning=true/false`) mask entire projected state tokens/query offsets or the entire visual memory. They preserve model dimensions, parameter names, initialization order, and checkpoint compatibility. The default full model behavior is preserved. State-only masks query residuals as well, avoiding indirect state leakage through visual memory. An optional local diffusion generator makes rollout noise deterministic by chunk index without changing default sampling.

## Completed intervention results

Three held-out episodes (18, 21, 27), two frames each, and three fixed diffusion seeds were tested per checkpoint. The values are averages of chunk-level right-joint RMSE in radians. There are three independent episodes, not eighteen independent trajectories.

| Input intervention | Dense | Query/local |
|---|---:|---:|
{chr(10).join(table)}

![Conditioning intervention measurements](conditioning_interventions.png)

Removing all explicit state worsens held-out expert-action predictions for both checkpoints. This establishes functional dependence on state, but does not establish that including state during training caused failure. Masking only query-state offsets improves query/local mean joint error across all three tested diffusion seeds. That result identifies a candidate harmful route, with a small sample and a masking combination the original training did not explicitly use. Visual-memory masking is also a stress test outside the normal full-conditioning training distribution.

The simulator generated five physically consistent scenes: baseline, object X -4 cm, X +4 cm, Y +3 cm, and world-Z rotation +0.25 rad. All three cameras are rerendered; object identity, instruction and commanded initial robot state are fixed. The simulator expert succeeds in **5/5** scenes. Against the expert first-chunk end-effector response to the three translations, dense predictions oppose that response in **9/9** comparisons and query/local in **5/9**, across three diffusion seeds. This is stronger evidence of incorrect visual targeting than a black-image test, but is limited to one object/initial robot pose.

![Rendered object interventions, ordered base/X-/X+/Y+/yaw+](controlled_scenes/target_interventions.png)

Static FK compares sixteen-command predicted chunks with simulator-generated expert chunks. The expert reference uses the current simulator's recording cadence; it is not asserted to reproduce the historical training cadence. FK evaluates commanded geometry, not tracking dynamics, grasp success, or end-effector contact. These diagnostics do not establish seen-scene rollout success.

## Correction to the state description and data coverage

RoboTwin `get_left_arm_jointState/get_right_arm_jointState` read joint **drive targets**, whereas the `*_real_jointState` functions read physical joint positions. The observation vector and converted demonstration states use the drive-target path. The converter pairs state at t with command at t+1. Calling these measured physical joint states was inaccurate. Command history can be predictive of the next command even without successful contact.

Both the converted data audit and original source `scene_info.json` contain **100 right-arm episodes, zero left-arm episodes**. This is a limitation of the source pool, rather than an identified conversion drop of left-arm episodes. The causal controls preserve this pool; evaluation uses known right-side seeds 200000/300000. Full-task training will require new left-arm demonstrations and regenerated normalization statistics.

## Fresh matched training controls

Dense architecture is held fixed. The existing seed-42 dense training is the full-conditioning reference. Two fresh controls train without explicit state or without visual information. Neither starts from an old action checkpoint. Three seed-43 configs reproduce full, vision-only, and state-only conditioning for an independent training repetition.

All controls use the original 90/10 episode split (validation seed 42), cameras/native preprocessing, task instruction, optimizer, batch/accumulation, and frozen Omega512 reference. Each uses **10000 updates / 80000 samples**, with **5000/10000 resumable saves**. The masks consume the same training-dropout RNG draws; they do not change module initialization. Vision-only still sees robot pose in RGB.

| Run | Status | GPU | PID | Logged step |
|---|---|---:|---:|---:|
{jobs}

The local queue automatically evaluates each selected finished checkpoint on the same two right-side scenes, saves videos, and reevaluates the original dense reference with the same diffusion seed 4000 + chunk index. Original full/query evaluation was 0/3 and used an uncontrolled sampling seed, so it is not substituted for this matched evaluation. Inference uses 20 DDIM steps and executes eight actions before replanning. Two right-side scenes are an initial smoke test, not a comprehensive success-rate estimate.

Five new runs would require about 100 GiB with the current complete checkpoint policy. Home had about 44 GiB free, so the first two controls are selected for durable home storage; seed-43 configs are prepared pending a storage choice. The queue reserves checkpoint space and will not silently fill the disk. No GAAS job was submitted.

GPU preflights for both selected conditions pass: correct native images/task, two real gradient updates, expected zero gradients on masked routes, finite DDIM predictions, frozen Omega gradients, and strict policy reload. **27 focused tests passed in separate processes**, including masking, fixed-noise behavior, client ordering/chunk seeds, configuration compatibility, and resume equivalence. Combining those suites in one process exposed their incompatible Accelerate singleton initialization; isolated suite runs passed.

## How to interpret the pending comparison

Evidence for harmful explicit-state reliance would be improved visual response **and** improved matched rollout performance after retraining the vision-only model, repeated across training seeds. A low-error state-only control would quantify how much the expert data can be predicted through the shortcut. All-state masking at inference worsening error alone does not establish overreliance. If vision-only training also fails, prioritize visual target supervision, adapter/expert replay validation, and grasp-specific data rather than treating state removal as a solution.

The candidate remedies remain conditional on this comparison: test stronger state dropout, a vision-only object-goal prediction branch with simulator labels, and recovery demonstrations from displaced objects/imperfect approaches. Those follow-up interventions were not mixed into the causal controls.

## Artifacts and reproduction

- [Dense probe with actions and FK](dense_probe.json), [query/local probe](state_query_local_probe.json), [scene/expert verification](controlled_scenes/scenes.json).
- [Live training manifest](training_controls/manifest.json), [summary snapshot](summary.json), [vision-only GPU preflight](vision_only_preflight.json), [state-only GPU preflight](state_only_preflight.json).
- Diagnostics: `tools/robotwin_omega_intervention_scenes.py` (RoboTwin environment), `tools/omega512_causal_probe.py` (starVLA environment). Run scene capture to a fresh output directory, probe each checkpoint against those scenes, then FK scoring via `--predictions`.
- Refresh this report: `python tools/summarize_omega512_causal.py --directory {directory}`.
- Queue PID is {manifest.get('worker_pid')}; logs are in `training_controls/`. `tools/omega512_training_status.py --manifest training_controls/manifest.json` reports cursors and complete periodic states. Extra configs are not automatically submitted to GAAS.
"""  # noqa: E501 - preserve embedded commands or diagnostic text
    (directory / "report.md").write_text(text)
    print(directory / "report.md")


if __name__ == "__main__":
    main()
