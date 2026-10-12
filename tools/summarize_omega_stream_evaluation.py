"""Build training/action/control comparison and a paired rollout video gallery."""

import argparse
import datetime
import html
import json
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--require-completed", action="store_true")
    args = parser.parse_args()
    path = args.manifest.resolve()
    root = path.parent
    m = json.loads(path.read_text())
    rows = {}
    offline = {}
    states = {}
    summary = {"training": {}, "offline": {}, "rollouts": {}, "expert_controls": {}}
    for run in m["runs"]:
        name = run["variant"]
        d = root / name
        summary["training"][name] = {
            k: run[k]
            for k in ["best_validation", "final_validation_mse", "last_100_logged_train_loss_mean", "checkpoint_sha256"]
        }
        rows[name] = (
            json.loads((d / "rollouts/results.json").read_text()) if (d / "rollouts/results.json").exists() else []
        )
        offline[name] = (
            json.loads((d / "offline/results.json").read_text()) if (d / "offline/results.json").exists() else {}
        )
        states[name] = {
            stage: json.loads((d / file).read_text()) if (d / file).exists() else {"status": "pending"}
            for stage, file in [("rollouts", "status.json"), ("offline", "offline/status.json")]
        }
        summary["offline"][name] = offline[name]
        summary["rollouts"][name] = {}
        for split in ["train", "heldout_validation", "unseen_eval"]:
            rs = [r for r in rows[name] if r["split"] == split]
            summary["rollouts"][name][split] = dict(
                n=len(rs),
                success=sum(r["success"] for r in rs),
                grasps=sum(r["grasp_contact"] for r in rs),
                lifts=sum(r["lift_4cm"] for r in rs),
                arms={
                    side: dict(
                        n=sum(r["arm"] == side for r in rs), success=sum(r["success"] and r["arm"] == side for r in rs)
                    )
                    for side in ["left", "right"]
                },
            )
    expert_path = root / "direct/expert_controls/results.json"
    expert = json.loads(expert_path.read_text()) if expert_path.exists() else []
    roundtrip_path = root / "direct/expert_roundtrip_controls/results.json"
    roundtrip = json.loads(roundtrip_path.read_text()) if roundtrip_path.exists() else []
    summary["expert_controls"] = {
        split: dict(
            n=sum(r["split"] == split for r in expert), success=sum(r["success"] and r["split"] == split for r in expert)
        )
        for split in ["train", "heldout_validation"]
    }
    summary["expert_roundtrip_controls"] = {
        split: dict(
            n=sum(r["split"] == split for r in roundtrip),
            success=sum(r["success"] and r["split"] == split for r in roundtrip),
        )
        for split in ["train", "heldout_validation"]
    }
    complete = (
        all(
            len(rows[name]) == 15
            and set(offline[name]) == {"best", "final"}
            and all(s["status"] == "completed" for s in states[name].values())
            for name in rows
        )
        and len(expert) == 10
        and len(roundtrip) == 10
    )
    summary.update(
        status="completed" if complete else "in_progress",
        worker_states=states,
        updated_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
    )
    if args.require_completed and not complete:
        raise RuntimeError("Evaluation is not complete; check worker states")
    (root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    lines = [
        "# adjust_bottle: independent Omega camera-stream conditioning",
        "",
        (
            f"Evaluation status: **{summary['status']}**. Both training runs completed 10,000 updates and 80,000 sampled"
            " observations."
        ),
        "",
        (
            "Direct keeps all 408 camera/register tokens; fused uses a two-layer transformer and 32 learned query"
            " tokens. Both use three independent eight-image Omega calls, a frozen backbone, learned camera/time/type"
            " embeddings and the same action decoder. No temporal RoPE."
        ),
        "",
        "## Training outcomes",
        "",
        (
            "| Conditioning | Last 1,000 updates: mean denoising loss | Best validation action MSE | Best update | Final"
            " validation MSE |"
        ),
        "|---|---:|---:|---:|---:|",
    ]
    for run in m["runs"]:
        b = run["best_validation"]
        lines.append(
            f"| {run['variant']} | {run['last_100_logged_train_loss_mean']:.5f} | {b['validation_action_mse']:.5f} |"
            f" {b['step']} | {run['final_validation_mse']:.5f} |"
        )
    lines += [
        "",
        (
            "Both train losses converge. Direct reaches approximately half the validation error of fused. Best"
            " checkpoints are selected by the original 128-example held-out DDIM20 validation, not by rollout outcomes."
            " Direct ran about 17h57m; fused about 17h46m. Peak allocated GPU memory was 11.91 and 12.05 GiB"
            " respectively."
        ),
        "",
        "![Training curves](training_curves.png)",
        "",
        "## Recorded trajectory action prediction",
        "",
        (
            "Teacher forcing with recorded states and chronological per-camera histories: all 45 training episodes,"
            " eight equally spaced anchors each (360); all five held-out episodes, 16 anchors each (80). Predictions are"
            " matched by sample/batch order and diffusion noise. The held-out episodes were used for checkpoint"
            " selection. They are not an independent test set."
        ),
        "",
        (
            "| Conditioning/checkpoint | Split | 16-action MSE | Valid-future MSE | First eight MSE | First action MSE |"
            " Active-arm joint RMSE (rad) | Gripper accuracy |"
        ),
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name in ["direct", "fused"]:
        for kind in ["best", "final"]:
            for split, r in offline[name].get(kind, {}).get("by_split", {}).items():
                lines.append(
                    f"| {name}/{kind} | {split} | {r['normalized_mse']:.5f} | {r['valid_normalized_mse']:.5f} |"
                    f" {r['executed8_normalized_mse']:.5f} | {r['first_action_normalized_mse']:.5f} |"
                    f" {r['active_arm_joint_rmse_rad']:.4f} | {100*r['gripper_accuracy']:.2f}% |"
                )
    lines += [
        "",
        (
            "MSE uses training-normalized actions; lower is better. Valid-future metrics exclude episode-end target"
            " padding. Joint RMSE is the mean of per-anchor RMSEs after inverse normalization; gripper accuracy covers"
            " both grippers. A hold-current-state reference has valid-future MSE 0.10140 on train and 0.11761 on"
            " held-out examples. Noise seed 4000 + batch index differs from the training-time validation seed, and"
            " anchor selection differs, so these values need not equal the saved validation MSE."
        ),
        "",
        "![Paired trajectory errors and action horizon](offline_comparison.png)",
        "",
    ]
    # Paired uncertainty at trajectory level, rather than treating frames as independent.
    if all("best" in offline[n] for n in ["direct", "fused"]):
        metrics = {
            n: json.loads((root / n / "offline/best_sample_metrics.json").read_text()) for n in ["direct", "fused"]
        }
        paired = {}
        for split in ["train", "heldout_validation"]:
            episodes = sorted({r["episode"] for r in metrics["direct"] if r["split"] == split})
            difference = np.array(
                [
                    np.mean([r["valid_normalized_mse"] for r in metrics["fused"] if r["episode"] == ep])
                    - np.mean([r["valid_normalized_mse"] for r in metrics["direct"] if r["episode"] == ep])
                    for ep in episodes
                ]
            )
            bootstrap = np.random.default_rng(42).choice(difference, (10000, len(difference)), replace=True).mean(axis=1)
            paired[split] = dict(
                fused_minus_direct_mean=float(difference.mean()),
                bootstrap95=np.quantile(bootstrap, [0.025, 0.975]).tolist(),
                direct_better_trajectories=int((difference > 0).sum()),
                n=len(difference),
            )
            lines.append(
                f"Paired {split}: direct has lower valid-future MSE on"
                f" {sum(difference>0)}/{len(difference)} trajectories; fused minus direct mean {difference.mean():.5f},"
                f" trajectory-bootstrap 95% interval [{np.quantile(bootstrap,.025):.5f},"
                f" {np.quantile(bootstrap,.975):.5f}]."
            )
        summary["paired_offline"] = paired
    lines += [
        "",
        "## Closed-loop task completion",
        "",
        (
            "Use each validation-selected best checkpoint. Execute eight of each 16-action prediction, replan with"
            " DDIM20, and update image history at every control observation. Resize simulator RGB to dataset 640x480"
            " before unchanged official Omega preprocessing. Both models receive identical initial scenes and noise-seed"
            " schedules. Each case runs to success or the official 400-step limit."
        ),
        "",
        "| Conditioning | Seen training scenes | Held-out validation scenes | Fresh evaluation seeds |",
        "|---|---:|---:|---:|",
    ]
    for name in ["direct", "fused"]:
        cells = [
            f"{summary['rollouts'][name][s]['success']}/{summary['rollouts'][name][s]['n']}"
            for s in ["train", "heldout_validation", "unseen_eval"]
        ]
        lines.append(f"| {name} | {' | '.join(cells)} |")
    if complete:
        lines += [
            "",
            (
                "Direct predicts recorded actions more accurately and solves one additional training scene in this small"
                " rollout set. Both policies complete two of five held-out scenes and two of five fresh scenes; the"
                " 32-token fuser provides no observed success advantage here. Generalization remains weak. These results"
                " demonstrate that camera/register-only conditioning can drive the policy, but comparisons with a"
                " trained state-only policy and RGB/patch-token baselines are needed to establish the benefit of the"
                " geometric backbone."
            ),
            "",
        ]
        successful = {name: {(r["split"], r["seed"]): r for r in rows[name] if r["success"]} for name in rows}
        common = set(successful["direct"]) & set(successful["fused"])
        means = {name: float(np.mean([successful[name][key]["steps"] for key in common])) for name in rows}
        summary["paired_successful_rollout_steps"] = dict(n=len(common), mean_control_steps=means)
        lines.append(
            f"On the {len(common)} scenes that both policies solve, direct uses {means['direct']:.1f} control steps on"
            f" average versus {means['fused']:.1f} for fused. This comparison is conditional on success; failures run to"
            " 400 steps."
        )
    lines += [
        "",
        (
            "Primary endpoint is the unchanged simulator criterion: bottle functional point above 0.9m and on the target"
            " side. Secondary signals are two-gripper-body contact with a closed gripper and any actor-origin lift"
            " >=4cm; they do not replace full success."
        ),
        "",
        "![Closed-loop bottle height](rollout_heights.png)",
        "",
        "| Conditioning / split | Grasp contact | Lift >=4cm | Fresh-seed arm breakdown |",
        "|---|---:|---:|---|",
    ]
    for name in ["direct", "fused"]:
        for split, r in summary["rollouts"][name].items():
            arms = (
                ", ".join(f"{side}: {count['success']}/{count['n']} success" for side, count in r["arms"].items())
                if split == "unseen_eval"
                else "right-arm demonstrations"
            )
            lines.append(f"| {name}/{split} | {r['grasps']}/{r['n']} | {r['lifts']}/{r['n']} | {arms} |")
    lines += ["", "## Failure inspection", ""]
    for name in ["direct", "fused"]:
        failed = [r for r in rows[name] if not r["success"]]
        off_table = [r for r in failed if r["final_object_position"][2] < 0.5]
        no_grasp = [r for r in failed if not r["grasp_contact"]]
        lines.append(
            f"{name}: {len(failed)} failed cases; {len(no_grasp)} have no recorded closed two-finger contact;"
            f" {len(off_table)} finish with the bottle actor below 0.5m. Failed cases: "
            + ", ".join(f"{r['split']} episode {r['episode']} / seed {r['seed']}" for r in failed)
            + "."
        )
    if (root / "direct_train_failure.png").exists():
        lines += [
            "",
            (
                "A representative direct failure is training episode 22 / source episode 24 / seed 80. Its trace records"
                " palm contact at control steps 13–15, first gripper closure at step 21 (recorded expert closure at"  # noqa: RUF001 - intentional mathematical/visual labels
                " command index 26), no closed two-finger contact and at most 2mm lift. The head video shows the bottle"
                " being knocked off the table. These are observed failure signals; they do not isolate the backbone,"
                " readout or controller as the cause."
            ),
            "",
            "![Direct training-scene failure](direct_train_failure.png)",
            "",
        ]
    lines += [
        "",
        "## Expert execution control",
        "",
        (
            "Recorded raw expert actions pass through the same ModelClient chunk scheduling, action-order conversion and"
            " simulator controller. This oracle bypasses learned inference and its normalization; it validates scene"
            " recovery and execution."
        ),
        "",
    ]
    for split, r in summary["expert_controls"].items():
        lines.append(f"- {split}: {r['success']}/{r['n']} expert replays succeed.")
    lines += [
        "",
        (
            "A second oracle uses the exact training action transform followed by inverse normalization. Its joint"
            " commands differ from recorded values by at most 1.8e-7 rad, while gripper commands become binary and"
            " differ by up to 0.4724. This checks the action-target execution path as well as the raw recording."
        ),
        "",
    ]
    for split, r in summary["expert_roundtrip_controls"].items():
        lines.append(f"- {split}: {r['success']}/{r['n']} normalization-roundtrip expert replays succeed.")
    lines += [
        "",
        "## Scope and artifacts",
        "",
        (
            "- All 50 recorded demonstrations use the right arm. The six left-arm action dimensions have min=max=0;"
            " min/max inverse normalization therefore fixes them at zero regardless of predicted normalized values."
            " Fresh left-arm cases are outside the demonstrated distribution and have this additional action-space"
            " restriction; arm-specific counts are reported above."
        ),
        (
            "- Fresh seed starts are advanced only when setup is unstable, without expert-feasibility rejection."
            " Accepted seeds are 100001, 200001, 300000, 400000, 500000. Their results are an unseen-scene diagnostic,"
            " not a full official benchmark submission."
        ),
        (
            "- Rollout histories are sampled per control observation. TOPP determines variable physical elapsed time, so"
            " their eight-frame window is not guaranteed to match nominal training 15Hz."
        ),
        (
            "- Five scenes per subset and one training seed give a small comparison. These runs alone cannot establish"
            " that geometry beats RGB/state alternatives."
        ),
        (
            "- Converted-to-source mappings were checked against every recorded state and next action. Seen/held-out"
            " rollouts also require matching initial RGB, state and bottle model."
        ),
        (
            "- GPU rendering used isolated driver libraries matching kernel 595.71.05 and isolated Warp 1.9.1; system"
            " installations were unchanged."
        ),
        (
            "- [Paired video gallery](videos.html), [machine-readable summary](summary.json), [registered"
            " manifest](manifest.json), [training plots PDF](training_curves.pdf)."
        ),
        (
            "- [Per-trajectory action metrics CSV](trajectory_action_metrics.csv), [all rollout outcomes"
            " CSV](rollout_results.csv), [video validation](video_validation.json), [validation sampling"
            " audit](validation_sampling_audit.json)."
        ),
        (
            "- Raw action predictions, per-anchor metrics and per-step rollout action/state/object traces are under the"
            " direct and fused subdirectories. Videos were fully decoded before completion."
        ),
        "- The evaluation protocol and scripts are archived under code/.",
        "",
    ]
    (root / "report.md").write_text("\n".join(lines))
    sections = []
    for scene in m["rollout_scenes"]:
        tiles = []
        for name in ["direct", "fused"]:
            r = next((r for r in rows[name] if r["seed"] == scene["seed"] and r["split"] == scene["split"]), None)
            if r:
                video = Path(r["video"]).relative_to(root)
                caption = (
                    f"{name}: {'SUCCESS' if r['success'] else 'FAIL'} · {r['steps']} steps · {r['arm']} arm · max lift"
                    f" {r['max_lift_m']*100:.1f}cm"
                )
                tiles.append(
                    f'<article><p>{html.escape(caption)}</p><video controls preload="metadata"'
                    f' src="{html.escape(str(video))}"></video></article>'
                )
            else:
                tiles.append(f"<article><p>{name}: pending</p></article>")
        sections.append(
            f"<section><h2>{scene['split']} · dataset episode {scene['episode']} · seed {scene['seed']}</h2><div"
            f" class='pair'>{''.join(tiles)}</div></section>"
        )
    controls = []
    for r in expert + roundtrip:
        controls.append(
            f"<article><p>{r['mode']} · {r['split']} episode {r['episode']} ·"
            f" {'SUCCESS' if r['success'] else 'FAIL'}</p><video controls preload='metadata'"
            f" src='{Path(r['video']).relative_to(root)}'></video></article>"
        )
    (root / "videos.html").write_text(
        '<!doctype html><html><head><meta charset="utf-8"><title>Omega stream policy'
        " comparison</title><style>body{font-family:system-ui;margin:2rem;background:#fafafa;color:#20252b}section{margin:2rem"  # noqa: E501 - preserve embedded commands or diagnostic text
        " 0}h2{font-size:1.15rem}.pair{display:grid;grid-template-columns:1fr"
        " 1fr;gap:1rem}article{background:white;padding:1rem;border:1px solid"
        " #ddd;border-radius:8px}video{width:100%}a{color:#176cb1}</style></head><body><h1>adjust_bottle: direct vs"
        " fused</h1><p>Validation-selected checkpoints; three independent eight-frame camera streams. <a"
        ' href="report.md">Report</a> · <a href="summary.json">Metrics</a></p>'
        + "".join(sections)
        + '<h2>Recorded expert controls</h2><div class="pair">'
        + "".join(controls)
        + "</div></body></html>"
    )
    (root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    if complete:
        m.update(status="completed", finished_utc=summary["updated_utc"], report=str(root / "report.md"))
        path.write_text(json.dumps(m, indent=2) + "\n")
    print(
        json.dumps(
            {"status": summary["status"], "report": str(root / "report.md"), "rollouts": summary["rollouts"]}, indent=2
        )
    )


if __name__ == "__main__":
    main()
