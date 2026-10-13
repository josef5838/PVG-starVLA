"""Summarize conditioning rollouts and paired active-joint trajectory differences."""

import argparse
import csv
import datetime
import fcntl
import io
import json
import math
import os
from pathlib import Path


def read(path):
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, required=True)
    args = parser.parse_args()
    manifest = read(args.study / "manifest.json")
    root = Path(manifest["result_root"])
    # Serialize captures so a slow partial reader cannot overwrite a later final report.
    lock = (root / ".conditioning_comparison.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX)
    report = dict(observed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                  checkpoint_step=manifest["checkpoint_step"], model="A", variants={}, paired_with_full=[],
                  limitations="Inference masking is out of training distribution. Later trajectories differ in feedback states; action RMSE does not prove useful conditioning. TCP-to-center distance ignores object shape and is not grasp success.")
    rows, results = [], {}
    for run in manifest["runs"]:
        variant = run["variant"]
        status = read(root / variant / "status.json") or {}
        completed = read(root / variant / "rollouts/results.json") or []
        results[variant] = {r["episode"]: r for r in completed}
        report["variants"][variant] = dict(status=status.get("status", "queued"), completed_episodes=len(completed),
            state_conditioning=run["state_conditioning"], visual_conditioning=run["visual_conditioning"],
            splits={split:dict(completed=sum(r["split"] == split for r in completed),
                              successes=sum(r["success"] for r in completed if r["split"] == split),
                              grasp_contacts=sum(r["grasp_contact"] for r in completed if r["split"] == split),
                              lifts_4cm=sum(r["lift_4cm"] for r in completed if r["split"] == split))
                    for split in ("test", "train")})
        for row in completed:
            rows.append(dict(variant=variant, split=row["split"], episode=row["episode"], seed=row["seed"],
                success=row["success"], grasp_contact=row["grasp_contact"], lift_4cm=row["lift_4cm"],
                max_lift_m=row["max_lift_m"], failure_phase=row["phase_diagnostics"]["diagnostic_failure_phase"],
                minimum_tcp_to_object_center_m=row.get("minimum_tcp_to_object_center_m"),
                final_tcp_to_object_center_m=row.get("final_tcp_to_object_center_m"),
                minimum_orientation_error_deg=row["minimum_orientation_error_deg"]))
    for variant in results:
        if variant == "full":
            continue
        for episode in sorted(results["full"].keys() & results[variant].keys()):
            full = read(Path(results["full"][episode]["trace"]))
            masked = read(Path(results[variant][episode]["trace"]))
            pair = dict(variant=variant, episode=episode, seed=results[variant][episode]["seed"])
            for label, start, end in (("first_chunk_rmse_rad",0,8), ("whole_rollout_rmse_rad",0,min(len(full),len(masked)))):
                values = [(a["action"][j]-b["action"][j])**2 for a,b in zip(full[start:end],masked[start:end])
                          for j in range(7,13)]
                pair[label] = math.sqrt(sum(values)/len(values))
            report["paired_with_full"].append(pair)
    report["status"] = "completed" if all(v["status"] == "completed" and v["completed_episodes"] == 8
                                           for v in report["variants"].values()) else "partial"
    def write(name, text):
        target = root / name
        temporary = target.with_name(target.name + f".tmp.{os.getpid()}")
        temporary.write_text(text)
        temporary.replace(target)
    write("conditioning_comparison.json", json.dumps(report, indent=2)+"\n")
    if rows:
        stream = io.StringIO()
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
        write("conditioning_episode_outcomes.csv",stream.getvalue())
    print(json.dumps(dict(status=report["status"], completed={v:d["completed_episodes"] for v,d in report["variants"].items()})))


if __name__ == "__main__":
    main()
