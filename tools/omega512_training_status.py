"""Report local Omega training cursors without loading model weights."""

import argparse
import datetime
import json
import re
import subprocess
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    original = json.loads(args.manifest.read_text())
    report = {"observed_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(), "runs": []}
    for run in original["runs"]:
        root = Path(run["checkpoint_dir"])
        log = Path(run["log"]).read_text(errors="replace")
        matches = re.findall(r"(\d+)/10000", log[-300000:])
        pid = run.get("pid")
        running = pid is not None and Path(f"/proc/{pid}").exists()
        markers = list(root.glob("checkpoints/steps_*/complete.json"))
        best = root / "best_model/training_state/complete.json"
        report["runs"].append(
            {
                "variant": run["variant"],
                "gpu": run["gpu"],
                "pid": pid,
                "process_exists": running,
                "logged_step": int(matches[-1]) if matches else None,
                "target_steps": 10000,
                "log": run["log"],
                "checkpoint_dir": str(root),
                "best_saved_state": json.loads(best.read_text()) if best.exists() else None,
                "complete_periodic_states": [str(p.parent) for p in sorted(markers)],
            }
        )
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,memory.used,memory.total,utilization.gpu", "--format=csv"],
        capture_output=True,
        text=True,
    )
    report["gpu_status"] = result.stdout.strip()
    text = json.dumps(report, indent=2) + "\n"
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text)
    print(text)


if __name__ == "__main__":
    main()
