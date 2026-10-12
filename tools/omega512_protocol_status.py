"""Read status of the two protocol pipelines without loading any model."""

import argparse
import datetime
import json
import subprocess
from pathlib import Path


def read(path):
    return json.loads(path.read_text()) if path.exists() else {}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    args = parser.parse_args()
    manifest = args.manifest.resolve()
    m = read(manifest)
    root = manifest.parent
    policy, probe = m["policy"], m["visual_probe"]
    probe_root = Path(probe["output_dir"])
    report = dict(
        observed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        manifest=str(manifest),
        policy=dict(
            gpu=policy["gpu"],
            pid=policy.get("pid"),
            process_exists=Path(f'/proc/{policy.get("pid")}').exists(),
            status=read(root / "reference_status.json"),
            audit=read(root / "reference_data_audit.json"),
            complete_checkpoints=[
                str(p.parent) for p in sorted(Path(policy["checkpoint_dir"]).glob("checkpoints/steps_*/complete.json"))
            ],
            log=policy["log"],
        ),
        visual_probe=dict(
            gpu=probe["gpu"],
            pid=probe.get("pid"),
            process_exists=Path(f'/proc/{probe.get("pid")}').exists(),
            status=read(probe_root / "status.json"),
            labels=read(probe_root / "labels/status.json"),
            cached_feature_episodes=len(list((probe_root / "features").glob("episode_*.pt"))),
            log=probe["log"],
        ),
    )
    gpu = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,memory.used,utilization.gpu", "--format=csv,noheader"],
        capture_output=True,
        text=True,
    )
    report["gpu_status"] = gpu.stdout.strip()
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
