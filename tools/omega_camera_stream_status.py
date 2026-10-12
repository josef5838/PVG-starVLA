"""Read progress for a launched independent-camera conditioning experiment."""

import argparse
import json
from pathlib import Path


def process_alive(pid, expected_argument):
    if not pid:
        return False
    try:
        return str(expected_argument).encode() in (Path("/proc") / str(pid) / "cmdline").read_bytes()
    except (FileNotFoundError, ProcessLookupError, PermissionError):
        return False


def read_status(manifest_path):
    """Reconcile historical queue registrations with actual trainer/process state."""
    manifest = json.loads(manifest_path.read_text())
    rows = []
    sample_orders = {}
    for run in manifest["runs"]:
        progress = Path(run["output_dir"]) / "stream_training_status.json"
        row = json.loads(progress.read_text()) if progress.exists() else dict(status=run.get("status", "prepared"))
        row["process_alive"] = process_alive(run.get("pid"), run["config"])
        if row["status"] in {"starting", "running"} and not row["process_alive"]:
            row["status"] = "interrupted"
        row.update(variant=run["variant"], gpu=run["gpu"], pid=run["pid"], log=run["log"], output_dir=run["output_dir"])
        rows.append(row)
        order_path = Path(run["output_dir"]) / "initial_sample_order.jsonl"
        if order_path.exists():
            order = [json.loads(x) for x in order_path.read_text().splitlines() if x]
            if len(order) >= 16:
                sample_orders[run["variant"]] = order[:16]
    order_audit = dict(
        compared_variants=list(sample_orders),
        matched=len(sample_orders) >= 2
        and all(order == next(iter(sample_orders.values())) for order in sample_orders.values()),
        all_variants_available=len(sample_orders) == len(manifest["runs"]),
    )
    workers = [
        dict(worker, process_alive=process_alive(worker.get("pid"), manifest_path.resolve()))
        for worker in manifest.get("queue_workers", [])
    ]
    status = manifest["status"]
    outcomes = {row["status"] for row in rows}
    if outcomes <= {"completed", "cancelled"}:
        status = "completed_with_cancellations" if "cancelled" in outcomes else "completed"
    elif "interrupted" in outcomes:
        status = "interrupted"
    return dict(
        manifest=str(manifest_path.resolve()),
        status=status,
        registered_status=manifest["status"],
        queue_workers=workers,
        initial_sample_order_audit=order_audit,
        runs=rows,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(read_status(args.manifest), indent=2))


if __name__ == "__main__":
    main()
