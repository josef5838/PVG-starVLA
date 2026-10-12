"""Preserve lightweight local experiment evidence independently of ignored results/."""

import argparse
import csv
import datetime
import hashlib
import io
import json
import platform
import re
import socket
import subprocess
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
EXCLUDED_PARTS = {
    "wandb",
    "code",
    "code_initial",
    "code_final",
    "before_renderer_fix",
    "__pycache__",
    "data_split",
    "normalization_cache",
    "runtime_site",
}
RECORD_NAMES = {
    "manifest.json",
    "summary.json",
    "results.json",
    "metrics.json",
    "report.json",
    "comparison_metrics.json",
    "stage_metrics.json",
    "geometry_metrics.json",
    "probe_results.json",
    "feature_manifest.json",
    "geometry_registration.json",
    "report.md",
    "README.md",
    "protocol.md",
    "execution_protocol.md",
    "interpretation.md",
    "completed_results.md",
    "dataset_statistics.json",
    "config.full.yaml",
    "config.yaml",
    "complete.json",
}
SECRET_KEY = re.compile(r"^(password|api_key|access_token|refresh_token|authorization|private_key)$", re.I)


def redact(value):
    if isinstance(value, dict):
        return {key: "[REDACTED]" if SECRET_KEY.match(key) else redact(item) for key, item in value.items()}
    if isinstance(value, list):
        return [redact(item) for item in value]
    return value


def read_json(path):
    return json.loads(path.read_text())


def json_lines(path):
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(errors="replace").splitlines():
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            # A live writer may not have finished the final line yet.
            continue
    return rows


def process_alive(pid, expected):
    if not pid:
        return False
    path = Path("/proc") / str(pid) / "cmdline"
    try:
        return expected.encode() in path.read_bytes()
    except (FileNotFoundError, ProcessLookupError, PermissionError):
        return False


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(redact(value), indent=2) + "\n")


def selected(path):
    if path.suffix == ".yaml":
        return True
    if path.name in RECORD_NAMES:
        return True
    return path.suffix in {".json", ".md"} and any(
        word in path.name
        for word in ["audit", "validation", "preflight", "amendment", "progress_results", "recovery", "status"]
    )


def training_record(directory):
    cfg_path = directory / "config.full.yaml"
    if not cfg_path.exists():
        cfg_path = directory / "config.yaml"
    cfg = yaml.safe_load(cfg_path.read_text()) if cfg_path.exists() else {}
    cfg = cfg or {}
    rows = json_lines(directory / "stream_training_metrics.jsonl")
    periodic = json_lines(directory / "summary.jsonl")
    status_path = directory / "stream_training_status.json"
    status = read_json(status_path) if status_path.exists() else {}
    recorded_steps = [row.get("step", row.get("steps", 0)) for row in rows + periodic]
    completed = max([status.get("completed_updates", 0), *recorded_steps], default=0)
    final = directory / "final_model/pytorch_model.pt"
    target = cfg.get("trainer", {}).get("max_train_steps")
    outcome = status.get("status", "completion_unverified")
    if final.exists() and outcome not in {"running", "starting"}:
        outcome = "completed" if target and completed >= target else "final_weights_present_budget_unverified"
    if outcome in {"running", "starting"}:
        pid = None
        for manifest_path in (ROOT / "results/omega_camera_stream_training").glob("*/manifest.json"):
            manifest = read_json(manifest_path)
            for run in manifest.get("runs", []):
                if Path(run.get("output_dir", "")).name == directory.name:
                    pid = run.get("pid")
                    if not process_alive(pid, run["config"]):
                        outcome = "interrupted"
        if pid is None:
            outcome = "live_process_unverified"
    validations = [row for row in rows if "validation_action_mse" in row]
    best_file = directory / "best_model/validation.json"
    best = read_json(best_file) if best_file.exists() else None
    if not best and validations:
        row = min(validations, key=lambda item: item["validation_action_mse"])
        best = {"step": row["step"], "validation_action_mse": row["validation_action_mse"]}
    weights = [str(path.relative_to(ROOT)) for path in directory.glob("*/pytorch_model.pt")]
    weights += [str(path.relative_to(ROOT)) for path in (directory / "checkpoints").glob("*_pytorch_model.pt")]
    wandb_summaries = []
    for path in sorted(directory.glob("wandb/**/wandb-summary.json")):
        try:
            wandb_summaries.append(dict(path=str(path.relative_to(ROOT)), metrics=read_json(path)))
        except json.JSONDecodeError:
            continue
    return dict(
        run_id=directory.name,
        status=outcome,
        completed_updates=completed,
        target_updates=target,
        framework=cfg.get("framework", {}).get("name"),
        data_mix=cfg.get("datasets", {}).get("vla_data", {}).get("data_mix"),
        seed=cfg.get("seed"),
        best_validation=best,
        latest_progress=status,
        available_weight_paths=weights,
        wandb_terminal_summaries=wandb_summaries,
        validation_curve=[
            {key: row[key] for key in ("step", "validation_action_mse", "validation_full_action_mse") if key in row}
            for row in validations
        ],
        note="Completion uses final weight evidence; missing weights can reflect historical storage cleanup.",
    )


def snapshot(output, simulator_results=()):
    output.mkdir(parents=True, exist_ok=True)
    results = ROOT / "results"
    artifacts, families, copied, errors = [], {}, 0, []
    original_sources = {}
    for path in sorted(results.rglob("*")):
        if not path.is_file() or path.is_symlink():
            continue
        relative = path.relative_to(results)
        if any(part in {"code", "code_initial", "code_final"} for part in relative.parts):
            if path.suffix in {".py", ".sh", ".md", ".yaml"} and path.stat().st_size <= 512 * 1024:
                data = path.read_bytes()
                digest = hashlib.sha256(data).hexdigest()
                # Text suffix keeps historical code outside runtime lint/test discovery.
                destination = output / "original_sources" / (digest + path.suffix + ".txt")
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(data)
                original_sources[str(path.relative_to(ROOT))] = dict(
                    sha256=digest, evidence_path=str(destination.relative_to(output))
                )
            continue
        if any(part in EXCLUDED_PARTS or part.startswith("preflight_split") for part in relative.parts):
            continue
        family = relative.parts[0]
        entry = dict(path=str(path.relative_to(ROOT)), bytes=path.stat().st_size)
        families.setdefault(family, dict(artifact_count=0, evidence_files=[]))["artifact_count"] += 1
        # Index weights/videos/meshes without reading or copying their contents.
        if selected(path) and path.stat().st_size <= 2 * 1024**2:
            data = path.read_bytes()
            entry["source_sha256"] = hashlib.sha256(data).hexdigest()
            destination = output / "evidence" / relative
            try:
                if path.suffix == ".json":
                    write_json(destination, json.loads(data))
                elif path.suffix == ".yaml":
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    value = redact(yaml.safe_load(data))
                    destination.write_text(yaml.safe_dump(value, sort_keys=False))
                else:
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    destination.write_bytes(data)
                entry["evidence_path"] = str(destination.relative_to(output))
                entry["snapshot_sha256"] = hashlib.sha256(destination.read_bytes()).hexdigest()
                families[family]["evidence_files"].append(entry["evidence_path"])
                copied += 1
            except (ValueError, yaml.YAMLError) as exc:
                errors.append(dict(path=str(relative), error=str(exc)))
        artifacts.append(entry)
    runs = [training_record(path) for path in sorted((results / "Checkpoints").iterdir()) if path.is_dir()]
    external = []
    for root in simulator_results:
        for path in sorted(root.rglob("_result.txt")):
            data = path.read_bytes()
            destination = output / "simulator_results" / root.parent.name / path.relative_to(root)
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(data)
            external.append(
                dict(
                    source=str(path),
                    evidence_path=str(destination.relative_to(output)),
                    sha256=hashlib.sha256(data).hexdigest(),
                )
            )
    inventory = dict(
        captured_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        hostname=socket.gethostname(),
        platform=platform.platform(),
        project_root=str(ROOT),
        git_base=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        branch=subprocess.check_output(["git", "branch", "--show-current"], cwd=ROOT, text=True).strip(),
        scope=(
            "Every artifact discoverable under this project's results/; historical or deleted runs are limited to"
            " retained evidence."
        ),
        families=families,
        training_runs=runs,
        evidence_file_count=copied,
        collection_errors=errors,
        external_simulator_results=external,
        original_source_archive=original_sources,
        interpretation=(
            "Historical manifests may say running after completion. Consult observed training status and dated results,"
            " not old PIDs."
        ),
    )
    write_json(output / "inventory.json", inventory)
    write_json(output / "artifact_index.json", artifacts)
    table = io.StringIO()
    writer = csv.DictWriter(
        table,
        fieldnames=[
            "run_id",
            "framework",
            "data_mix",
            "status",
            "completed_updates",
            "target_updates",
            "best_validation_mse",
            "best_step",
        ],
    )
    writer.writeheader()
    for run in runs:
        best = run["best_validation"] or {}
        writer.writerow(
            {
                **{key: run[key] for key in writer.fieldnames[:6]},
                "best_validation_mse": best.get("validation_action_mse"),
                "best_step": best.get("step"),
            }
        )
    (output / "training_runs.csv").write_text(table.getvalue())
    print(
        json.dumps(
            dict(
                output=str(output), families=len(families), training_runs=len(runs), evidence_files=copied, errors=errors
            ),
            indent=2,
        )
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--simulator-results", type=Path, action="append", default=[])
    args = parser.parse_args()
    snapshot(args.output.resolve(), [path.resolve() for path in args.simulator_results])


if __name__ == "__main__":
    main()
