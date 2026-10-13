"""Capture compact, dated evidence for GAAS A/C training and milestone rollouts."""

import argparse
import csv
import datetime
import hashlib
import io
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    artifacts = []

    def capture(source, target):
        source = Path(source)
        if not source.is_file():
            return None
        data = source.read_bytes()
        assert len(data) < 8*1024*1024, f"Unexpected large evidence file: {source}"
        if source.suffix == ".json":
            value = json.loads(data)
        else:
            value = data.decode()
        destination = out/target
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(data)
        artifacts.append(dict(original_path=str(source), snapshot_path=target, bytes=len(data),
                              sha256=hashlib.sha256(data).hexdigest()))
        return value

    def job_status(job_id, target):
        result = subprocess.run(["qstat", "-xf", "-F", "json", job_id],capture_output=True,text=True)
        if result.returncode:
            status = dict(job_id=job_id, lookup_error=result.stderr.strip())
        else:
            job = json.loads(result.stdout)["Jobs"][job_id]
            status = {key:job.get(key) for key in ("job_state", "Exit_status", "comment", "exec_host",
                                                  "resources_used", "stime", "estimated")}
            status["job_id"] = job_id
        path = out/target
        path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(json.dumps(status,indent=2)+"\n")
        return status

    observed = datetime.datetime.now(datetime.timezone.utc)
    summary = dict(observed_utc=observed.isoformat(),
                   observed_sgt=observed.astimezone(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),
                   snapshot_type="Dated reads of live files; not a synchronized optimizer/checkpoint transaction",
                   training={}, evaluations={})
    histories = {}
    for variant, directory in (("A","rotate_qrcode_joint24_gaas_20261012"),
                               ("C","rotate_qrcode_highway_C_gaas_20261012")):
        study = ROOT/"results/omega_camera_stream_training"/directory
        prefix = "training/"+variant
        manifest = capture(study/"manifest.json", prefix+"/manifest.json")
        for name in ("production.yaml", "preflight.json", "preflight_dataset_statistics.json",
                     "batch_tuning.json", "cpu_data_audit.json", "cpu_validation.json", "pbs_submission.json"):
            capture(study/name, prefix+"/"+name)
        source_hashes = study/"source_hashes.json"
        if not source_hashes.exists():
            source_hashes = Path(manifest["source_snapshot"])/"source_hashes.json"
        capture(source_hashes, prefix+"/source_hashes.json")
        output = Path(manifest["output_dir"])
        status = capture(output/"stream_training_status.json", prefix+"/stream_training_status.json")
        best = capture(output/"best_model/validation.json", prefix+"/best_validation.json")
        for name in ("dataset_statistics.json", "config.full.yaml", "initial_sample_order.jsonl"):
            capture(output/name, prefix+"/"+name)
        capture(output/"latest_training_state/complete.json",prefix+"/latest_training_state_complete.json")
        raw_metrics = capture(output/"stream_training_metrics.jsonl", prefix+"/stream_training_metrics.jsonl")
        rows = [json.loads(line) for line in raw_metrics.splitlines() if line.strip()]
        histories[variant] = {row["step"]:row for row in rows if "validation_action_mse" in row}
        fields = ["step", "training_samples", "validation_action_mse", "validation_full_action_mse",
                  "validation_samples", "learning_rate/base"]
        csv_text = io.StringIO()
        writer = csv.DictWriter(csv_text,fieldnames=fields,extrasaction="ignore")
        writer.writeheader()
        writer.writerows(histories[variant].values())
        (out/prefix/"validation_history.csv").write_text(csv_text.getvalue())
        pbs = job_status(manifest["pbs_job_id"],prefix+"/pbs_status.json")
        summary["training"][variant] = dict(job_id=manifest["pbs_job_id"],pbs=pbs,
            status=status["status"],completed_updates=status["completed_updates"],
            target_updates=status["target_updates"],consumed_observations=status["consumed_observations"],
            best_validation=best,latest_validation=list(histories[variant].values())[-1],
            peak_allocated_gpu_gib=status["peak_allocated_gpu_gib"],
            source_commit=manifest["source_commit"],output_dir=str(output))
    common = sorted(histories["A"].keys() & histories["C"].keys())
    summary["matched_validation"] = [dict(step=step,
        A=histories["A"][step]["validation_action_mse"],C=histories["C"][step]["validation_action_mse"])
        for step in common]
    study = ROOT/"results/omega_camera_stream_evaluation/rotate_qrcode_AC_10000_20261013"
    manifest = capture(study/"manifest.json","evaluation/manifest.json")
    for name in ("cpu_validation.json", "source_hashes.json", "pbs_submission_A.json", "pbs_submission_C.json"):
        capture(study/name,"evaluation/"+name)
    for receipt in manifest["pbs_jobs"]:
        variant = receipt["variant"]
        output = Path(manifest["result_root"])/variant
        prefix = "evaluation/"+variant
        state = capture(output/"status.json",prefix+"/status.json")
        preflight = capture(output/"scene_preflight/results.json",prefix+"/scene_preflight.json") or []
        results = capture(output/"rollouts/results.json",prefix+"/rollout_results.json") or []
        capture(output/"summary.json",prefix+"/completed_summary.json")
        pbs = job_status(receipt["job_id"],prefix+"/pbs_status.json")
        summary["evaluations"][variant] = dict(job_id=receipt["job_id"],pbs=pbs,
            phase=state["status"] if state else "queued",source_commit=manifest["source_commit"],
            checkpoint_step=10000,expected_episodes=8,completed_episodes=len(results),
            scene_preflight_passed=len(preflight)==8 and all(r["status"]=="passed" for r in preflight),
            partial_outcomes={split:dict(successes=sum(r["success"] for r in results if r["split"]==split),
                                       completed=sum(r["split"]==split for r in results),
                                       requested=3 if split=="test" else 5) for split in ("test","train")},
            episode_results=[{key:r[key] for key in ("episode","seed","split","success","steps",
                             "grasp_contact","lift_4cm","phase_diagnostics")} for r in results],
            output_dir=str(output))
    (out/"artifact_index.json").write_text(json.dumps(artifacts,indent=2)+"\n")
    (out/"summary.json").write_text(json.dumps(summary,indent=2)+"\n")
    print(json.dumps(dict(snapshot=str(out),observed_sgt=summary["observed_sgt"],
                         training={v:{k:r[k] for k in ("completed_updates","best_validation")} for v,r in summary["training"].items()},
                         evaluations={v:{k:r[k] for k in ("phase","completed_episodes","partial_outcomes")} for v,r in summary["evaluations"].items()}),indent=2))


if __name__ == "__main__":
    main()
