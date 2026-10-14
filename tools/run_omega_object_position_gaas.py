"""Register or run an immutable A20K controlled object-position diagnostic."""

import argparse
import datetime
import hashlib
import json
import os
import signal
import socket
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def digest(file):
    value = hashlib.sha256()
    with Path(file).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def prepare(args):
    reference = args.reference_study.resolve() / "manifest.json"
    previous = json.loads(reference.read_text())
    assert json.loads((Path(previous["result_root"]) / "conditioning_comparison.json").read_text())["status"] == "completed"
    run = next(r for r in previous["runs"] if r["variant"] == "full")
    assert run["model"] == "A" and run["checkpoint_step"] == 20000
    assert previous["training_completion"]["completed_updates"] == 20000
    assert previous["training_completion"]["consumed_observations"] == 160000
    for key in ("checkpoint", "config", "normalization"):
        assert digest(run[key]) == run[key + "_sha256"], f"Changed {key}"
    assert digest(previous["scene_reference_data"]) == previous["scene_reference_sha256"]
    cases = [dict(name=name, offset_m=offset) for name, offset in (
        ("base", [0, 0, 0]), ("x_minus_1cm", [-0.01, 0, 0]),
        ("x_plus_1cm", [0.01, 0, 0]), ("y_minus_2cm", [0, -0.02, 0]),
        ("y_minus_4cm", [0, -0.04, 0]))]
    for scene in previous["rollout_scenes"]:
        assert scene["arm"] == "right"
        for case in cases:
            x, y = [scene["position"][i] + case["offset_m"][i] for i in (0, 1)]
            assert 0.05 <= x <= 0.25 and -0.2 <= y <= 0
    conditions = [dict(name=name, state_conditioning=state, visual_conditioning=visual) for name, state, visual in (
        ("full", "full", True), ("vision_only", "none", True), ("state_only", "full", False))]
    keys = ["task", "simulator_root", "simulator_python", "policy_python", "camera_order",
            "scene_reference_data", "scene_reference_sha256", "initial_image_validation", "rollout_scenes"]
    manifest = {k: previous[k] for k in keys}
    manifest.update(status="prepared", model="A", checkpoint_step=20000, run=run,
        reference_study=str(reference), reference_manifest_sha256=digest(reference),
        reference_evaluation_completed=True, native_camera_contract=True,
        result_root=str(args.result_root.resolve()), position_cases=cases, conditions=conditions,
        noise_seeds=[4000, 4001, 4002], num_ddim_steps=20, camera_history_length=8,
        total_captured_cases=40, total_policy_predictions=360, total_repeat_checks=72,
        created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        protocol="Matched initial images, repeated history, unchanged command state and physical qpos, task and noise. All three cameras rerendered after moving only the target actor without stepping physics. x +/-1cm and y -2/-4cm retain all eight scenes in the demonstrated right-arm workspace.",
        interpretation="Compare commanded world-frame TCP responses with object displacement and paired expert responses at horizons 8 and 16. State-only predictions must be invariant. Static FK is not policy rollout success; initial expert approach need not track the full object displacement. Failed expert references are retained and excluded only from successful-expert-pair summaries.")
    args.study.resolve().mkdir(parents=True, exist_ok=False)
    args.result_root.resolve().mkdir(parents=True, exist_ok=False)
    (args.study / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(dict(study=str(args.study.resolve()), captured_cases=40, policy_predictions=360, repeat_checks=72)))


def run(args):
    manifest = json.loads((args.study / "manifest.json").read_text())
    out = Path(manifest["result_root"])
    assert not (out / "status.json").exists(), "Refusing duplicate worker"
    state = dict(status="starting", model="A", checkpoint_step=20000, pbs_job_id=os.environ.get("PBS_JOBID"),
                 hostname=socket.gethostname(), source_commit=manifest["source_commit"])
    child = None

    def update(status, **extra):
        state.update(status=status, updated_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), **extra)
        temp = out / "status.tmp"
        temp.write_text(json.dumps(state, indent=2) + "\n")
        temp.replace(out / "status.json")

    def terminate(signum, frame):
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGTERM, terminate)
    signal.signal(signal.SIGINT, terminate)
    env = dict(os.environ, STARVLA_FRAMEWORK_MODULES="starVLA.model.framework.VLM4A.OmegaCameraStreamDiT",
               ROBOTWIN_PATH=manifest["simulator_root"], ROBOTWIN_SUPPRESS_EVAL_CONFIG="1",
               PYTHONUNBUFFERED="1", OMP_NUM_THREADS="4", STARVLA_DISABLE_DEEPSPEED="1",
               NO_ALBUMENTATIONS_UPDATE="1", PYTEST_DISABLE_PLUGIN_AUTOLOAD="1",
               PYTHONPATH=str(ROOT) + ":" + os.environ.get("PYTHONPATH", ""))
    env.pop("DISPLAY", None)
    sim_env = dict(env, PYTHONPATH=str(ROOT) + ":" + manifest["simulator_root"],
                   PATH=str(Path(manifest["simulator_python"]).parent) + ":" + env["PATH"])

    def execute(command, name, environment):
        nonlocal child
        with (out / (name + ".log")).open("w") as stream:
            child = subprocess.Popen(command, cwd=ROOT, env=environment, stdout=stream,
                                     stderr=subprocess.STDOUT, start_new_session=True)
            code = child.wait()
        child = None
        if code:
            raise RuntimeError(f"{name} exited {code}; see {out / (name + '.log')}")

    try:
        update("verifying_inputs")
        for key in ("checkpoint", "config", "normalization"):
            assert digest(manifest["run"][key]) == manifest["run"][key + "_sha256"]
        assert digest(manifest["scene_reference_data"]) == manifest["scene_reference_sha256"]
        update("cpu_preflight")
        execute([manifest["policy_python"], "-m", "pytest", "-q",
                 "tests/test_omega_conditioning_interventions.py", "tests/test_omega_object_position_metrics.py"],
                "cpu_preflight", env)
        update("capturing_scenes")
        sim = [manifest["simulator_python"], str(ROOT / "tools/robotwin_omega_object_position.py"),
               "--study", str(args.study.resolve())]
        execute(sim + ["--phase", "capture"], "capture", sim_env)
        scenes = json.loads((out / "scenes/scenes.json").read_text())
        assert len(scenes) == 40 and all(r["state_max_difference"] == 0 and r["physical_qpos_max_difference"] == 0 for r in scenes)
        update("predicting", captured_cases=40, expert_successes=sum(r["expert_success"] for r in scenes))
        execute([manifest["policy_python"], str(ROOT / "tools/probe_omega_object_position.py"),
                 "--study", str(args.study.resolve())], "probe", env)
        update("scoring_forward_kinematics", policy_predictions=360)
        execute(sim + ["--phase", "score"], "score", sim_env)
        report = json.loads((out / "object_position_comparison.json").read_text())
        assert report["status"] == "completed" and len(report["records"]) == 360
        assert len(report["reproducibility_checks"]) == 72
        update("completed", policy_predictions=360, repeat_checks=72, summary=report["summary"])
    except BaseException as error:
        update("failed", failed_phase=state["status"], error=repr(error))
        raise
    finally:
        if child is not None and child.poll() is None:
            os.killpg(child.pid, signal.SIGTERM)
            try:
                child.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, required=True)
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--reference-study", type=Path)
    parser.add_argument("--result-root", type=Path)
    args = parser.parse_args()
    if args.prepare and (args.reference_study is None or args.result_root is None):
        parser.error("--prepare requires --reference-study and --result-root")
    prepare(args) if args.prepare else run(args)


if __name__ == "__main__":
    main()
