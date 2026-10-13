"""Verify scenes, serve a fixed 10K checkpoint, record and decode eight rollouts."""

import argparse
import datetime
import json
import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

from prepare_omega_joint24_gaas import digest

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, required=True)
    parser.add_argument("--variant", choices=["A", "C"], required=True)
    args = parser.parse_args()
    manifest_path = args.study.resolve() / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    run = next(r for r in manifest["runs"] if r["variant"] == args.variant)
    out = Path(manifest["result_root"]) / args.variant
    out.mkdir(exist_ok=False)
    status_path = out / "status.json"
    state = dict(status="starting", variant=args.variant, pbs_job_id=os.environ.get("PBS_JOBID"),
                 hostname=os.uname().nodename, checkpoint=run["checkpoint"], checkpoint_step=10000,
                 completed_episodes=0, source_commit=manifest["source_commit"])
    server, child = None, None

    def update(status, **kwargs):
        state.update(status=status, updated_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), **kwargs)
        temporary = status_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(state,indent=2)+"\n")
        temporary.replace(status_path)

    def terminate(signum, frame):
        raise SystemExit(128+signum)

    def finish(process):
        if process is not None and process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()

    signal.signal(signal.SIGTERM, terminate)
    signal.signal(signal.SIGINT, terminate)
    env = dict(os.environ, STARVLA_FRAMEWORK_MODULES="starVLA.model.framework.VLM4A.OmegaCameraStreamDiT",
               ROBOTWIN_PATH=manifest["simulator_root"], ROBOTWIN_SUPPRESS_EVAL_CONFIG="1",
               PYTHONUNBUFFERED="1", OMP_NUM_THREADS="4", STARVLA_DISABLE_DEEPSPEED="1",
               NO_ALBUMENTATIONS_UPDATE="1",
               PYTHONPATH=str(ROOT)+":"+manifest["simulator_root"]+":"+os.environ.get("PYTHONPATH", ""),
               PATH=str(Path(manifest["simulator_python"]).parent)+":"+os.environ["PATH"])
    env.pop("DISPLAY", None)
    # The simulator owns its native dependencies; only the policy needs runtime_site.
    sim_env = dict(env, PYTHONPATH=str(ROOT)+":"+manifest["simulator_root"])
    port = 5780 if args.variant == "A" else 5781
    sim_command = [manifest["simulator_python"], str(ROOT/"tools/rollout_omega_camera_streams.py"),
                   "--manifest", str(manifest_path), "--variant", args.variant, "--port", str(port),
                   "--robotwin-root", manifest["simulator_root"]]

    def simulate(command, log):
        nonlocal child
        with (out/log).open("w") as stream:
            child = subprocess.Popen(command,cwd=ROOT,env=sim_env,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
            code = child.wait()
        child = None
        if code:
            raise RuntimeError(f"Simulator exited {code}; see {out/log}")

    try:
        update("verifying_inputs")
        for key, hash_key in (("checkpoint","checkpoint_sha256"), ("config","config_sha256"),
                              ("normalization","normalization_sha256")):
            assert digest(run[key]) == run[hash_key], f"Changed {key}"
        assert digest(manifest["scene_reference_data"]) == manifest["scene_reference_sha256"]
        update("scene_preflight")
        simulate(sim_command+["--verify-scenes-only","--out",str(out/"scene_preflight")], "scene_preflight.log")
        preflight = json.loads((out/"scene_preflight/results.json").read_text())
        assert len(preflight) == 8 and all(row["status"] == "passed" for row in preflight)
        update("loading_checkpoint", scene_preflight_passed=True)
        with (out/"server.log").open("w") as stream:
            server = subprocess.Popen(
                [manifest["policy_python"], str(ROOT/"deployment/model_server/server_policy.py"),
                 "--ckpt_path",run["checkpoint"],"--port",str(port),"--use_bf16"],
                cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
        deadline = time.monotonic()+600
        while time.monotonic() < deadline:
            if server.poll() is not None:
                raise RuntimeError(f"Server exited {server.returncode}; see {out/'server.log'}")
            try:
                with socket.create_connection(("127.0.0.1",port),timeout=1):
                    break
            except OSError:
                time.sleep(2)
        else:
            raise TimeoutError("Policy server did not become ready within 600 seconds")
        update("rollouts", server_pid=server.pid, port=port)
        simulate(sim_command+["--out",str(out/"rollouts")], "rollouts.log")
        results = json.loads((out/"rollouts/results.json").read_text())
        assert len(results) == 8 and sum(r["split"]=="test" for r in results) == 3
        assert sum(r["split"]=="train" for r in results) == 5
        finish(server)
        server = None
        update("validating_videos", completed_episodes=8)
        import av
        video_frames = {}
        for row in results:
            with av.open(row["video"]) as container:
                count = sum(1 for _ in container.decode(video=0))
            assert count >= 2, row["video"]
            video_frames[row["video"]] = count
        summary = {split:dict(successes=sum(r["success"] for r in results if r["split"]==split),
                             episodes=sum(r["split"]==split for r in results)) for split in ("test","train")}
        (out/"summary.json").write_text(json.dumps(dict(variant=args.variant,checkpoint_step=10000,
                outcomes=summary,decoded_video_frames=video_frames,episodes=results),indent=2)+"\n")
        update("completed",completed_episodes=8,outcomes=summary,video_validation="passed")
    except BaseException as error:
        update("failed",failed_phase=state["status"],error=repr(error))
        raise
    finally:
        finish(child)
        finish(server)


if __name__ == "__main__":
    main()
