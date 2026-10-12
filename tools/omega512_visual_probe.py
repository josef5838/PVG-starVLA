"""Frozen visual-only object/EE/displacement probes with trajectory splits."""

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
from launch_omega512_protocol import (  # noqa: E402
    SIM_PYTHON,
    write,
)


class SpatialVisualProbe(nn.Module):
    """Only argument is the fixed-view, spatial RGB feature tensor."""

    def __init__(self, grid=8):
        super().__init__()
        self.token_projection = nn.Sequential(nn.LayerNorm(2048), nn.Linear(2048, 64), nn.SiLU())
        self.head = nn.Sequential(nn.Flatten(1), nn.Linear(3 * grid * grid * 64, 256), nn.SiLU(), nn.Linear(256, 9))

    def forward(self, features):
        return self.head(self.token_projection(features))


def metric(predicted, target):
    groups = {}
    for i, name in enumerate(["object", "gripper_reference", "object_minus_gripper"]):
        error = predicted[:, i * 3 : (i + 1) * 3] - target[:, i * 3 : (i + 1) * 3]
        distance = np.linalg.norm(error, axis=1) * 100
        groups[name] = dict(
            mean_error_cm=float(distance.mean()),
            median_error_cm=float(np.median(distance)),
            p90_error_cm=float(np.quantile(distance, 0.9)),
            axis_rmse_cm=(np.sqrt((error**2).mean(0)) * 100).tolist(),
        )
    groups["prediction_displacement_consistency_cm"] = float(
        np.linalg.norm(predicted[:, :3] - predicted[:, 3:6] - predicted[:, 6:9], axis=1).mean() * 100
    )
    return groups


def extract(manifest, out, status):
    from omegaconf import OmegaConf

    from starVLA.model.framework.VLM4A.Omega512DiT import Omega512DiT

    cfg = OmegaConf.load(manifest["policy"]["config"])
    source = Path(cfg.framework.omega.source_root)
    sys.path.insert(0, str(source))
    from vggt_omega.models import VGGTOmega

    checkpoint = Path(cfg.framework.omega.checkpoint_path)
    hasher = hashlib.sha256()
    with checkpoint.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            hasher.update(block)
    assert hasher.hexdigest() == cfg.framework.omega.checkpoint_sha256
    backbone = (
        VGGTOmega(enable_camera=True, enable_depth=True, enable_alignment=False).cuda().eval().requires_grad_(False)
    )
    backbone.load_state_dict(torch.load(checkpoint, map_location="cpu", mmap=True, weights_only=True), strict=True)
    # Reuse the policy's exact native-PIL preprocessing without constructing an action head.
    holder = SimpleNamespace(visual_proj=nn.Linear(1, 1).cuda())
    feature_dir = out / "features"
    feature_dir.mkdir(exist_ok=True)
    records = []
    for split, rows in manifest["visual_probe"]["splits"].items():
        for row in rows:
            ep = row["training_episode"]
            target = feature_dir / f"episode_{ep:03d}.pt"
            audit = json.loads((out / "labels" / f"episode_{ep:03d}.json").read_text())
            assert audit["verified"]
            labels = np.load(out / "labels" / f"episode_{ep:03d}.npz")
            frames = labels["frames"].tolist()
            values = np.concatenate(
                [
                    labels["object_pose"][frames, :3],
                    labels["right_ee"][frames, :3],
                    labels["object_pose"][frames, :3] - labels["right_ee"][frames, :3],
                ],
                axis=-1,
            ).astype(np.float32)
            if not target.exists():
                features = []
                hashes = []
                for start in range(0, len(frames), 2):
                    examples = []
                    for t in frames[start : start + 2]:
                        images = [Image.fromarray(im) for im in labels["images"][frames.index(t)]]
                        hashes.append([hashlib.sha256(image.tobytes()).hexdigest() for image in images])
                        examples.append(dict(image=images))
                    images = Omega512DiT._prepare_images(holder, examples)
                    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                        layers, patch_start = backbone.aggregator(images)
                        tokens = layers[int(manifest["visual_probe"]["feature_layer"])][:, :, patch_start:]
                        b, v, n, d = tokens.shape
                        h, w = images.shape[-2] // 16, images.shape[-1] // 16
                        assert n == h * w and d == 2048
                        grid = int(manifest["visual_probe"]["spatial_grid"])
                        pooled = F.adaptive_avg_pool2d(tokens.permute(0, 1, 3, 2).reshape(b * v, d, h, w), (grid, grid))
                        features.append(
                            pooled.reshape(b, v, d, grid * grid)
                            .permute(0, 1, 3, 2)
                            .contiguous()
                            .cpu()
                            .to(torch.bfloat16)
                        )
                    del layers, tokens, pooled, images
                torch.save(
                    dict(
                        features=torch.cat(features),
                        targets=torch.from_numpy(values),
                        frames=torch.tensor(frames),
                        source_images_sha256=hashes,
                        source_episode=row["source_episode"],
                        omega_sha256=hasher.hexdigest(),
                        grid=grid,
                    ),
                    target,
                )
            shard = torch.load(target, map_location="cpu", weights_only=True)
            assert torch.equal(shard["frames"], torch.tensor(frames)) and shard["omega_sha256"] == hasher.hexdigest()
            np.testing.assert_allclose(shard["targets"].numpy(), values, rtol=0, atol=1e-7)
            records.append(
                dict(training_episode=ep, split=split, frames=len(frames), feature_shape=list(shard["features"].shape))
            )
            status.update(stage="frozen_feature_extraction", completed_feature_episodes=len(records))
            write(out / "status.json", status)
            print("FEATURES " + json.dumps(records[-1]), flush=True)
    assert all(parameter.grad is None for parameter in backbone.parameters())
    write(
        out / "feature_manifest.json",
        dict(
            records=records,
            omega_sha256=hasher.hexdigest(),
            backbone_frozen=True,
            preprocessing=(
                "Exact Omega512DiT native PIL preparation; three cameras jointly encoded; layer 23, 8x8 grid per view."
            ),
            inputs_exclude=[
                "robot_state",
                "actions",
                "instruction",
                "episode_id",
                "frame_id",
                "time",
                "ground_truth_camera_calibration",
            ],
        ),
    )
    del backbone, holder
    torch.cuda.empty_cache()


def load_split(config, out, name):
    rows = config["splits"][name]
    shards = {
        r["training_episode"]: torch.load(
            out / "features" / f'episode_{r["training_episode"]:03d}.pt', weights_only=True
        )
        for r in rows
    }
    x = torch.cat([shards[r["training_episode"]]["features"] for r in rows]).cuda()
    y = torch.cat([shards[r["training_episode"]]["targets"] for r in rows]).cuda()
    # No identity enters the head: IDs only construct an offline shuffle control.
    shuffled = torch.cat(
        [shards[config["episode_shuffle"][name][str(r["training_episode"])]]["features"] for r in rows]
    ).cuda()
    return x, y, shuffled


def fit(config, out, data, name, shuffled=False, resume=False):
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    np.random.seed(42)
    model = SpatialVisualProbe(config["spatial_grid"]).cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
    train_x, train_y, shuffle_x = data["train"]
    val_x, val_y, shuffle_val = data["validation"]
    mean = train_y.mean(0)
    scale = train_y.std(0).clamp_min(0.01)
    train_x = shuffle_x if shuffled else train_x
    val_x = shuffle_val if shuffled else val_x
    best, stale, start_epoch = np.inf, 0, 1

    def save(path, epoch, best_score):
        temp = path.with_suffix(".tmp")
        torch.save(
            dict(
                model=model.state_dict(),
                optimizer=optimizer.state_dict(),
                epoch=epoch,
                best_validation_error_m=best_score,
                target_mean=mean.cpu(),
                target_scale=scale.cpu(),
                early_stopping_wait=stale,
                torch_rng=torch.get_rng_state(),
                cuda_rng=torch.cuda.get_rng_state(),
                numpy_rng=np.random.get_state(),
                config=config,
                input_signature="features[B,3,64,2048] only",
            ),
            temp,
        )
        temp.replace(path)

    if resume:
        paths = [*list(out.glob(f"{name}_epoch_*.pt")), out / f"{name}_best.pt"]
        states = [torch.load(p, map_location="cpu", weights_only=False) for p in paths if p.exists()]
        if states:
            saved = max(states, key=lambda state: state["epoch"])
            model.load_state_dict(saved["model"])
            optimizer.load_state_dict(saved["optimizer"])
            torch.testing.assert_close(mean.cpu(), saved["target_mean"])
            torch.testing.assert_close(scale.cpu(), saved["target_scale"])
            torch.set_rng_state(saved["torch_rng"])
            torch.cuda.set_rng_state(saved["cuda_rng"])
            np.random.set_state(saved["numpy_rng"])
            best, stale, start_epoch = saved["best_validation_error_m"], saved["early_stopping_wait"], saved["epoch"] + 1
    with torch.no_grad():
        initial = model(train_x[:2].float())
        assert initial.shape == (2, 9) and torch.isfinite(initial).all()
    epoch = start_epoch - 1
    for epoch in range(start_epoch, config["epochs"] + 1):
        model.train()
        order = torch.randperm(len(train_x), device="cuda")
        loss_sum = 0.0
        for indices in order.split(config["batch_size"]):
            optimizer.zero_grad(set_to_none=True)
            pred = model(train_x[indices].float())
            loss = F.mse_loss(pred, (train_y[indices] - mean) / scale)
            assert torch.isfinite(loss)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            loss_sum += float(loss.detach()) * len(indices)
        model.eval()
        with torch.no_grad():
            pred = torch.cat([model(batch.float()) * scale + mean for batch in val_x.split(config["batch_size"])])
            score = float(torch.linalg.vector_norm((pred - val_y).reshape(-1, 3), dim=-1).mean())
        if score < best:
            best, stale = score, 0
            save(out / f"{name}_best.pt", epoch, best)
        else:
            stale += 1
        if epoch % 25 == 0:
            save(out / f"{name}_epoch_{epoch:03d}.pt", epoch, best)
        result = dict(
            model=name,
            epoch=epoch,
            train_normalized_mse=loss_sum / len(train_x),
            validation_macro_error_cm=score * 100,
            best_validation_macro_error_cm=best * 100,
            early_stopping_wait=stale,
        )
        with (out / "fit_metrics.jsonl").open("a") as stream:
            stream.write(json.dumps(result) + "\n")
        write(out / "status.json", dict(status="running", stage="probe_fitting", **result))
        if epoch == 1 or epoch % 10 == 0:
            print("PROBE_FIT " + json.dumps(result), flush=True)
        if stale >= config["patience"]:
            break
    saved = torch.load(out / f"{name}_best.pt", map_location="cuda", weights_only=False)
    model.load_state_dict(saved["model"])
    model.eval()

    def predict(x):
        with torch.inference_mode():
            return (
                torch.cat([model(batch.float()) * scale + mean for batch in x.split(config["batch_size"])]).cpu().numpy()
            )

    x, y, sx = data["test"]
    truth = y.cpu().numpy()
    prediction = predict(sx if shuffled else x)
    metrics = metric(prediction, truth)
    output = dict(
        metrics=metrics,
        best_epoch=saved["epoch"],
        best_validation_macro_error_cm=best * 100,
        trainable_parameters=sum(p.numel() for p in model.parameters()),
        epochs_run=epoch,
    )
    arrays = dict(prediction=prediction, target=truth)
    if not shuffled:
        other = predict(sx)
        output["test_images_shuffled"] = metric(other, truth)
        arrays["test_images_shuffled_prediction"] = other
    np.savez_compressed(out / f"{name}_test_predictions.npz", **arrays)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--resume-probes", action="store_true")
    args = parser.parse_args()
    args.manifest = args.manifest.resolve()
    manifest = json.loads(args.manifest.read_text())
    cfg = manifest["visual_probe"]
    out = Path(cfg["output_dir"])
    out.mkdir(parents=True, exist_ok=True)
    status = dict(status="running", stage="simulator_labels", started_unix=time.time())
    write(out / "status.json", status)
    try:
        if args.preflight:
            import copy

            subset = copy.deepcopy(manifest)
            subset["visual_probe"]["splits"] = {"train": [cfg["splits"]["train"][0]], "validation": [], "test": []}
            extract(subset, out, status)
            ep = cfg["splits"]["train"][0]["training_episode"]
            shard = torch.load(out / "features" / f"episode_{ep:03d}.pt", weights_only=True)
            x, y = shard["features"].cuda().float(), shard["targets"].cuda().float()
            torch.manual_seed(42)
            torch.cuda.manual_seed(42)
            head = SpatialVisualProbe(cfg["spatial_grid"]).cuda()
            optimizer = torch.optim.AdamW(head.parameters(), lr=0.001)
            mean = y.mean(0)
            scale = y.std(0).clamp_min(0.01)
            losses = []
            for _ in range(3):
                optimizer.zero_grad(set_to_none=True)
                prediction = head(x[:4])
                loss = F.mse_loss(prediction, (y[:4] - mean) / scale)
                loss.backward()
                assert head.token_projection[1].weight.grad.norm() > 0 and torch.isfinite(loss)
                optimizer.step()
                losses.append(float(loss.detach()))
            head.eval()
            with torch.no_grad():
                difference = float((head(x[:4]) - head(x[-4:])).abs().mean())
            assert difference > 0
            write(
                out / "preflight.json",
                dict(
                    status="passed",
                    feature_shape=list(x.shape),
                    targets=list(y.shape),
                    losses=losses,
                    changed_visual_features_prediction_mae=difference,
                    input="Frozen image features only",
                    updates_discarded=True,
                ),
            )
            print("VISUAL_PROBE_PREFLIGHT_PASSED", flush=True)
            return
        subprocess.run(
            [str(SIM_PYTHON), str(ROOT / "tools/collect_omega512_probe_labels.py"), "--manifest", str(args.manifest)],
            check=True,
        )
        extract(manifest, out, status)
        data = {name: load_split(cfg, out, name) for name in ["train", "validation", "test"]}
        episode_sets = [{r["training_episode"] for r in cfg["splits"][name]} for name in data]
        assert all(not a.intersection(b) for i, a in enumerate(episode_sets) for b in episode_sets[i + 1 :])
        mean = data["train"][1].mean(0).cpu().numpy()
        truth = data["test"][1].cpu().numpy()
        results = dict(train_mean=metric(np.repeat(mean[None], len(truth), axis=0), truth))
        results["matched_visual_head"] = fit(cfg, out, data, "matched", resume=args.resume_probes)
        results["trajectory_shuffled_visual_head"] = fit(
            cfg, out, data, "shuffled", shuffled=True, resume=args.resume_probes
        )
        write(
            out / "probe_results.json",
            dict(
                status="completed",
                results=results,
                splits={name: [r["training_episode"] for r in rows] for name, rows in cfg["splits"].items()},
                target_coordinates=cfg["coordinate_frame"],
                input_contract=cfg["inputs"],
                selection="Best validation macro localization error; test labels only used for final reporting.",
                limits=(
                    "40 right-arm trajectories; this spatially pooled feature probe does not test every possible feature"
                    " readout or policy success."
                ),
            ),
        )
        write(
            out / "status.json", dict(status="completed", stage="completed", finished_unix=time.time(), results=results)
        )
        print("PROBE_COMPLETED " + json.dumps(results), flush=True)
    except Exception as error:
        write(out / "status.json", dict(status="failed", error=repr(error), finished_unix=time.time()))
        raise


if __name__ == "__main__":
    main()
