"""Benchmark disposable joint24 microbatches, then write a fresh production config."""

import argparse
import copy
import json
import sys
import time
from pathlib import Path

import torch
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from starVLA.dataloader.lerobot_datasets import get_vla_dataset  # noqa: E402
from starVLA.model.framework.VLM4A.OmegaCameraStreamDiT import OmegaCameraStreamDiT  # noqa: E402
from starVLA.training.trainer_utils.trainer_tools import build_param_lr_groups  # noqa: E402


def apply_batch_schedule(cfg, selected):
    """Change batch/update counts while preserving the reference observation budget."""
    original_effective_batch = int(cfg.datasets.vla_data.per_device_batch_size) * int(cfg.trainer.gradient_accumulation_steps)
    original_updates = int(cfg.trainer.max_train_steps)
    effective_batch = int(selected["microbatch"]) * int(selected["accumulation"])
    assert effective_batch == selected["effective_batch"]
    budget = original_updates * original_effective_batch
    assert budget % effective_batch == 0
    cfg.datasets.vla_data.per_device_batch_size = selected["microbatch"]
    cfg.trainer.gradient_accumulation_steps = selected["accumulation"]
    cfg.trainer.validation_batch_size = 4
    cfg.trainer.sample_budget = budget
    cfg.trainer.planned_training_samples = budget
    cfg.trainer.max_train_steps = budget // effective_batch
    cfg.trainer.step_limit = cfg.trainer.max_train_steps
    for key in ("num_warmup_steps", "save_interval", "eval_interval"):
        samples_per_event = int(cfg.trainer[key]) * original_effective_batch
        assert samples_per_event % effective_batch == 0
        cfg.trainer[key] = samples_per_event // effective_batch
    cfg.trainer.milestone_updates = [int(step)*original_effective_batch//effective_batch
                                    for step in cfg.trainer.milestone_updates]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    cfg = OmegaConf.load(args.config)
    torch.set_num_threads(4)
    torch.manual_seed(int(cfg.seed))
    torch.cuda.manual_seed(int(cfg.seed))
    data_cfg = copy.deepcopy(cfg.datasets.vla_data)
    data_cfg.statistics_cache_dir = str(args.report.parent / "batch_tuning_split")
    dataset = get_vla_dataset(data_cfg, mode="train")
    samples = [dataset[i] for i in range(16)]
    model = OmegaCameraStreamDiT(cfg).cuda().train()
    optimizer = torch.optim.AdamW(
        build_param_lr_groups(model, cfg), lr=cfg.trainer.learning_rate.base,
        betas=tuple(cfg.trainer.optimizer.betas), eps=cfg.trainer.optimizer.eps,
        weight_decay=cfg.trainer.optimizer.weight_decay, fused=True,
    )
    total_memory = torch.cuda.get_device_properties(0).total_memory
    records = []
    loss = None
    for size, accumulation in ((4, 2), (8, 1), (16, 1)):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        elapsed = []
        try:
            # First try increasing microbatch without changing effective batch.
            for update in range(3):
                torch.cuda.synchronize()
                start = time.perf_counter()
                optimizer.zero_grad(set_to_none=True)
                for _ in range(accumulation):
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        loss = model(samples[:size])["action_loss"] / accumulation
                    if not torch.isfinite(loss):
                        raise ValueError("Nonfinite batch-benchmark loss")
                    loss.backward()
                    loss = None
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                torch.cuda.synchronize()
                if update:
                    elapsed.append(time.perf_counter() - start)
            peak = torch.cuda.max_memory_allocated()
            record = dict(microbatch=size, accumulation=accumulation, effective_batch=accumulation*size, status="passed",
                          seconds_per_update=sum(elapsed)/len(elapsed),
                          observations_per_second=accumulation*size*len(elapsed)/sum(elapsed),
                          peak_allocated_gpu_gib=peak/2**30, memory_fraction=peak/total_memory)
            records.append(record)
        except torch.cuda.OutOfMemoryError:
            records.append(dict(microbatch=size, status="out_of_memory"))
            loss = None
            optimizer.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
            break
        print(json.dumps(records[-1]), flush=True)
        args.report.write_text(json.dumps(dict(status="benchmarking", gpu=torch.cuda.get_device_name(), records=records), indent=2)+"\n")
    eligible = [r for r in records if r["status"] == "passed" and r["memory_fraction"] <= 0.85]
    if not eligible or eligible[0]["microbatch"] != 4:
        raise RuntimeError("Production baseline batch did not pass with sufficient memory headroom")
    selected = eligible[0]
    for record in eligible[1:]:
        if record["observations_per_second"] > selected["observations_per_second"] * 1.05:
            selected = record
    # Keep the full 160K observation budget and all schedule events aligned by samples.
    apply_batch_schedule(cfg, selected)
    cfg.config_yaml = str(args.out)
    OmegaConf.save(cfg, args.out, resolve=True)
    report = dict(status="completed", gpu=torch.cuda.get_device_name(), torch_version=torch.__version__,
                  records=records, selected=selected, target_updates=int(cfg.trainer.max_train_steps),
                  total_training_observations=int(cfg.trainer.sample_budget), validation_batch_size=4,
                  learning_rate=float(cfg.trainer.learning_rate.base),
                  schedules_aligned_by_observations=True,
                  note="Disposable benchmark weights are discarded; production initializes fresh from seed 42.")
    args.report.write_text(json.dumps(report, indent=2)+"\n")


if __name__ == "__main__":
    main()
