# BEHAVIOR-1K benchmark harness

This directory is the control plane for comparing `vggt_surflo`, `pi0.5`, and
Cosmos Policy on BEHAVIOR-1K.  It intentionally does not vendor baseline
implementations or combine their Python environments.

```text
StarVLA checkout                 External, pinned checkouts
----------------                 -------------------------
benchmark protocol and results   OpenPI (pi0.5)
dataset validation               Cosmos Policy
BEHAVIOR evaluation launcher     BEHAVIOR-1K / OmniGibson
vggt_surflo train and server
```

The normal lifecycle is:

1. Prepare and validate a LeRobot v3 dataset on RTX; keep the raw HDF5 source
   immutable.
2. Copy the prepared dataset, its `transfer.sha256`, and this repository's
   commit to a training machine (or train on an idle RTX machine).
3. Train one policy, producing a complete method bundle with its normalisation
   statistics and resolved configuration.
4. Move that bundle to the RTX machine if it was trained elsewhere.
5. Start the policy locally on RTX and invoke the common BEHAVIOR evaluator.
6. Collect per-rollout JSON files into a single CSV summary.

The scripts print their command lines by default.  They perform an external
action only with `--execute`.

## Prerequisites

- BEHAVIOR-1K `v3.9.2` on the RTX evaluation machine.
- A prepared LeRobot v3 data root containing `data/`, `meta/`, and `videos/`.
- Separate environments/checkouts for StarVLA, OpenPI, Cosmos Policy, and
  BEHAVIOR. Do not install their dependency sets into one environment.

The benchmark protocol is in `protocol.json`; it is the immutable source of
truth for a reported result.  Copy it into every run directory before training.

## First smoke benchmark: turning_on_radio

On either an H100 machine or an otherwise idle RTX machine:

```bash
export B1K_DATASET=/absolute/path/behavior-1k-2026
python benchmarks/behavior1k/scripts/validate_dataset.py \
  --dataset "$B1K_DATASET" --task turning_on_radio

# Preview only.  Add --execute when GPUs are available.
bash benchmarks/behavior1k/scripts/train_method.sh \
  --method vggt_surflo --dataset "$B1K_DATASET" \
  --run-id behavior_radio_smoke
```

For pi0.5, use the OpenPI `behavior` branch that contains the B1K adapter.  It
must first define the portable `pi05_b1k_radio` configuration documented in
`methods/pi05.json`; the current branch's `pi05_b1k` configuration points to a
different machine.  After that, preview/run:

```bash
export OPENPI_ROOT=/absolute/path/openpi
export OPENPI_CONFIG=pi05_b1k_radio
bash benchmarks/behavior1k/scripts/train_method.sh \
  --method pi05 --dataset "$B1K_DATASET" \
  --run-id pi05_radio_smoke
```

The Cosmos entry is deliberately marked `planned`: its official repository has
no BEHAVIOR/R1Pro adapter in this checkout.  Do not claim a Cosmos comparison
until `adapters/cosmos_policy.md` is completed and its conformance test passes.

## RTX evaluation

Start the selected policy server locally on the RTX machine.  The helper has
method-specific commands and is also dry-run by default:

```bash
export B1K_TASKS_JSONL=/absolute/path/behavior-1k-2026/meta/tasks.jsonl
export CKPT_PATH=/absolute/path/to/vggt/checkpoint.pt
bash benchmarks/behavior1k/scripts/start_policy_server.sh \
  --method vggt_surflo --task turning_on_radio
```

Then execute the official evaluator, pointing it to `127.0.0.1`:

```bash
export BEHAVIOR_ROOT=/absolute/path/BEHAVIOR-1K
export BEHAVIOR_PYTHON=/absolute/path/behavior/bin/python
bash benchmarks/behavior1k/scripts/eval_method.sh \
  --method vggt_surflo --task turning_on_radio \
  --mode train --instance-indices 0 --execute
```

Use `--mode public_test` for final benchmark results.  Keep development and
public-test result directories distinct.

## Results and provenance

Each method/run directory must contain:

- `protocol.json` copied from this directory unchanged;
- `provenance.json` recording source commits, training seed, data hash, and
  runtime device;
- policy checkpoint bundle and normalisation assets;
- raw evaluator rollout JSONs and optional videos.

Aggregate result files after evaluation:

```bash
python benchmarks/behavior1k/scripts/collect_results.py \
  --input-root /absolute/path/b1k-eval/pi05_radio \
  --method pi05 --output summary.csv
```

`summary.csv` retains each rollout.  The terminal output reports aggregate
`q_score`, elapsed time, and any discovered distance metric without discarding
the raw evidence.
