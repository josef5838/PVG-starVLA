# BEHAVIOR-1K Docker Development Runtime

This image serves StarVLA checkpoints for BEHAVIOR-1K evaluation. Training,
datasets, and OmniGibson remain outside the container. Model bundles are mounted
read-only, so a newly trained checkpoint does not require rebuilding the image.

## Prerequisites

- Docker with BuildKit / buildx
- NVIDIA Container Toolkit
- A local checkout of `vggt-omega-action-pred`
- A StarVLA run containing `config.yaml`, `dataset_statistics.json`, and a
  `.pt` or `.safetensors` checkpoint two directory levels below the run root

## Prepare a model bundle

Run this in the host StarVLA environment:

```bash
python docker/behavior2026/prepare_model_bundle.py \
  --checkpoint results/Checkpoints/RUN_ID/final_model/pytorch_model.pt \
  --output-root /data/starvla-models/RUN_ID

cd /data/starvla-models/RUN_ID
sha256sum -c SHA256SUMS
```

The default hard-link mode avoids duplicating large checkpoint and CLIP files
when the bundle is on the same filesystem. Use `--file-mode copy` before moving
the bundle to another machine or archival store.

## Build the runtime image

```bash
IMAGE_TAG=starvla-behavior:dev \
VGGT_OMEGA_CONTEXT=../vggt-omega-action-pred \
bash docker/behavior2026/build.sh
```

The checkpoint is not part of this image. Rebuild only when StarVLA,
VGGT-Omega, the Dockerfile, or runtime dependencies change.

## Run and test

```bash
bash docker/behavior2026/run_dev.sh /data/starvla-models/RUN_ID

# In another terminal:
python docker/behavior2026/smoke_test.py --host 127.0.0.1 --port 8000
```

The service binds `0.0.0.0:8000` inside the container. The development runner
publishes it only through the selected host port. `POLICY_DTYPE=auto` selects
BF16 on supported GPUs and FP16 on older CUDA GPUs such as Titan RTX.

Useful overrides:

```bash
POLICY_PORT=8010 POLICY_DTYPE=fp16 EXECUTE_HORIZON=8 \
  bash docker/behavior2026/run_dev.sh /data/starvla-models/RUN_ID
```

## Development contract

1. Train outside Docker and create a new model bundle.
2. Run the container with the new bundle without rebuilding it.
3. Require the health check and WebSocket smoke test to pass.
4. Rebuild after source or dependency changes.
5. Before submission, build an immutable image that copies the selected bundle
   into the same `/models` layout and rerun the identical tests on a 24 GB GPU.

