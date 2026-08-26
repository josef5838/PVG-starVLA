#!/usr/bin/env bash
set -euo pipefail

model_bundle="${1:?Usage: run_dev.sh /absolute/path/to/model-bundle}"
image_tag="${IMAGE_TAG:-starvla-behavior:dev}"
host_port="${POLICY_PORT:-8000}"

model_bundle="$(realpath "${model_bundle}")"
shopt -s nullglob
checkpoints=("${model_bundle}"/run/final_model/*.pt "${model_bundle}"/run/final_model/*.safetensors)
shopt -u nullglob
if [[ ${#checkpoints[@]} -ne 1 ]]; then
  echo "Invalid model bundle: expected one checkpoint under run/final_model" >&2
  exit 2
fi

exec docker run --rm --gpus all \
  --name starvla-behavior-dev \
  --shm-size 4g \
  --publish "${host_port}:8000" \
  --volume "${model_bundle}:/models:ro" \
  --env POLICY_PORT=8000 \
  --env POLICY_DTYPE="${POLICY_DTYPE:-auto}" \
  --env EXECUTE_HORIZON="${EXECUTE_HORIZON:-8}" \
  "${image_tag}"
