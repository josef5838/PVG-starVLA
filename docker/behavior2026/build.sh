#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
vggt_context="${VGGT_OMEGA_CONTEXT:-${repo_root}/../vggt-omega-action-pred}"
image_tag="${IMAGE_TAG:-starvla-behavior:dev}"

if ! command -v docker >/dev/null 2>&1; then
  echo "docker is not installed or not on PATH" >&2
  exit 2
fi
if ! docker buildx version >/dev/null 2>&1; then
  echo "docker buildx is required for the named VGGT-Omega build context" >&2
  exit 2
fi
if [[ ! -f "${vggt_context}/pyproject.toml" ]]; then
  echo "VGGT_OMEGA_CONTEXT is not a VGGT-Omega checkout: ${vggt_context}" >&2
  exit 2
fi

docker buildx build --load \
  --build-context "vggt_omega=${vggt_context}" \
  --file "${repo_root}/docker/behavior2026/Dockerfile" \
  --tag "${image_tag}" \
  "$@" \
  "${repo_root}"
