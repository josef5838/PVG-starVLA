# syntax=docker/dockerfile:1.7

ARG BASE_IMAGE=pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime
FROM ${BASE_IMAGE}

ENV DEBIAN_FRONTEND=noninteractive \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    HF_HUB_DISABLE_TELEMETRY=1 \
    TOKENIZERS_PARALLELISM=false \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    STARVLA_FRAMEWORK_MODULES=starVLA.model.framework.VLM4A.VGGTOmegaOFT

RUN apt-get update \
    && apt-get install -y --no-install-recommends curl tini \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/starvla

COPY docker/behavior2026/requirements-runtime.txt /tmp/requirements-runtime.txt
RUN python -m pip install --no-cache-dir -r /tmp/requirements-runtime.txt \
    && python -c "import torch, torchvision; assert torch.__version__.startswith('2.6.0'); assert torchvision.__version__.startswith('0.21.0')"

COPY --from=vggt_omega / /opt/vggt_omega
RUN python -m pip install --no-cache-dir --no-deps /opt/vggt_omega

COPY pyproject.toml LICENSE README.md ./
COPY starVLA ./starVLA
COPY deployment ./deployment
COPY examples/Behavior/serve_vggt_omega_2026.py ./examples/Behavior/serve_vggt_omega_2026.py
RUN python -m pip install --no-cache-dir --no-deps .

COPY docker/behavior2026/entrypoint.sh /usr/local/bin/starvla-behavior-server
COPY docker/behavior2026/smoke_test.py /opt/starvla/docker/behavior2026/smoke_test.py

RUN useradd --create-home --uid 10001 starvla \
    && chmod 0755 /usr/local/bin/starvla-behavior-server \
    && chown -R starvla:starvla /opt/starvla

USER starvla

EXPOSE 8000
HEALTHCHECK --interval=10s --timeout=3s --start-period=90s --retries=6 \
    CMD curl --fail --silent http://127.0.0.1:8000/healthz || exit 1

ENTRYPOINT ["/usr/bin/tini", "--", "/usr/local/bin/starvla-behavior-server"]
