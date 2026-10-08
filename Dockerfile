# syntax=docker/dockerfile:1
FROM node:24-bookworm-slim AS frontend
WORKDIR /web
COPY frontend/package.json frontend/package-lock.json ./frontend/
RUN cd frontend && npm ci
COPY frontend ./frontend
COPY docs/results ./docs/results
RUN cd frontend && npm run build

FROM ghcr.io/astral-sh/uv:0.12.19 AS uv
FROM python:3.12-slim AS builder
COPY --from=uv /uv /uvx /usr/local/bin/
ENV UV_LINK_MODE=copy UV_COMPILE_BYTECODE=1 UV_PROJECT_ENVIRONMENT=/opt/venv
WORKDIR /build
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --extra serve --no-install-project
COPY src ./src
COPY README.md LICENSE ./
RUN uv sync --frozen --no-dev --extra serve --no-editable

FROM python:3.12-slim AS runtime
RUN apt-get update \
    && apt-get install --no-install-recommends -y libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd --gid 10001 fleetguard \
    && useradd --uid 10001 --gid 10001 --no-create-home --shell /usr/sbin/nologin fleetguard
COPY --from=builder --chown=10001:10001 /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    FLEETGUARD_RELEASE_DIR=/models/release \
    FLEETGUARD_ALLOW_SYNTHETIC=false \
    MLFLOW_DISABLE_TELEMETRY=true \
    MLFLOW_DISABLE_AGENT_HINT=true \
    OMP_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    MPLCONFIGDIR=/tmp/matplotlib
WORKDIR /app
USER 10001:10001
EXPOSE 8000
HEALTHCHECK --interval=10s --timeout=3s --start-period=10s --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health/ready', timeout=2).read()"]
CMD ["python", "-m", "fleetguard", "serve", "--host", "0.0.0.0", "--port", "8000"]

FROM runtime AS web
COPY --from=frontend --chown=10001:10001 /web/frontend/dist /app/web
ENV FLEETGUARD_WEB_DIR=/app/web

# Named context supplied only for cloud builds, outside the source checkout context.
FROM web AS cloud
USER root
RUN chown 10001:10001 /tmp && chmod 1777 /tmp
# Fargate copies these directory permissions to its task-scoped bind volume.
VOLUME ["/tmp"]
USER 10001:10001
COPY --from=modelbundle --chown=10001:10001 / /models/release/
ARG GIT_SHA
ARG BUNDLE_SHA256
ENV FLEETGUARD_GIT_SHA=$GIT_SHA FLEETGUARD_BUNDLE_SHA256=$BUNDLE_SHA256
LABEL org.opencontainers.image.revision=$GIT_SHA fleetguard.model-bundle-sha256=$BUNDLE_SHA256
RUN python -c "import os,re; assert re.fullmatch('[0-9a-f]{40}',os.environ['FLEETGUARD_GIT_SHA']); assert re.fullmatch('[0-9a-f]{64}',os.environ['FLEETGUARD_BUNDLE_SHA256'])"

# Default local/CI image still accepts the read-only model mount from update 05.
FROM web AS final
