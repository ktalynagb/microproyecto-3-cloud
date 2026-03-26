# Multi-stage build para FastAPI Backend
FROM python:3.11-slim AS build

ENV PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    VENV_PATH=/opt/venv

COPY --from=ghcr.io/astral-sh/uv:0.10.9 /uv /uvx /bin/

WORKDIR /src

COPY pyproject.toml uv.lock ./

# Instalar PyTorch CPU
RUN uv pip install --system \
    "torch @ https://download.pytorch.org/whl/cpu/torch-2.6.0%2Bcpu-cp311-cp311-manylinux_2_28_x86_64.whl" \
    "torchvision @ https://download.pytorch.org/whl/cpu/torchvision-0.21.0%2Bcpu-cp311-cp311-manylinux_2_28_x86_64.whl"

RUN uv sync --locked --no-dev

COPY deployment/api/*.py ./

# Runtime
FROM python:3.11-slim AS runtime

LABEL maintainer="Flux Solutions Cali"
LABEL description="PCB Defect Inspection – Backend FastAPI (CPU)"

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    API_HOST=0.0.0.0 \
    API_PORT=8080 \
    LOG_LEVEL=INFO \
    PATH="/opt/venv/bin:$PATH"

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY --from=build /opt/venv /opt/venv
COPY --from=build /src/*.py ./

# Usuario no-root
RUN useradd --no-create-home --shell /bin/false appuser && chown -R appuser /app
USER appuser

EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:${API_PORT}/api/v1/health || exit 1

CMD ["sh", "-c", "uvicorn backend:app --host ${API_HOST} --port ${API_PORT} --log-level $(echo ${LOG_LEVEL} | tr A-Z a-z)"]
