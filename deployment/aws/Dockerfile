# Multi-stage build para AWS App Runner
FROM python:3.11-slim AS build

ENV PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    VENV_PATH=/opt/venv

COPY --from=ghcr.io/astral-sh/uv:0.10.9 /uv /uvx /bin/

WORKDIR /src

# Dependencias del sistema necesarias para build
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential gcc libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml uv.lock ./

# PyTorch CPU
RUN uv pip install --system \
    "torch @ https://download.pytorch.org/whl/cpu/torch-2.6.0%2Bcpu-cp311-cp311-manylinux_2_28_x86_64.whl" \
    "torchvision @ https://download.pytorch.org/whl/cpu/torchvision-0.21.0%2Bcpu-cp311-cp311-manylinux_2_28_x86_64.whl"

RUN uv sync --locked --no-dev

COPY . .

# Runtime
FROM python:3.11-slim AS runtime

ENV PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:$PATH"

# Solo dependencias de runtime (OpenCV)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY --from=build /opt/venv /opt/venv
COPY --from=build /src/app ./app/
COPY --from=build /src/pyproject.toml /src/uv.lock ./

EXPOSE 8501

CMD ["uv", "run", "streamlit", "run", "app/main.py", "--server.port=8501", "--server.address=0.0.0.0"]