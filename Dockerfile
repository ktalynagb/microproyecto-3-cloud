# Multi-stage build para Streamlit Frontend
FROM python:3.11-slim AS build

ENV PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    VENV_PATH=/opt/venv

# Copiar uv desde imagen oficial
COPY --from=ghcr.io/astral-sh/uv:0.10.9 /uv /uvx /bin/

WORKDIR /src

# Cachear instalación de dependencias (antes de copiar el código)
COPY pyproject.toml uv.lock ./

# Instalar PyTorch CPU primero
RUN uv pip install --system \
    "torch @ https://download.pytorch.org/whl/cpu/torch-2.6.0%2Bcpu-cp311-cp311-manylinux_2_28_x86_64.whl" \
    "torchvision @ https://download.pytorch.org/whl/cpu/torchvision-0.21.0%2Bcpu-cp311-cp311-manylinux_2_28_x86_64.whl"

# Sincronizar resto de dependencias
RUN uv sync --locked --no-dev

# Copiar código
COPY app/ ./app/

# Runtime
FROM python:3.11-slim AS runtime

ENV PYTHONUNBUFFERED=1 \
    API_HOST=localhost \
    API_PORT=8000 \
    API_TIMEOUT=30 \
    PATH="/opt/venv/bin:$PATH"

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY --from=build /opt/venv /opt/venv
COPY --from=build /src/app ./app/

EXPOSE 8501

CMD ["streamlit", "run", "app/streamlit_app.py", "--server.address", "0.0.0.0"]