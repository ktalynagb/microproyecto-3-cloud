# Multi-stage build para FastAPI Backend
FROM python:3.11-slim AS build

ENV PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /src

# Copiar requirements.txt
COPY deployment/api/requirements.txt .

# Instalar uv
RUN pip install uv

# Instalar PyTorch CPU primero
RUN uv pip install --system \
    torch==2.6.0 torchvision==0.21.0 \
    --index-url https://download.pytorch.org/whl/cpu

# Instalar resto de dependencias (INCLUYENDO uvicorn)
RUN uv pip install --system --no-cache-dir -r requirements.txt

# Copiar código fuente
COPY deployment/api/backend.py .
COPY deployment/api/azure_ml_client.py .
COPY deployment/api/blob_manager.py .
COPY deployment/api/models.py .

# Runtime
FROM python:3.11-slim AS runtime

LABEL maintainer="Flux Solutions Cali"
LABEL description="PCB Defect Inspection – Backend FastAPI (CPU)"

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    API_HOST=0.0.0.0 \
    API_PORT=8080 \
    LOG_LEVEL=INFO \
    PATH="/usr/local/bin:$PATH"

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates curl apt-transport-https lsb-release gnupg \
    && curl -sL https://aka.ms/InstallAzureCLIDeb | bash \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copiar TODOS los site-packages del build (incluyendo uvicorn, fastapi, etc)
COPY --from=build /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages

# Copiar TAMBIÉN los binarios instalados (uvicorn, etc)
COPY --from=build /usr/local/bin /usr/local/bin

# Copiar código fuente
COPY --from=build /src/*.py ./

# Usuario no-root con home directory para montar credenciales de Azure CLI
RUN useradd -m --shell /bin/bash appuser && chown -R appuser /app

ENV HOME=/home/appuser

USER appuser

EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:${API_PORT}/api/v1/health || exit 1

CMD ["uvicorn", "backend:app", "--host", "0.0.0.0", "--port", "8080"]