# ── Stage 1: builder ─────────────────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /build

# Instalar dependencias en un venv aislado
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt

# ── Stage 2: runtime ──────────────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

LABEL maintainer="Flux Solutions Cali"
LABEL description="PCB Defect Inspection – Backend FastAPI"

# Copiar venv compilado
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

WORKDIR /app

# Copiar código fuente del backend
COPY backend.py azure_ml_client.py blob_manager.py models.py ./

# Variables de entorno con valores por defecto (no secrets)
ENV API_HOST=0.0.0.0
ENV API_PORT=8080
ENV LOG_LEVEL=INFO
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

EXPOSE 8080

# Usuario no-root por seguridad
RUN useradd --no-create-home --shell /bin/false appuser \
 && chown -R appuser /app
USER appuser

HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:${API_PORT}/api/v1/health')"

CMD ["sh", "-c", "uvicorn backend:app --host ${API_HOST} --port ${API_PORT} --log-level $(echo ${LOG_LEVEL} | tr A-Z a-z)"]
