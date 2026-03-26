FROM python:3.11-slim

# Instalamos dependencias de sistema para OpenCV
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libxcb1 \
    libx11-6 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Instalamos uv
RUN pip install uv

# Copiamos pyproject.toml y uv.lock (desde la raíz del proyecto)
COPY pyproject.toml ./

# Instalamos dependencias con uv
RUN uv sync --no-venv

# Copiamos los scripts
COPY components/ ./components/

# Verificar dependencias críticas
RUN python -c "import datasets; import cv2; import ultralytics; print('✓ Dependencies OK')"

CMD ["/bin/bash"]