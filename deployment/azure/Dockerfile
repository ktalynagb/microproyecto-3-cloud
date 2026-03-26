FROM python:3.11-slim

# Instalamos dependencias de sistema para OpenCV, X11 y rendering
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libxcb1 \
    libx11-6 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copiamos conda.yml para instalar dependencias
COPY conda.yml ./

# Instalamos conda/mamba o usamos pip directamente
RUN apt-get update && apt-get install -y wget && \
    wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh && \
    bash /tmp/miniconda.sh -b -p /opt/conda && \
    rm /tmp/miniconda.sh && \
    /opt/conda/bin/conda env create -f conda.yml && \
    /opt/conda/bin/conda clean -afy

# Activar el entorno
ENV PATH=/opt/conda/envs/pcb-azure-env-v2/bin:$PATH

# Copiamos los scripts del pipeline
COPY components/ ./components/

# Verificar que las dependencias críticas están instaladas
RUN python -c "import datasets; import cv2; import ultralytics; print('✓ Dependencies OK')"

CMD ["/bin/bash"]