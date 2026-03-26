FROM python:3.11-slim

# Versión: 2.0 (2026-03-26)
# Cambios:
#   - Agregado azureml-core y azureml-sdk para batch endpoint driver
#   - Agregado roboflow para ingesta de datos
#   - Sincronizado con todos los imports de componentes Python
# Mantener en sincronía con: components/*.py, batch_inference/*.py, conda.yml

RUN apt-get update && apt-get install -y \
    libgl1 libglib2.0-0 libxcb1 libx11-6 libsm6 libxext6 libxrender1 libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Instalar todas las librerías necesarias directamente
RUN pip install --no-cache-dir \
    azureml-core>=1.53.0 \
    azureml-sdk>=1.53.0 \
    azureml-mlflow \
    roboflow>=1.1.0 \
    datasets>=2.14.0 \
    huggingface-hub>=1.7.2 \
    ultralytics>=8.3.0 \
    torch>=2.0.0 \
    torchvision>=0.15.0 \
    opencv-python-headless>=4.10.0 \
    Pillow>=11.2.1 \
    numpy>=1.24.0 \
    pandas>=2.2.3 \
    reportlab>=4.2.0 \
    mlflow>=2.0.0 \
    requests>=2.31.0 \
    azure-ai-ml>=1.32.0 \
    azure-identity>=1.25.3 \
    python-dotenv>=1.0.0

COPY components/ ./components/

RUN python -c "import cv2; import ultralytics; print('✓ Dependencies OK')"

CMD ["/bin/bash"]