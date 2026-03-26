FROM python:3.11-slim

RUN apt-get update && apt-get install -y \
    libgl1 libglib2.0-0 libxcb1 libx11-6 libsm6 libxext6 libxrender1 libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Instalar todas las librerías necesarias directamente
RUN pip install \
    datasets>=2.14.0 \
    huggingface-hub>=1.7.2 \
    ultralytics>=8.3.0 \
    opencv-python-headless>=4.10.0 \
    Pillow>=11.2.1 \
    pandas>=2.2.3 \
    reportlab>=4.2.0 \
    azure-ai-ml>=1.32.0 \
    azure-identity>=1.25.3 \
    mlflow \
    azureml-mlflow \
    python-dotenv>=1.0.0

COPY components/ ./components/

RUN python -c "import datasets; import cv2; import ultralytics; print('✓ Dependencies OK')"

CMD ["/bin/bash"]