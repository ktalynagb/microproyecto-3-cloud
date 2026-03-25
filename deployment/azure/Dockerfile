FROM python:3.11-slim

# Instalamos dependencias de sistema para OpenCV y PDF
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copiamos los archivos de configuración de uv
COPY pyproject.toml uv.lock ./

# Instalamos uv y las dependencias del proyecto
RUN pip install uv && uv sync --frozen

# Copiamos el resto del código
COPY . .

# Exponemos el puerto de Streamlit
EXPOSE 8501

# Comando para arrancar la app
CMD ["uv", "run", "streamlit", "run", "app/main.py", "--server.port=8501", "--server.address=0.0.0.0"]