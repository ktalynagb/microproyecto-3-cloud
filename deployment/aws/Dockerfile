FROM python:3.11-slim

# Dependencias críticas para OpenCV y reportes en AWS Linux
RUN apt-get update && apt-get install -y \
    libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Instalamos uv globalmente en el contenedor
RUN pip install uv

# Copiamos solo lo necesario para instalar dependencias primero (cache)
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev

# Copiamos el resto del proyecto
COPY . .

# AWS App Runner usa el puerto 8080 por defecto usualmente, 
# pero mantendremos 8501 para Streamlit
EXPOSE 8501

CMD ["uv", "run", "streamlit", "run", "app/main.py", "--server.port=8501", "--server.address=0.0.0.0"]