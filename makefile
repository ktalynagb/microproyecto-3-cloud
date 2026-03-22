.PHONY: help install clean clean-grpc grpc proto-gen grpc-server test test-preprocessing test-inference test-coverage gui link_model healthcheck inference mlflow

# Detectar SO
UNAME := $(shell uname)

ifeq ($(OS),Windows_NT)
    # Windows
    RM_DIR := rmdir /s /q
    MKDIR := mkdir
else
    # Linux / Mac
    RM_DIR := rm -rf
    MKDIR := mkdir -p
endif

# ============================================
# TARGETS GENERALES
# ============================================

help:
	@echo "Comandos disponibles:"
	@echo "  make install            - Instalar dependencias (uv sync)"
	@echo "  make gui                - Ejecutar Streamlit GUI (app/app.py)"
	@echo "  make grpc               - Generar stubs gRPC desde proto/inference.proto"
	@echo "  make proto-gen          - Alias de 'make grpc' (genera stubs gRPC)"
	@echo "  make grpc-server        - Iniciar servidor gRPC de inferencia"
	@echo "  make clean-grpc         - Limpiar stubs gRPC generados"
	@echo "  make clean              - Limpiar todo (__pycache__, .pytest_cache, proto/generated)"
	@echo "  make test               - Ejecutar todos los tests con pytest"
	@echo "  make test-preprocessing - Ejecutar solo tests del modulo de preprocesamiento"
	@echo "  make test-inference     - Ejecutar solo tests del motor de inferencia"
	@echo "  make test-coverage      - Ejecutar tests con reporte de cobertura HTML"
	@echo "  make inference          - Ejecutar script de inferencia"
	@echo "  make mlflow             - Iniciar servidor de MLflow UI"
	@echo "  make link_model         - Establecer HF_MODEL_ID y ejecutar health check"
	@echo "  make healthcheck        - Ejecutar health check del modelo MLflow"
	@echo "  make app                - Ejecutar Streamlit GUI (app/streamlit_app.py)"


# ============================================
# INSTALACIÓN Y DEPENDENCIAS
# ============================================

install:
	@echo "Installing dependencies with uv..."
	uv sync

# ============================================
# GUI (Streamlit)
# ============================================

gui:
	uv run -m streamlit run app/streamlit_app.py

app: gui

# ============================================
# gRPC TARGETS (Multiplataforma)
# ============================================

grpc:
	@echo "Creating proto/generated directory if it doesn't exist..."
ifeq ($(OS),Windows_NT)
	@if not exist proto\generated mkdir proto\generated
else
	@mkdir -p proto/generated
endif
	@echo "Generating gRPC stubs from proto/inference.proto..."
	uv run -m grpc_tools.protoc -I proto --python_out=proto/generated --grpc_python_out=proto/generated proto/inference.proto
	@echo "gRPC stubs generated successfully in proto/generated/"

proto-gen: grpc

grpc-server:
	@echo "Starting gRPC inference server..."
	uv run service/inference_server.py

clean-grpc:
	@echo "Removing gRPC generated stubs..."
ifeq ($(OS),Windows_NT)
	@if exist proto\generated $(RM_DIR) proto\generated
	@echo "gRPC stubs removed."
else
	@$(RM_DIR) proto/generated
	@echo "gRPC stubs removed."
endif

# ============================================
# INFERENCE
# ============================================

link_model:
ifeq ($(OS),Windows_NT)
	@echo "Setting HF_MODEL_ID and running health check..."
	@cmd /c "set HF_MODEL_ID=Ateeqq/ai-vs-human-image-detector && uv run -m service.inference.mlflow_health_check"
else
	@echo "Setting HF_MODEL_ID and running health check..."
	HF_MODEL_ID=Ateeqq/ai-vs-human-image-detector uv run -m service.inference.mlflow_health_check
endif

inference:
	@echo "Running inference script..."
	uv run service/inference_server.py

mlflow:
	@echo "Running MLflow tracking server..."
	uv run mlflow ui --host 127.0.0.1 --port 5000 --backend-store-uri sqlite:///mlflow.db

healthcheck:
	@echo "Running health check..."
	uv run -m service.inference.mlflow_health_check

# ============================================
# LIMPIEZA
# ============================================

clean:
	@echo "Cleaning Python cache files..."
ifeq ($(OS),Windows_NT)
	@if exist __pycache__ $(RM_DIR) __pycache__
	@if exist service\__pycache__ $(RM_DIR) service\__pycache__
	@if exist app\__pycache__ $(RM_DIR) app\__pycache__
	@if exist tests\__pycache__ $(RM_DIR) tests\__pycache__
	@if exist .pytest_cache $(RM_DIR) .pytest_cache
	@if exist proto\generated $(RM_DIR) proto\generated
	@echo "Clean completed."
else
	@$(RM_DIR) __pycache__ .pytest_cache proto/generated 2>/dev/null || true
	@echo "Clean completed."
endif

# ============================================
# TESTS
# ============================================

test:
	@echo "Running all tests with pytest..."
	uv run -m pytest tests/ -v

test-preprocessing:
	@echo "Running preprocessing module tests..."
	uv run -m pytest tests/test_preprocessing.py -v

test-inference:
	@echo "Running inference engine module tests..."
	uv run -m pytest tests/test_inference_engine.py -v

test-coverage:
	@echo "Running tests with coverage..."
	uv run -m pytest tests/ --cov --cov-report=html

# ============================================
# DOCKER
# ============================================

# Targets para build y push de Docker image
IMAGE_NAME := davids117/image-classifier:latest

dcupbuild:
	docker-compose up --build

build-image:
	docker build -t $(IMAGE_NAME) -f /Dockerfile .

local-run:
	docker run --rm -p 50051:50051 --env-file .env $(IMAGE_NAME)

# Push a Docker Hub (o taggear para ACR)
push-image:
	docker push $(IMAGE_NAME)