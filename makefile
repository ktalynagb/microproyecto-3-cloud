.PHONY: help install clean api-server test test-preprocessing test-inference test-coverage gui app

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
	@echo "  make gui                - Ejecutar Streamlit GUI (app/streamlit_app.py)"
	@echo "  make app                - Alias de 'make gui'"
	@echo "  make api-server         - Iniciar servidor FastAPI de inferencia"
	@echo "  make clean              - Limpiar todo (__pycache__, .pytest_cache)"
	@echo "  make test               - Ejecutar todos los tests con pytest"
	@echo "  make test-preprocessing - Ejecutar solo tests del modulo de preprocesamiento"
	@echo "  make test-inference     - Ejecutar solo tests del motor de inferencia"
	@echo "  make test-coverage      - Ejecutar tests con reporte de cobertura HTML"


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
# API REST (FastAPI)
# ============================================

api-server:
	@echo "Starting FastAPI inference server..."
	uv run service/inference_server.py

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
	@echo "Clean completed."
else
	@$(RM_DIR) __pycache__ .pytest_cache 2>/dev/null || true
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
IMAGE_NAME := davids117/pcb-defect-inspector:latest

dcupbuild:
	docker-compose up --build

build-image:
	docker build -t $(IMAGE_NAME) -f /Dockerfile .

local-run:
	docker run --rm -p 8000:8000 --env-file .env $(IMAGE_NAME)

# Push a Docker Hub (o taggear para ACR)
push-image:
	docker push $(IMAGE_NAME)