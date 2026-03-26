"""Script para crear un Real-time Endpoint en Azure ML.

Crea un Online Endpoint de Azure ML para inferencia en tiempo real (<1s),
como alternativa al Batch Endpoint (2-3 min) para cargas pequeñas.

Uso:
    uv run python deployment/azure/create_realtime_endpoint.py

Variables de entorno requeridas (ver .env.example):
    AZURE_SUBSCRIPTION_ID
    AZURE_RESOURCE_GROUP
    AZURE_WORKSPACE_NAME
"""

from __future__ import annotations

import logging
import os

from azure.ai.ml import MLClient
from azure.ai.ml.entities import (
    ManagedOnlineDeployment,
    ManagedOnlineEndpoint,
    Model,
    CodeConfiguration,
    Environment,
)
from azure.identity import DefaultAzureCredential
from dotenv import load_dotenv

load_dotenv()

LOG = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

# ── Configuración ─────────────────────────────────────────────────────────────

SUBSCRIPTION_ID = os.environ["AZURE_SUBSCRIPTION_ID"]
RESOURCE_GROUP = os.environ.get("AZURE_RESOURCE_GROUP", "pcb-ml-rg")
WORKSPACE_NAME = os.environ.get("AZURE_WORKSPACE_NAME", "pcb-ml-workspace")

ENDPOINT_NAME = os.environ.get("PCB_REALTIME_ENDPOINT_NAME", "pcb-realtime-inference")
DEPLOYMENT_NAME = os.environ.get("PCB_REALTIME_DEPLOYMENT_NAME", "pcb-yolov8n-realtime")
MODEL_NAME = os.environ.get("PCB_MODEL_NAME", "pcb-yolov8n")
INSTANCE_TYPE = os.environ.get("PCB_RT_INSTANCE_TYPE", "Standard_DS3_v2")
INSTANCE_COUNT = int(os.environ.get("PCB_RT_INSTANCE_COUNT", "1"))


def main() -> None:
    """Crea el Online Endpoint y un deployment con el modelo registrado."""

    LOG.info("Conectando al workspace %s...", WORKSPACE_NAME)
    ml_client = MLClient(
        credential=DefaultAzureCredential(),
        subscription_id=SUBSCRIPTION_ID,
        resource_group_name=RESOURCE_GROUP,
        workspace_name=WORKSPACE_NAME,
    )

    # ── 1. Crear (o actualizar) el endpoint ──────────────────────────────────

    LOG.info("Creando endpoint: %s", ENDPOINT_NAME)
    endpoint = ManagedOnlineEndpoint(
        name=ENDPOINT_NAME,
        description="Real-time PCB defect detection endpoint (YOLOv8n)",
        auth_mode="key",
        tags={"project": "pcb-inspection", "type": "realtime"},
    )
    ml_client.online_endpoints.begin_create_or_update(endpoint).result()
    LOG.info("Endpoint '%s' listo.", ENDPOINT_NAME)

    # ── 2. Obtener la última versión del modelo ───────────────────────────────

    latest_model: Model = ml_client.models.get(name=MODEL_NAME, label="latest")
    LOG.info("Modelo: %s v%s", latest_model.name, latest_model.version)

    # ── 3. Definir el environment de inferencia ───────────────────────────────

    env = Environment(
        name="pcb-inference-env",
        conda_file="deployment/azure/conda.yml",
        image="mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu20.04",
        description="Entorno de inferencia PCB con YOLOv8",
    )

    # ── 4. Crear el deployment ────────────────────────────────────────────────

    LOG.info("Creando deployment: %s", DEPLOYMENT_NAME)
    deployment = ManagedOnlineDeployment(
        name=DEPLOYMENT_NAME,
        endpoint_name=ENDPOINT_NAME,
        model=latest_model.id,
        environment=env,
        code_configuration=CodeConfiguration(
            code="deployment/azure/batch_inference",
            scoring_script="score.py",
        ),
        instance_type=INSTANCE_TYPE,
        instance_count=INSTANCE_COUNT,
        environment_variables={
            "PCB_CONF_THRESHOLD": "0.25",
            "PCB_IOU_THRESHOLD": "0.45",
            "PCB_INFERENCE_TIMEOUT": "30",
        },
        request_settings={
            "request_timeout_ms": 60000,
            "max_concurrent_requests_per_instance": 4,
        },
        liveness_probe={
            "initial_delay": 10,
            "period": 10,
            "timeout": 5,
            "success_threshold": 1,
            "failure_threshold": 30,
        },
    )

    ml_client.online_deployments.begin_create_or_update(deployment).result()
    LOG.info("Deployment '%s' listo.", DEPLOYMENT_NAME)

    # ── 5. Enrutar el 100% del tráfico al nuevo deployment ───────────────────

    endpoint.traffic = {DEPLOYMENT_NAME: 100}
    ml_client.online_endpoints.begin_create_or_update(endpoint).result()

    # ── 6. Mostrar info del endpoint ─────────────────────────────────────────

    endpoint_info = ml_client.online_endpoints.get(ENDPOINT_NAME)
    scoring_url: str = endpoint_info.scoring_uri or ""
    LOG.info("✅ Endpoint activo:")
    LOG.info("   Scoring URL : %s", scoring_url)
    LOG.info("   Auth mode   : %s", endpoint_info.auth_mode)

    print("\n=== Endpoint creado ===")
    print(f"  Nombre  : {ENDPOINT_NAME}")
    print(f"  URL     : {scoring_url}")
    print(f"  Modelo  : {latest_model.name} v{latest_model.version}")
    print(f"  Instancia: {INSTANCE_TYPE} x{INSTANCE_COUNT}")
    print()
    print("Para obtener la API Key:")
    print(f"  az ml online-endpoint get-credentials --name {ENDPOINT_NAME}")
    print()
    print("Nota: el Batch Endpoint sigue disponible como fallback para >10 imágenes.")


if __name__ == "__main__":
    main()
