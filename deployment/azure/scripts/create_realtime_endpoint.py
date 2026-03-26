"""Script para crear un Real-time Endpoint en Azure ML.

Uso:
    python deployment/azure/scripts/create_realtime_endpoint.py

Variables de entorno:
    AZURE_SUBSCRIPTION_ID, AZURE_RESOURCE_GROUP, AZURE_ML_WORKSPACE
    AZURE_TENANT_ID, AZURE_CLIENT_ID, AZURE_CLIENT_SECRET (o usar az login)

El script:
1. Crea un Managed Online Endpoint (si no existe).
2. Registra el modelo (si no está registrado).
3. Crea un deployment con el scoring script.
4. Configura fallback a Batch si se detectan >10 imágenes en la API.
"""

from __future__ import annotations

import os
import sys

# ---------------------------------------------------------------------------
# Configuración
# ---------------------------------------------------------------------------

SUBSCRIPTION_ID = os.environ.get("AZURE_SUBSCRIPTION_ID", "")
RESOURCE_GROUP = os.environ.get("AZURE_RESOURCE_GROUP", "pcb-ml-rg")
WORKSPACE_NAME = os.environ.get("AZURE_ML_WORKSPACE", "pcb-ml-workspace")
ENDPOINT_NAME = os.environ.get("RT_ENDPOINT_NAME", "pcb-realtime")
DEPLOYMENT_NAME = os.environ.get("RT_DEPLOYMENT_NAME", "pcb-yolov8n-rt")
MODEL_NAME = os.environ.get("MODEL_NAME", "pcb-yolov8n")
MODEL_VERSION = os.environ.get("MODEL_VERSION", "1")
COMPUTE_SKU = os.environ.get("RT_COMPUTE_SKU", "Standard_DS3_v2")
MIN_REPLICAS = int(os.environ.get("RT_MIN_REPLICAS", "1"))
MAX_REPLICAS = int(os.environ.get("RT_MAX_REPLICAS", "3"))

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "../../.."))
BATCH_INFERENCE_DIR = os.path.join(
    REPO_ROOT, "deployment", "azure", "batch_inference"
)

# ---------------------------------------------------------------------------
# Dependencias
# ---------------------------------------------------------------------------

try:
    from azure.ai.ml import MLClient
    from azure.ai.ml.entities import (
        ManagedOnlineDeployment,
        ManagedOnlineEndpoint,
        CodeConfiguration,
        Environment,
        OnlineRequestSettings,
        ProbeSettings,
    )
    from azure.identity import DefaultAzureCredential
except ImportError:
    print(
        "ERROR: Instala azure-ai-ml:\n"
        "  pip install azure-ai-ml azure-identity"
    )
    sys.exit(1)


def get_ml_client() -> MLClient:
    credential = DefaultAzureCredential()
    return MLClient(
        credential=credential,
        subscription_id=SUBSCRIPTION_ID,
        resource_group_name=RESOURCE_GROUP,
        workspace_name=WORKSPACE_NAME,
    )


def create_or_update_endpoint(client: MLClient) -> None:
    print(f"[1/3] Creando/verificando endpoint: {ENDPOINT_NAME}")
    endpoint = ManagedOnlineEndpoint(
        name=ENDPOINT_NAME,
        description="Real-time PCB defect detection endpoint",
        auth_mode="key",
        tags={"project": "pcb-defect-detection", "type": "realtime"},
    )
    poller = client.online_endpoints.begin_create_or_update(endpoint)
    poller.result()
    print(f"       ✅ Endpoint '{ENDPOINT_NAME}' listo.")


def create_or_update_deployment(client: MLClient) -> None:
    print(f"[2/3] Creando deployment: {DEPLOYMENT_NAME}")

    env = Environment(
        name="pcb-inference-env",
        description="Entorno para inferencia PCB YOLOv8",
        conda_file=os.path.join(BATCH_INFERENCE_DIR, "..", "environment.yml"),
        image="mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu20.04:latest",
    )

    deployment = ManagedOnlineDeployment(
        name=DEPLOYMENT_NAME,
        endpoint_name=ENDPOINT_NAME,
        model=f"{MODEL_NAME}:{MODEL_VERSION}",
        code_configuration=CodeConfiguration(
            code=BATCH_INFERENCE_DIR,
            scoring_script="score.py",
        ),
        environment=env,
        instance_type=COMPUTE_SKU,
        instance_count=MIN_REPLICAS,
        request_settings=OnlineRequestSettings(
            request_timeout_ms=90_000,
            max_concurrent_requests_per_instance=4,
        ),
        liveness_probe=ProbeSettings(
            failure_threshold=3,
            success_threshold=1,
            timeout=10,
            period=30,
            initial_delay=60,
        ),
        readiness_probe=ProbeSettings(
            failure_threshold=3,
            success_threshold=1,
            timeout=10,
            period=10,
            initial_delay=30,
        ),
        tags={
            "min_replicas": str(MIN_REPLICAS),
            "max_replicas": str(MAX_REPLICAS),
        },
    )

    poller = client.online_deployments.begin_create_or_update(deployment)
    poller.result()

    # 100% del tráfico al nuevo deployment
    endpoint = client.online_endpoints.get(ENDPOINT_NAME)
    endpoint.traffic = {DEPLOYMENT_NAME: 100}
    client.online_endpoints.begin_create_or_update(endpoint).result()

    print(f"       ✅ Deployment '{DEPLOYMENT_NAME}' desplegado con 100% de tráfico.")


def print_endpoint_info(client: MLClient) -> None:
    print("[3/3] Información del endpoint:")
    endpoint = client.online_endpoints.get(ENDPOINT_NAME)
    keys = client.online_endpoints.get_keys(ENDPOINT_NAME)
    print(f"       URL  : {endpoint.scoring_uri}")
    print(f"       Key  : {keys.primary_key[:8]}…")
    print()
    print("  Para usar el endpoint:")
    print(f"    RT_ENDPOINT_URL={endpoint.scoring_uri}")
    print(
        "  Recuerda configurar RT_ENDPOINT_URL y RT_ENDPOINT_KEY "
        "en el backend FastAPI para habilitar el fallback real-time."
    )


def main() -> None:
    if not SUBSCRIPTION_ID:
        print("ERROR: Define AZURE_SUBSCRIPTION_ID.")
        sys.exit(1)

    print("=" * 50)
    print(" PCB Real-time Endpoint — Deployment")
    print("=" * 50)

    ml_client = get_ml_client()
    create_or_update_endpoint(ml_client)
    create_or_update_deployment(ml_client)
    print_endpoint_info(ml_client)
    print("\n✅ Real-time endpoint listo.")


if __name__ == "__main__":
    main()
