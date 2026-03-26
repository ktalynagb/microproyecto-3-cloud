"""Orquestador del pipeline de inferencia por lote (Batch Endpoint) en Azure ML.

Registra el modelo YOLOv8n entrenado y crea/actualiza el Batch Endpoint de
inferencia para inspección de defectos en PCB sobre el Workspace de Azure ML.

Uso (después de ejecutar el pipeline de entrenamiento):
    uv run python deployment/azure/inference_pipeline.py

Requisitos previos:
    1. El pipeline de entrenamiento debe haber completado:
           uv run python deployment/azure/pipeline_azure.py
    2. ``config.json`` en la raíz del proyecto con subscription_id,
       resource_group y workspace_name válidos.
    3. El modelo entrenado (``best.pt``) debe estar disponible en el
       Blob Store de Azure ML o en la ruta indicada por MODEL_OUTPUT_PATH.
    4. Las variables de entorno del archivo ``.env`` deben estar configuradas.

Variables de entorno usadas:
    AZURE_STORAGE_ACCOUNT   Nombre de la cuenta de Blob Storage.
    AZURE_STORAGE_KEY       Clave de acceso a Blob Storage.
    AZURE_CONTAINER_NAME    Contenedor para resultados (default: pcb-results).
    PCB_MODEL_PATH          Ruta local al best.pt (fallback si no se descarga).
    PCB_CONF_THRESHOLD      Umbral de confianza (default: 0.25).
    PCB_INFERENCE_TIMEOUT   Timeout por imagen en segundos (default: 30).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from azure.ai.ml import MLClient
from azure.ai.ml.constants import BatchDeploymentOutputAction
from azure.ai.ml.entities import (
    AmlCompute,
    BatchDeployment,
    BatchEndpoint,
    BatchRetrySettings,
    CodeConfiguration,
    Environment,
    Model,
)
from azure.identity import DefaultAzureCredential

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# ── Rutas de referencia ───────────────────────────────────────────────────
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_CONFIG_FILE = _REPO_ROOT / "config.json"
_BATCH_INFERENCE_DIR = Path(__file__).resolve().parent / "batch_inference"

# ── Constantes ────────────────────────────────────────────────────────────
COMPUTE_NAME = "cpu-cluster-ds3"
ENVIRONMENT_NAME = "pcb-yolo-env"
ENVIRONMENT_VERSION = "8"

# Nombre del modelo registrado en Azure ML Model Registry
MODEL_NAME = "pcb-yolov8n"

# Nombre y deployment del Batch Endpoint
BATCH_ENDPOINT_NAME = "pcb-batch-inference"
BATCH_DEPLOYMENT_NAME = "pcb-yolov8n-deployment"

# Ruta por defecto del artefacto best.pt generado por el pipeline de
# entrenamiento (coincide con el ``path`` de la salida de evaluate_component
# en pipeline_azure.py).
DEFAULT_MODEL_OUTPUT_PATH = (
    "azureml://datastores/workspaceblobstore/paths/pcb-results/"
)

_ENV_REF = f"azureml:{ENVIRONMENT_NAME}:{ENVIRONMENT_VERSION}"


# ── 1. Helpers de autenticación y configuración ───────────────────────────

def _load_config() -> dict:
    """Lee la configuración de Azure ML desde config.json."""
    if not _CONFIG_FILE.exists():
        raise FileNotFoundError(
            f"No se encontró config.json en {_REPO_ROOT}.\n"
            "Crea el archivo con subscription_id, resource_group y workspace_name."
        )
    with open(_CONFIG_FILE, encoding="utf-8") as fh:
        cfg = json.load(fh)
    required = {"subscription_id", "resource_group", "workspace_name"}
    missing = required - cfg.keys()
    if missing:
        raise KeyError(f"Faltan las siguientes claves en config.json: {missing}")
    return cfg


def _get_ml_client() -> MLClient:
    """Autentica y retorna un cliente de Azure ML."""
    cfg = _load_config()
    credential = DefaultAzureCredential()
    client = MLClient(
        credential=credential,
        subscription_id=cfg["subscription_id"],
        resource_group_name=cfg["resource_group"],
        workspace_name=cfg["workspace_name"],
    )
    logger.info(
        "Conectado al Workspace '%s' (RG: %s)",
        cfg["workspace_name"],
        cfg["resource_group"],
    )
    return client


# ── 2. Registro del modelo ────────────────────────────────────────────────

def register_model(
    ml_client: MLClient,
    model_path: str = DEFAULT_MODEL_OUTPUT_PATH,
) -> Model:
    """Registra el modelo entrenado en Azure ML Model Registry.

    Args:
        ml_client: Cliente autenticado de Azure ML.
        model_path: URI del directorio con best.pt (salida del pipeline de
            entrenamiento). Puede ser una ruta local o un URI de Azure ML.

    Returns:
        El objeto Model registrado.
    """
    logger.info("Registrando modelo '%s' desde: %s", MODEL_NAME, model_path)
    model = Model(
        path=model_path,
        name=MODEL_NAME,
        description=(
            "YOLOv8n fine-tuned para detección/segmentación de defectos en PCB. "
            "Clases: dry_joint, incorrect_installation, pcb_damage, short_circuit."
        ),
        type="custom_model",
    )
    registered = ml_client.models.create_or_update(model)
    logger.info(
        "Modelo registrado: %s (versión %s)",
        registered.name,
        registered.version,
    )
    return registered


# ── 3. Creación del Batch Endpoint ────────────────────────────────────────

def create_batch_endpoint(ml_client: MLClient) -> BatchEndpoint:
    """Crea (o actualiza) el Batch Endpoint de inferencia.

    Args:
        ml_client: Cliente autenticado de Azure ML.

    Returns:
        El objeto BatchEndpoint creado/actualizado.
    """
    logger.info("Creando/actualizando Batch Endpoint '%s'...", BATCH_ENDPOINT_NAME)
    endpoint = BatchEndpoint(
        name=BATCH_ENDPOINT_NAME,
        description=(
            "Batch inference endpoint para detección de defectos en PCBs "
            "(hasta 10 imágenes por lote)."
        ),
    )
    ml_client.batch_endpoints.begin_create_or_update(endpoint).result()
    logger.info("Batch Endpoint '%s' listo.", BATCH_ENDPOINT_NAME)
    return endpoint


# ── 4. Creación del Batch Deployment ─────────────────────────────────────

def create_batch_deployment(
    ml_client: MLClient,
    registered_model: Model,
) -> BatchDeployment:
    """Crea (o actualiza) el Batch Deployment que apunta al script de scoring.

    El script ``score.py`` en ``batch_inference/`` implementa ``init()`` y
    ``run()`` según el contrato de Azure ML Managed Batch Endpoints.

    Args:
        ml_client: Cliente autenticado de Azure ML.
        registered_model: Modelo registrado en el Model Registry.

    Returns:
        El objeto BatchDeployment creado/actualizado.
    """
    logger.info("Creando/actualizando Batch Deployment '%s'...", BATCH_DEPLOYMENT_NAME)
    deployment = BatchDeployment(
        name=BATCH_DEPLOYMENT_NAME,
        endpoint_name=BATCH_ENDPOINT_NAME,
        model=registered_model.id,
        code_configuration=CodeConfiguration(
            code=str(_BATCH_INFERENCE_DIR),
            scoring_script="score.py",
        ),
        environment=_ENV_REF,
        compute=COMPUTE_NAME,
        instance_count=1,
        max_concurrency_per_instance=1,
        # Azure ML invocará run() con mini-lotes de hasta 10 imágenes
        mini_batch_size=10,
        output_action=BatchDeploymentOutputAction.APPEND_ROW,
        output_file_name="predictions.jsonl",
        retry_settings=BatchRetrySettings(max_retries=2, timeout=60),
        environment_variables={
            "PCB_CONF_THRESHOLD": "0.25",
            "PCB_IOU_THRESHOLD": "0.45",
            "PCB_INFERENCE_TIMEOUT": "30",
        },
    )
    ml_client.batch_deployments.begin_create_or_update(deployment).result()
    logger.info(
        "Batch Deployment '%s' desplegado en endpoint '%s'.",
        BATCH_DEPLOYMENT_NAME,
        BATCH_ENDPOINT_NAME,
    )
    return deployment


# ── 5. Punto de entrada ───────────────────────────────────────────────────

def main(model_path: str = DEFAULT_MODEL_OUTPUT_PATH) -> None:
    """Registra el modelo y crea/actualiza el Batch Endpoint completo.

    Args:
        model_path: URI del artefacto best.pt generado por el pipeline de
            entrenamiento. Por defecto usa la salida estándar del pipeline.
    """
    ml_client = _get_ml_client()

    # 1. Registrar el modelo
    registered_model = register_model(ml_client, model_path)

    # 2. Crear el Batch Endpoint
    create_batch_endpoint(ml_client)

    # 3. Crear el Batch Deployment
    create_batch_deployment(ml_client, registered_model)

    # 4. Obtener el scoring URI del endpoint
    endpoint = ml_client.batch_endpoints.get(BATCH_ENDPOINT_NAME)
    logger.info("=" * 60)
    logger.info("Pipeline de inferencia desplegado con éxito.")
    logger.info("Endpoint: %s", BATCH_ENDPOINT_NAME)
    logger.info("Deployment: %s", BATCH_DEPLOYMENT_NAME)
    if hasattr(endpoint, "scoring_uri") and endpoint.scoring_uri:
        logger.info("Scoring URI: %s", endpoint.scoring_uri)
    logger.info("=" * 60)
    logger.info(
        "Configura en .env:\n"
        "  API_HOST=<scoring_uri_host>\n"
        "  API_PORT=443\n"
        "Para que el frontend Streamlit invoque el endpoint."
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Despliega el pipeline de inferencia batch en Azure ML."
    )
    parser.add_argument(
        "--model_path",
        default=DEFAULT_MODEL_OUTPUT_PATH,
        help="URI o ruta local al artefacto best.pt del pipeline de entrenamiento.",
    )
    args = parser.parse_args()
    main(model_path=args.model_path)
