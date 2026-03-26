"""Orquestador del pipeline de entrenamiento YOLOv8n para Azure ML (SDK v2).

Pipeline de 4 pasos para fine-tuning del modelo YOLOv8n de segmentación de
defectos en PCB (Flux Solutions Cali):

  1. Ingest Data      - Descarga automática del dataset desde Hugging Face.
  2. Preprocess Split - Transformación (resize 640×640) y partición 80/20.
  3. Train YOLOv8n    - Fine-tuning del modelo con registro MLflow.
  4. Evaluate Model   - Inferencia, métricas (mAP) y exportación de resultados.

Uso (Windows PowerShell con uv):
    uv run python deployment/azure/pipeline_azure.py

Requisitos previos:
    * config.json en la raíz del proyecto con subscription_id, resource_group
      y workspace_name válidos.
    * El Workspace de Azure ML (pcb-ml-workspace) debe existir y estar
      accesible con las credenciales activas (az login).
    * El clúster de cómputo se crea automáticamente si no existe (DS3 v2).
    * No es necesario tener un dataset local: el pipeline lo descarga
      automáticamente desde Hugging Face.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from azure.ai.ml import Input, MLClient, Output, command, dsl
from azure.ai.ml.constants import AssetTypes
from azure.ai.ml.entities import AmlCompute, Environment
from azure.identity import DefaultAzureCredential

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# ── Rutas de referencia ──────────────────────────────────────────────────
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_CONFIG_FILE = _REPO_ROOT / "config.json"
_CONDA_FILE = Path(__file__).resolve().parent / "conda.yml"
_COMPONENTS_DIR = Path(__file__).resolve().parent / "components"

# ── Constantes del pipeline ──────────────────────────────────────────────
COMPUTE_NAME = "cpu-cluster-ds3"
ENVIRONMENT_NAME = "pcb-yolo-env"
ENVIRONMENT_VERSION = "6"
PIPELINE_NAME = "pcb-defect-pipeline"
EXPERIMENT_NAME = "pcb-defect-yolov8-finetuning"

# Hiperparámetros de fine-tuning
FINETUNE_EPOCHS = 50
FINETUNE_IMGSZ = 640
FINETUNE_BATCH = 16
FINETUNE_LR0 = 0.01
CONFIDENCE_THRESHOLD = 0.25
TRAIN_SPLIT = 0.8
RANDOM_SEED = 42


# ── 1. Conectar al Workspace ─────────────────────────────────────────────

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
        "Conectado al Workspace '%s' (RG: %s, Sub: %s)",
        cfg["workspace_name"],
        cfg["resource_group"],
        cfg["subscription_id"],
    )
    return client


# ── 2. Infraestructura (cómputo y entorno) ───────────────────────────────

def _ensure_compute(ml_client: MLClient) -> None:
    """Crea el clúster DS3 v2 si no existe."""
    try:
        cluster = ml_client.compute.get(COMPUTE_NAME)
        logger.info("Clúster '%s' ya existe (estado: %s).", COMPUTE_NAME, cluster.provisioning_state)
    except Exception:
        logger.info("Creando clúster '%s' (DS3 v2, 0-2 instancias)...", COMPUTE_NAME)
        cluster = AmlCompute(
            name=COMPUTE_NAME,
            type="amlcompute",
            size="STANDARD_DS3_V2",
            min_instances=0,
            max_instances=2,
            idle_time_before_scale_down=120,
        )
        ml_client.compute.begin_create_or_update(cluster).result()
        logger.info("Clúster '%s' creado con éxito.", COMPUTE_NAME)


def _ensure_environment(ml_client: MLClient) -> None:
    """Registra el entorno desde el Dockerfile si no existe."""
    from azure.ai.ml.entities import BuildContext
    
    env_tag = f"{ENVIRONMENT_NAME}:{ENVIRONMENT_VERSION}"
    try:
        ml_client.environments.get(ENVIRONMENT_NAME, version=ENVIRONMENT_VERSION)
        logger.info("Entorno '%s' ya está registrado.", env_tag)
    except Exception:
        logger.info("Registrando entorno '%s' desde Dockerfile...", env_tag)
        dockerfile_path = Path(__file__).resolve().parent / "Dockerfile"
        
        if not dockerfile_path.exists():
            raise FileNotFoundError(f"Dockerfile no encontrado en {dockerfile_path}")
        
        # Usar BuildContext para construir la imagen desde Dockerfile
        env = Environment(
            name=ENVIRONMENT_NAME,
            version=ENVIRONMENT_VERSION,
            description="Entorno para pipeline YOLOv8 PCB - Dockerfile + Conda",
            build=BuildContext(path=str(dockerfile_path.parent)),
        )
        ml_client.environments.create_or_update(env)
        logger.info("Entorno '%s' registrado desde Dockerfile.", env_tag)

# ── 3. Definición de los componentes ────────────────────────────────────

_ENV_REF = f"azureml:{ENVIRONMENT_NAME}:{ENVIRONMENT_VERSION}"

ingest_component = command(
    name="ingest_data",
    display_name="1. Ingest Data (Hugging Face)",
    description="Descarga el dataset keremberke/pcb-defect-segmentation desde Hugging Face.",
    outputs={"output_data": Output(type=AssetTypes.URI_FOLDER)},
    code=str(_COMPONENTS_DIR),
    command="python ingest_data.py --output_data ${{outputs.output_data}}",
    environment=_ENV_REF,
    compute=COMPUTE_NAME,
)

preprocess_component = command(
    name="preprocess_split",
    display_name="2. Preprocess & Split (640x640, 80/20)",
    description="Redimensiona imágenes a 640x640 y divide el dataset en 80% train / 20% test.",
    inputs={"input_data": Input(type=AssetTypes.URI_FOLDER)},
    outputs={
        "train_output": Output(type=AssetTypes.URI_FOLDER),
        "test_output": Output(type=AssetTypes.URI_FOLDER),
    },
    code=str(_COMPONENTS_DIR),
    command=(
        "python preprocess_split.py "
        "--input_data ${{inputs.input_data}} "
        "--train_output ${{outputs.train_output}} "
        "--test_output ${{outputs.test_output}} "
        f"--image_size {FINETUNE_IMGSZ} "
        f"--train_ratio {TRAIN_SPLIT} "
        f"--seed {RANDOM_SEED}"
    ),
    environment=_ENV_REF,
    compute=COMPUTE_NAME,
)

train_component = command(
    name="train_yolo",
    display_name="3. Train YOLOv8n (fine-tuning)",
    description="Fine-tuning de YOLOv8n sobre el dataset PCB con registro MLflow.",
    inputs={"input_data": Input(type=AssetTypes.URI_FOLDER)},
    outputs={"output_data": Output(type=AssetTypes.URI_FOLDER)},
    code=str(_COMPONENTS_DIR),
    command=(
        "python train_yolo.py "
        "--input_data ${{inputs.input_data}} "
        "--output_data ${{outputs.output_data}} "
        f"--epochs {FINETUNE_EPOCHS} "
        f"--imgsz {FINETUNE_IMGSZ} "
        f"--batch {FINETUNE_BATCH} "
        f"--lr0 {FINETUNE_LR0}"
    ),
    environment=_ENV_REF,
    compute=COMPUTE_NAME,
)

evaluate_component = command(
    name="evaluate_model",
    display_name="4. Evaluate Model (mAP + exportación)",
    description="Inferencia sobre el test set, cálculo de mAP y exportación de resultados.",
    inputs={
        "input_data": Input(type=AssetTypes.URI_FOLDER),
        "model_data": Input(type=AssetTypes.URI_FOLDER),
    },
    outputs={
        "output_data": Output(
            type=AssetTypes.URI_FOLDER,
            path="azureml://datastores/workspaceblobstore/paths/pcb-results/",
        )
    },
    code=str(_COMPONENTS_DIR),
    command=(
        "python evaluate_model.py "
        "--input_data ${{inputs.input_data}} "
        "--model_data ${{inputs.model_data}} "
        "--output_data ${{outputs.output_data}} "
        f"--conf_threshold {CONFIDENCE_THRESHOLD}"
    ),
    environment=_ENV_REF,
    compute=COMPUTE_NAME,
)


# ── 4. Definición del pipeline ───────────────────────────────────────────

@dsl.pipeline(
    name=PIPELINE_NAME,
    description=(
        "Pipeline YOLOv8n para detección de defectos en PCB - Flux Solutions Cali. "
        "4 pasos: Ingest (HuggingFace) -> Preprocess/Split -> Train -> Evaluate."
    ),
    default_compute=COMPUTE_NAME,
)
def pcb_training_pipeline():
    """Pipeline modular de 4 pasos: ingesta, preprocesamiento, entrenamiento y evaluación."""
    step_ingest = ingest_component()

    step_preprocess = preprocess_component(
        input_data=step_ingest.outputs.output_data,
    )

    step_train = train_component(
        input_data=step_preprocess.outputs.train_output,
    )

    evaluate_component(
        input_data=step_preprocess.outputs.test_output,
        model_data=step_train.outputs.output_data,
    )


# ── 5. Punto de entrada ──────────────────────────────────────────────────

def main() -> None:
    ml_client = _get_ml_client()
    _ensure_compute(ml_client)
    _ensure_environment(ml_client)

    pipeline_job = pcb_training_pipeline()
    pipeline_job.experiment_name = EXPERIMENT_NAME

    submitted = ml_client.jobs.create_or_update(pipeline_job)
    logger.info("Pipeline enviado. Job name: %s", submitted.name)
    logger.info("Monitorea el progreso en: https://ml.azure.com")


if __name__ == "__main__":
    main()
