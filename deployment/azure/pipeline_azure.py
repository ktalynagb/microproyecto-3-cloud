"""Orquestador del pipeline de entrenamiento YOLOv8n para Azure ML (SDK v2).

Pipeline de 4 pasos para fine-tuning del modelo YOLOv8n de segmentación de
defectos en PCB (Flux Solutions Cali):

  1. Ingest Data      - Descarga automática del dataset desde Roboflow.
  2. Preprocess Split - Transformación (resize 640×640) y partición 80/20.
  3. Train YOLOv8n    - Fine-tuning del modelo con registro MLflow.
  4. Evaluate Model   - Inferencia, métricas (mAP) y exportación de resultados.

Adicionalmente registra el modelo entrenado como Batch Endpoint de Azure ML
para inferencia en lote (POST /score con hasta 10 imágenes PCB).

Uso (Windows PowerShell con uv):
    $env:ROBOFLOW_API_KEY = "<tu_api_key>"
    uv run python deployment/azure/pipeline_azure.py

Requisitos previos:
    * config.json en la raíz del proyecto con subscription_id, resource_group
      y workspace_name válidos.
    * ROBOFLOW_API_KEY exportada como variable de entorno (ver README.md sección 8).
    * El Workspace de Azure ML (pcb-ml-workspace) debe existir y estar
      accesible con las credenciales activas (az login).
    * El clúster de cómputo se crea automáticamente si no existe (DS3 v2).
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
import time  

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
_COMPONENTS_DIR = Path(__file__).resolve().parent / "components"
_BATCH_INFERENCE_DIR = Path(__file__).resolve().parent / "batch_inference"

# ── Constantes del pipeline ──────────────────────────────────────────────
COMPUTE_NAME = "cpu-cluster-ds3"
ENVIRONMENT_NAME = "pcb-yolo-env"
ENVIRONMENT_VERSION = "8"
PIPELINE_NAME = "pcb-defect-pipeline"
EXPERIMENT_NAME = "pcb-defect-yolov8-finetuning"

# Hiperparámetros de fine-tuning
FINETUNE_EPOCHS = 1
FINETUNE_IMGSZ = 640
FINETUNE_BATCH = 16
FINETUNE_LR0 = 0.01
CONFIDENCE_THRESHOLD = 0.25
TRAIN_SPLIT = 0.8
RANDOM_SEED = 42

# Nombre del endpoint de batch inference
BATCH_ENDPOINT_NAME = "pcb-batch-inference"
BATCH_DEPLOYMENT_NAME = "pcb-yolov8n-deployment"

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
    display_name="1. Ingest Data (Roboflow)",
    description=(
        "Descarga el dataset diplom-qz7q6/defects-2q87r v8 desde Roboflow. "
        "Requiere ROBOFLOW_API_KEY como variable de entorno."
    ),
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
    description=(
        "Fine-tuning de YOLOv8n sobre el dataset PCB con registro MLflow. "
        "Detecta automáticamente si los labels son segmentación o detección."
    ),
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
        f"--lr0 {FINETUNE_LR0} "
        "--task auto"
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
        "4 pasos: Ingest (Roboflow) -> Preprocess/Split -> Train -> Evaluate."
    ),
    default_compute=COMPUTE_NAME,
)
def pcb_training_pipeline():
    """Pipeline modular de 4 pasos: ingesta, preprocesamiento, entrenamiento y evaluación."""
    step_ingest = ingest_component()
    # Pasar ROBOFLOW_API_KEY al step de ingesta para descargar el dataset
    roboflow_api_key = os.environ.get("ROBOFLOW_API_KEY", "")
    if roboflow_api_key:
        step_ingest.environment_variables = {"ROBOFLOW_API_KEY": roboflow_api_key}
    else:
        logger.warning(
            "⚠️ ROBOFLOW_API_KEY no definida. El paso de ingesta usará fallback "
            "Hugging Face y puede obtener 0 imágenes. "
            "Define ROBOFLOW_API_KEY antes de ejecutar el pipeline (ver README.md sección 8)."
        )

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


# ── 5. Registro del Batch Endpoint ───────────────────────────────────────

def _register_batch_endpoint(ml_client: MLClient, model_output_path: str) -> None:
    """Registra el modelo entrenado como Batch Endpoint de Azure ML.

    Crea un endpoint de inferencia por lote que acepta hasta 10 imágenes PCB
    y devuelve detecciones con clase, bbox y confidence.

    Args:
        ml_client: Cliente autenticado de Azure ML.
        model_output_path: URI del directorio con best.pt (salida de train_yolo).
    """
    from azure.ai.ml.entities import (
        BatchDeployment,
        BatchEndpoint,
        BatchRetrySettings,
        CodeConfiguration,
        Model,
    )
    from azure.ai.ml.constants import BatchDeploymentOutputAction

    logger.info("Registrando modelo en Azure ML Model Registry...")
    model = Model(
        path=model_output_path,
        name="pcb-yolov8n",
        description="YOLOv8n fine-tuned para detección de defectos en PCB.",
        type="custom_model",
    )
    registered_model = ml_client.models.create_or_update(model)
    logger.info("Modelo registrado: %s (versión %s)", registered_model.name, registered_model.version)

    # Crear Batch Endpoint
    logger.info("Creando Batch Endpoint '%s'...", BATCH_ENDPOINT_NAME)
    endpoint = BatchEndpoint(
        name=BATCH_ENDPOINT_NAME,
        description="Batch inference endpoint para detección de defectos en PCBs.",
    )
    ml_client.batch_endpoints.begin_create_or_update(endpoint).result()

    # Crear Batch Deployment
    logger.info("Creando Batch Deployment '%s'...", BATCH_DEPLOYMENT_NAME)
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
        mini_batch_size=10,
        output_action=BatchDeploymentOutputAction.APPEND_ROW,
        retry_settings=BatchRetrySettings(max_retries=2, timeout=60),
    )
    ml_client.batch_deployments.begin_create_or_update(deployment).result()
    logger.info(
        "Batch Endpoint '%s' listo. Deployment: '%s'.",
        BATCH_ENDPOINT_NAME,
        BATCH_DEPLOYMENT_NAME,
    )


# ── 6. Punto de entrada ──────────────────────────────────────────────────

def main() -> None:
    ml_client = _get_ml_client()
    _ensure_compute(ml_client)
    _ensure_environment(ml_client)

    # 1. Ejecutar el pipeline de entrenamiento
    pipeline_job = pcb_training_pipeline()
    pipeline_job.experiment_name = EXPERIMENT_NAME

    submitted = ml_client.jobs.create_or_update(pipeline_job)
    logger.info("=" * 60)
    logger.info("Pipeline enviado. Job name: %s", submitted.name)
    logger.info("Monitorea el progreso en: https://ml.azure.com")
    logger.info("=" * 60)

    # 2. ✅ NUEVO: Esperar a que termine y registrar el modelo
    logger.info("\n⏳ Esperando a que termine el pipeline...")
    completed_job = ml_client.jobs.stream(submitted.name)
    
    if completed_job.status == "Completed":
        logger.info("✅ Pipeline completado exitosamente!")
        
        # 3. ✅ Obtener la salida (ruta del best.pt)
        # La salida viene del último paso (evaluate_model)
        outputs = completed_job.outputs
        if hasattr(outputs, "output_data") or "output_data" in dir(outputs):
            model_output_uri = outputs.output_data.path if hasattr(outputs, "output_data") else None
            
            if not model_output_uri:
                # Fallback: usar la ruta estándar
                model_output_uri = (
                    "azureml://datastores/workspaceblobstore/paths/pcb-results/best.pt"
                )
            
            logger.info("\n📦 Registrando modelo en Azure ML Model Registry...")
            logger.info("   URI del modelo: %s", model_output_uri)
            
            # 4. ✅ Registrar el modelo
            from azure.ai.ml.entities import Model
            
            model = Model(
                path=model_output_uri,
                name="pcb-yolov8n",
                version=str(int(time.time())),  # Usar timestamp como versión
                description=(
                    "YOLOv8n fine-tuned para detección/segmentación de defectos en PCB. "
                    f"Job: {submitted.name}"
                ),
                type="custom_model",
            )
            
            registered_model = ml_client.models.create_or_update(model)
            logger.info("✅ Modelo registrado exitosamente!")
            logger.info("   Nombre: %s", registered_model.name)
            logger.info("   Versión: %s", registered_model.version)
            logger.info("   ID: %s", registered_model.id)
            
            # 5. ✅ Guardar la información para referencia futura
            model_info = {
                "name": registered_model.name,
                "version": registered_model.version,
                "id": registered_model.id,
                "path": model_output_uri,
                "job_name": submitted.name,
                "timestamp": str(registered_model.creation_context.created_at),
            }
            
            model_info_file = _REPO_ROOT / "model_info.json"
            with open(model_info_file, "w") as f:
                json.dump(model_info, f, indent=2)
            logger.info("   Información guardada en: %s", model_info_file)
            
            logger.info("\n" + "=" * 60)
            logger.info("🚀 PRÓXIMO PASO:")
            logger.info("   Ejecuta el pipeline de inferencia:")
            logger.info("   uv run python deployment/azure/inference_pipeline.py")
            logger.info("=" * 60)
        else:
            logger.error("❌ No se pudo encontrar la salida del pipeline")
    else:
        logger.error("❌ Pipeline falló con estado: %s", completed_job.status)


if __name__ == "__main__":
    main()
