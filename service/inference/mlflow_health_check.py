"""
Health-check de MLflow para el servicio de inferencia.

Carga el modelo de HuggingFace y registra los metadatos en MLflow.
Este módulo NO ejecuta nada al importarse: toda la lógica está dentro
de main(), que solo se llama si el módulo se ejecuta directamente.

Uso (línea de comandos):
    make link_model   ->  Establece HF_MODEL_ID y ejecuta el health check
    make healthcheck  ->  Ejecuta el health check
                          (requiere HF_MODEL_ID en .env)

Variables de entorno leídas (desde .env o el entorno del SO):
    HF_MODEL_ID           - ID del modelo de HuggingFace (requerido)
    MLFLOW_TRACKING_URI   - URI del servidor de MLflow
                            (por defecto: sqlite:///mlflow.db)
"""

import os
import sys
from pathlib import Path

import mlflow
from dotenv import load_dotenv


def main() -> None:
    """Carga el modelo y reporta el resultado a MLflow.

    Resuelve los imports según cómo se invoca el módulo:
    - Como script directo
      (``python service/inference/mlflow_health_check.py``):
      añade la raíz del proyecto a sys.path y usa imports absolutos.
    - Como módulo del paquete
      (``python -m service.inference.mlflow_health_check``):
      usa imports relativos.
    """
    if __package__ is None or __package__ == "":
        # Ejecución directa como script: ajustar sys.path.
        # El import de model_loader debe estar aquí porque
        # necesita sys.path ya modificado para resolver la ruta.
        sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
        from service.inference.model_loader import (  # noqa: PLC0415
            init_inference_artifacts,
            report_loaded_to_mlflow,
        )
    else:
        # Ejecución como módulo del paquete (python -m ...):
        # usar import relativo.
        from .model_loader import (  # noqa: PLC0415
            init_inference_artifacts,
            report_loaded_to_mlflow,
        )

    # Cargar variables de entorno desde .env (si existe) para que
    # HF_MODEL_ID y MLFLOW_TRACKING_URI estén disponibles.
    load_dotenv()

    # Apuntar al mismo almacén SQLite que usa `make mlflow` para que
    # las ejecuciones sean visibles en la UI.
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
    mlflow.set_tracking_uri(tracking_uri)

    hf_model_id = os.getenv("HF_MODEL_ID", "").strip()

    mlflow.set_experiment("ImageAivsReal-Service-Health")

    with mlflow.start_run(run_name="startup-model-load"):
        artifacts = init_inference_artifacts(
            hf_model_id=hf_model_id,
            device="cpu",
        )
        report_loaded_to_mlflow(artifacts=artifacts)
        print("✅ OK: reportado a MLflow")


if __name__ == "__main__":
    main()
