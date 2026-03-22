"""Servidor FastAPI para detección de defectos en PCB - Flux Solutions Cali.

Expone un endpoint POST /predict que recibe una imagen y retorna el resultado
de la detección de defectos usando el modelo YOLOv8.

Simula un Endpoint de Azure Machine Learning con una API REST estándar.

Env vars:
    HF_MODEL_ID  - ID del modelo en Hugging Face
                   (default: keremberke/yolov8n-pcb-defect-segmentation)
    LOG_LEVEL    - Nivel de logging (default: INFO)
    API_HOST     - Host del servidor (default: 0.0.0.0)
    API_PORT     - Puerto del servidor (default: 8000)

Uso:
    uv run service/inference_server.py
    o:
    make api-server
"""

import logging
import os
from typing import Any, Dict, List

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel

from inference.inference_engine import get_model, run_inference

load_dotenv()

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", "8000"))

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL.upper(), logging.INFO),
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="PCB Defect Detection API - Flux Solutions Cali",
    description=(
        "API REST para inspección de calidad de PCB mediante visión artificial. "
        "Detecta defectos como Dry_joint, Incorrect_installation, PCB_damage, "
        "Short_circuit, Mousebites y Opens usando YOLOv8."
    ),
    version="2.0.0",
)

_model: Any = None


@app.on_event("startup")
async def startup_event() -> None:
    """Carga el modelo YOLO al iniciar el servidor."""
    global _model
    logger.info("Cargando modelo de detección de defectos en PCB...")
    _model = get_model()
    logger.info("Servidor listo para recibir peticiones.")


class DetectionResponse(BaseModel):
    """Respuesta estándar del endpoint /predict."""

    status: str
    processed_image_base64: str
    has_defects: bool
    defects_summary: List[Dict[str, Any]]
    error_message: str = ""


@app.post("/predict", response_model=DetectionResponse)
async def predict(file: UploadFile = File(...)) -> DetectionResponse:
    """Detecta defectos en una imagen PCB.

    Recibe una imagen JPG/PNG, ejecuta el modelo YOLOv8 de detección de
    defectos y retorna la imagen procesada (con bounding boxes) codificada
    en Base64, junto al resumen de defectos encontrados.

    Args:
        file: Imagen JPG o PNG de la PCB a inspeccionar.

    Returns:
        DetectionResponse con la imagen procesada en Base64, indicador de
        defectos y lista de hallazgos con clase y confianza.
    """
    if _model is None:
        logger.error("Modelo no disponible al recibir petición /predict")
        return DetectionResponse(
            status="error",
            processed_image_base64="",
            has_defects=False,
            defects_summary=[],
            error_message="El modelo no está disponible. El servidor aún puede estar inicializando.",
        )

    logger.info("Recibida petición /predict para archivo: %s", file.filename)
    image_bytes = await file.read()

    result = run_inference(image_bytes, _model)

    if result["status"] == "error":
        logger.warning("Error en inferencia: %s", result.get("error"))
        return DetectionResponse(
            status="error",
            processed_image_base64="",
            has_defects=False,
            defects_summary=[],
            error_message=result.get("error", "Error desconocido"),
        )

    logger.info(
        "Inferencia OK: has_defects=%s, defectos=%d, inference_ms=%.1f",
        result["has_defects"],
        len(result["defects_summary"]),
        result["timing"]["inference_ms"],
    )

    return DetectionResponse(
        status="ok",
        processed_image_base64=result["processed_image_base64"],
        has_defects=result["has_defects"],
        defects_summary=result["defects_summary"],
        error_message="",
    )


if __name__ == "__main__":
    uvicorn.run(
        app,
        host=API_HOST,
        port=API_PORT,
        log_level=LOG_LEVEL.lower(),
    )

