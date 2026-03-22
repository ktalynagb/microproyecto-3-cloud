"""Motor de inferencia para detección/segmentación de defectos en PCB.

Usa el modelo YOLOv8 de Ultralytics (keremberke/yolov8n-pcb-defect-segmentation)
para detectar fallas como Dry_joint, Incorrect_installation, PCB_damage,
Short_circuit, Mousebites y Opens.

La función run_inference recibe los bytes de la imagen, la convierte a array
NumPy/OpenCV y la pasa al modelo YOLO. Retorna la imagen procesada con bounding
boxes codificada en Base64, un booleano has_defects y una lista de defectos
con su clase y confianza.

La función NUNCA propaga excepciones al caller: cualquier error se captura
internamente y se retorna como respuesta estandarizada con status="error".

Uso:
    from service.inference.inference_engine import run_inference, get_model
    model = get_model()
    result = run_inference(image_bytes, model)
    if result["status"] == "ok":
        print(result["has_defects"])
        print(result["defects_summary"])
    else:
        print(result["error"])
"""

import base64
import logging
import os
import threading
import time
from typing import Any, Dict, List, Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)

_model: Optional[Any] = None
_model_lock = threading.Lock()


def get_model() -> Any:
    """Carga y cachea el modelo YOLO en memoria (singleton thread-safe).

    Lee HF_MODEL_ID del entorno. Por defecto usa el modelo de PCB de
    keremberke en Hugging Face.

    Returns:
        Instancia de ultralytics.YOLO lista para inferencia.
    """
    global _model
    if _model is None:
        with _model_lock:
            if _model is None:
                from ultralytics import YOLO

                hf_model_id = os.getenv(
                    "HF_MODEL_ID",
                    "keremberke/yolov8n-pcb-defect-segmentation",
                )
                logger.info("Cargando modelo YOLO: %s", hf_model_id)
                _model = YOLO(hf_model_id)
                logger.info("Modelo YOLO cargado correctamente.")
    return _model


def run_inference(
    image_bytes: bytes,
    model: Optional[Any] = None,
) -> Dict[str, Any]:
    """Ejecuta detección de defectos sobre los bytes de una imagen PCB.

    Convierte los bytes a un array NumPy/OpenCV, ejecuta el modelo YOLO,
    genera la imagen anotada con bounding boxes y extrae el resumen de
    defectos detectados.

    Esta función NUNCA propaga excepciones al caller. Cualquier error se
    captura y se retorna como dict con status="error".

    Args:
        image_bytes: Bytes crudos de la imagen (JPG o PNG).
        model: Instancia YOLO ya cargada. Si es None, se carga el modelo
            mediante get_model().

    Returns:
        Dict con la siguiente estructura en caso exitoso::

            {
                "status": "ok",
                "has_defects": True,
                "defects_summary": [
                    {"class": "Dry_joint", "confidence": 0.87},
                    {"class": "Short_circuit", "confidence": 0.72}
                ],
                "processed_image_base64": "<base64 string>",
                "timing": {"inference_ms": 123.456},
                "error": None
            }

        Dict con la siguiente estructura en caso de error::

            {
                "status": "error",
                "has_defects": False,
                "defects_summary": [],
                "processed_image_base64": "",
                "timing": {"inference_ms": 0.0},
                "error": "Descripcion del error."
            }
    """
    if model is None:
        model = get_model()

    t0 = time.perf_counter()

    try:
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            return {
                "status": "error",
                "has_defects": False,
                "defects_summary": [],
                "processed_image_base64": "",
                "timing": {"inference_ms": 0.0},
                "error": "No se pudo decodificar la imagen. Verifica que sea un JPG o PNG válido.",
            }

        results = model(img)

        t1 = time.perf_counter()
        inference_ms = round((t1 - t0) * 1000, 3)

        # Generar imagen anotada con bounding boxes / máscaras
        annotated_bgr = results[0].plot()
        # Codificar como JPEG y luego como Base64
        _, buffer = cv2.imencode(".jpg", annotated_bgr)
        processed_image_b64 = base64.b64encode(buffer.tobytes()).decode("utf-8")

        # Extraer resumen de defectos detectados
        defects_summary: List[Dict[str, Any]] = []
        has_defects = False

        if results[0].boxes is not None and len(results[0].boxes) > 0:
            has_defects = True
            for box in results[0].boxes:
                cls_id = int(box.cls.item())
                cls_name = model.names[cls_id]
                confidence = round(float(box.conf.item()), 4)
                defects_summary.append(
                    {"class": cls_name, "confidence": confidence}
                )

        logger.debug(
            "Inferencia OK: has_defects=%s, defectos=%d, inference_ms=%.3f",
            has_defects,
            len(defects_summary),
            inference_ms,
        )

        return {
            "status": "ok",
            "has_defects": has_defects,
            "defects_summary": defects_summary,
            "processed_image_base64": processed_image_b64,
            "timing": {"inference_ms": inference_ms},
            "error": None,
        }

    except Exception as exc:
        logger.error("Error durante la inferencia YOLO: %s", exc)
        return {
            "status": "error",
            "has_defects": False,
            "defects_summary": [],
            "processed_image_base64": "",
            "timing": {"inference_ms": 0.0},
            "error": f"Error durante la inferencia del modelo: {exc}",
        }

