"""Script de scoring para el Batch Endpoint de Azure ML.

Sigue el contrato de Azure ML Managed Batch Endpoints:
  - ``init()``   → llamado una vez al arrancar el nodo de cómputo.
  - ``run(mini_batch)`` → llamado por cada mini-lote de rutas de imagen.

El Batch Endpoint recibe hasta 10 imágenes por lote y retorna, para cada
una, una línea JSON con clase, bounding-box, máscara y confidence.

Variables de entorno esperadas:
    PCB_MODEL_PATH        Ruta al archivo best.pt (default: "best.pt").
    PCB_CONF_THRESHOLD    Umbral de confianza YOLOv8 (default: 0.25).
    PCB_IOU_THRESHOLD     Umbral IoU NMS (default: 0.45).
    PCB_INFERENCE_TIMEOUT Segundos máximos por imagen (default: 30).
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

# El directorio de este script se añade automáticamente por Azure ML al sys.path
# cuando se usa CodeConfiguration. No es necesario manipular sys.path manualmente.

from inference_engine import BatchInference
from batch_receiver import BatchReceiver
from config import config
from logger import get_logger
from post_processor import PostProcessor

logger = get_logger("score")

_engine: BatchInference | None = None
_receiver: BatchReceiver | None = None
_processor: PostProcessor | None = None


def init() -> None:
    """Inicializa el modelo y los servicios auxiliares (llamado una sola vez)."""
    global _engine, _receiver, _processor

    model_path = os.environ.get("PCB_MODEL_PATH", config.model_path)
    # Azure ML monta el modelo en AZUREML_MODEL_DIR
    azure_model_dir = os.environ.get("AZUREML_MODEL_DIR", "")
    if azure_model_dir:
        candidate = Path(azure_model_dir) / "best.pt"
        if candidate.exists():
            model_path = str(candidate)

    logger.info(f"[init] Cargando modelo desde: {model_path}")
    _engine = BatchInference(model_path=model_path)
    _receiver = BatchReceiver()
    _processor = PostProcessor()
    logger.info("[init] Modelo listo.")


def run(mini_batch: list[str]) -> list[str]:
    """Ejecuta inferencia sobre un mini-lote de rutas de archivo.

    Args:
        mini_batch: Lista de rutas absolutas a imágenes (proporcionadas por Azure ML).

    Returns:
        Lista de cadenas JSON, una por imagen, con las detecciones.
    """
    assert _engine is not None, "init() no se llamó antes de run()."
    assert _receiver is not None
    assert _processor is not None

    results: list[str] = []

    # Preparar el lote a partir de las rutas
    image_files: list[tuple[str, bytes]] = []
    for path_str in mini_batch:
        p = Path(path_str)
        if not p.exists():
            logger.warning(f"Archivo no encontrado: {path_str}")
            continue
        try:
            image_files.append((p.name, p.read_bytes()))
        except Exception as exc:  # noqa: BLE001
            logger.error(f"Error leyendo {path_str}: {exc}")
            continue

    if not image_files:
        return []

    # Etapa 1: recepción y resize
    batch = _receiver.receive(image_files)

    # Etapa 2: inferencia
    batch_result = _engine.run(batch)

    # Etapa 3: post-procesado
    annotated = _processor.process(batch, batch_result)

    # Serializar cada resultado como JSON
    annotated_map = {ai.filename: ai for ai in annotated}
    for ir in batch_result.image_results:
        ai = annotated_map.get(ir.filename)
        record: dict[str, Any] = {
            "filename": ir.filename,
            "has_defects": ai.has_defects if ai else False,
            "no_defect_notification": (
                ai.no_defect_notification if ai else "✅ PCB sin defectos"
            ),
            "detections_count": ai.detections_count if ai else 0,
            "inference_time_ms": ir.inference_time_ms,
            "error": ir.error,
            "detections": [
                {
                    "class_name": d.class_name,
                    "class_id": d.class_id,
                    "confidence": d.confidence,
                    "bbox": d.bbox,
                    "mask_points": d.mask_points,
                }
                for d in ir.detections
            ],
        }
        results.append(json.dumps(record, ensure_ascii=False))
        logger.info(
            f"[run] {ir.filename}: {len(ir.detections)} detección(es)",
            extra={"image": ir.filename, "stage": "score"},
        )

    return results
