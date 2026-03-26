"""Script de scoring para el Batch Endpoint de Azure ML.

Implementa las dos funciones requeridas por Azure ML:
  - ``init()``: llamada una sola vez al arrancar el worker.
  - ``run(mini_batch)``: llamada por cada mini-lote de imágenes.

IMPORTANTE: No se usa ``signal`` ya que Azure ML ejecuta este script en
worker threads, donde ``signal`` no está disponible. El timeout de
inferencia se gestiona con ``threading.Timer`` (ver ``inference_engine.py``).
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from config import config
from inference_engine import BatchInference
from batch_receiver import BatchReceiver
from logger import get_logger
from post_processor import PostProcessor

logger = get_logger("score")

_engine: BatchInference | None = None
_receiver: BatchReceiver | None = None
_processor: PostProcessor | None = None


def init() -> None:
    """Inicializa el modelo (llamada única al arrancar el worker)."""
    global _engine, _receiver, _processor

    model_path = os.environ.get("PCB_MODEL_PATH", config.model_path)

    # Azure ML inyecta la ruta del modelo registrado en AZUREML_MODEL_DIR
    azure_model_dir = os.environ.get("AZUREML_MODEL_DIR", "")
    if azure_model_dir:
        candidate = Path(azure_model_dir) / "best.pt"
        if candidate.exists():
            model_path = str(candidate)

    logger.info(f"[init] Cargando modelo desde: {model_path}")

    try:
        _engine = BatchInference(model_path=model_path)
        _receiver = BatchReceiver()
        _processor = PostProcessor()
        logger.info("[init] ✅ Modelo listo.")
    except Exception as exc:
        logger.error(f"[init] ❌ Error cargando modelo: {exc}", exc_info=True)
        raise


def run(mini_batch: list[str]) -> list[str]:
    """Ejecuta inferencia sobre un mini-lote de rutas de imágenes.

    Parámetros
    ----------
    mini_batch : list[str]
        Rutas absolutas a los archivos de imagen del mini-lote.

    Retorna
    -------
    list[str]
        Una línea JSON por imagen con los resultados.
    """
    assert _engine is not None, "init() no fue llamado antes de run()."
    assert _receiver is not None
    assert _processor is not None

    results: list[str] = []

    # --- Leer archivos ---
    image_files: list[tuple[str, bytes]] = []
    for path_str in mini_batch:
        p = Path(path_str)
        if not p.exists():
            logger.warning(f"[run] Archivo no encontrado: {path_str}")
            results.append(
                json.dumps(
                    {
                        "filename": p.name,
                        "has_defects": False,
                        "detections_count": 0,
                        "detections": [],
                        "inference_time_ms": 0,
                        "error": "Archivo no encontrado",
                    },
                    ensure_ascii=False,
                )
            )
            continue

        try:
            image_files.append((p.name, p.read_bytes()))
        except Exception as exc:
            logger.error(f"[run] Error leyendo {path_str}: {exc}")
            results.append(
                json.dumps(
                    {
                        "filename": p.name,
                        "has_defects": False,
                        "detections_count": 0,
                        "detections": [],
                        "inference_time_ms": 0,
                        "error": str(exc),
                    },
                    ensure_ascii=False,
                )
            )

    if not image_files:
        return results

    try:
        # Etapa 1: recepción y pre-procesamiento
        batch = _receiver.receive(image_files)

        # Etapa 2: inferencia (timeout via threading, no signal)
        batch_result = _engine.run(batch)

        # Etapa 3: post-procesado y anotación
        annotated = _processor.process(batch, batch_result)

        # Serializar resultados
        annotated_map = {ai.filename: ai for ai in annotated}
        for ir in batch_result.image_results:
            ai = annotated_map.get(ir.filename)
            record: dict[str, Any] = {
                "filename": ir.filename,
                "has_defects": ai.has_defects if ai else False,
                "no_defect_notification": (
                    ai.no_defect_notification
                    if ai
                    else "✅ PCB sin defectos"
                ),
                "detections_count": ai.detections_count if ai else 0,
                "inference_time_ms": round(ir.inference_time_ms, 2),
                "error": ir.error,
                "detections": [
                    {
                        "class_name": d.class_name,
                        "class_id": d.class_id,
                        "confidence": round(d.confidence, 4),
                        "bbox": [round(v, 6) for v in d.bbox],
                        "mask_points": [[round(v, 6) for v in p] for p in d.mask_points],
                    }
                    for d in ir.detections
                ],
            }
            results.append(json.dumps(record, ensure_ascii=False))
            logger.info(
                f"[run] {ir.filename}: {len(ir.detections)} detección(es) "
                f"en {ir.inference_time_ms:.1f}ms"
            )

    except Exception as exc:
        logger.error(f"[run] Error crítico en el batch: {exc}", exc_info=True)
        for image_name, _ in image_files:
            results.append(
                json.dumps(
                    {
                        "filename": image_name,
                        "has_defects": False,
                        "detections_count": 0,
                        "detections": [],
                        "inference_time_ms": 0,
                        "error": str(exc),
                    },
                    ensure_ascii=False,
                )
            )

    return results
