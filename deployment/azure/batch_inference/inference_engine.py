"""Motor de inferencia YOLOv8 para Azure ML Batch.

NOTA IMPORTANTE: Se usa ``threading.Timer`` en lugar de ``signal.alarm``
porque Azure ML Batch ejecuta el scoring en worker threads, y ``signal``
solo funciona en el thread principal del intérprete de Python.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from pathlib import Path
from time import perf_counter
from typing import Any

from config import config
from logger import get_logger

logger = get_logger("inference_engine")


# ---------------------------------------------------------------------------
# Timeout sin signal (compatible con worker threads de Azure ML)
# ---------------------------------------------------------------------------

class _TimeoutContext:
    """Context manager que interrumpe la ejecución tras ``timeout_s`` segundos.

    A diferencia de ``signal.alarm``, usa un daemon thread para el temporizador
    de forma que sea compatible con hilos worker (como los de Azure ML Batch).
    """

    def __init__(self, timeout_s: int) -> None:
        self._timeout_s = timeout_s
        self._timed_out = False
        self._timer: threading.Timer | None = None

    def _on_timeout(self) -> None:
        self._timed_out = True

    @property
    def timed_out(self) -> bool:
        """Indica si el timeout se activó durante la ejecución."""
        return self._timed_out

    def __enter__(self) -> "_TimeoutContext":
        if self._timeout_s > 0:
            self._timer = threading.Timer(self._timeout_s, self._on_timeout)
            self._timer.daemon = True
            self._timer.start()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        if self._timer is not None:
            self._timer.cancel()
        if self._timed_out:
            raise TimeoutError(
                f"Inferencia excedió el tiempo límite de {self._timeout_s}s."
            )


# ---------------------------------------------------------------------------
# Modelos de datos
# ---------------------------------------------------------------------------

@dataclass
class Detection:
    class_name: str
    class_id: int
    confidence: float
    bbox: list[float]          # [x1, y1, x2, y2] normalizadas 0-1
    mask_points: list[list[float]]  # [[x, y], ...] normalizadas 0-1


@dataclass
class ImageResult:
    filename: str
    detections: list[Detection] = field(default_factory=list)
    inference_time_ms: float = 0.0
    error: str | None = None


@dataclass
class BatchResult:
    image_results: list[ImageResult] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Motor principal
# ---------------------------------------------------------------------------

class BatchInference:
    """Envuelve un modelo YOLOv8 de segmentación/detección para inferencia batch."""

    def __init__(self, model_path: str | None = None) -> None:
        self._model_path = model_path or config.model_path
        self._model: Any = None
        self._load_model()

    def _load_model(self) -> None:
        """Carga el modelo desde disco. Llamado una sola vez en ``init()``."""
        try:
            from ultralytics import YOLO  # type: ignore[import]

            self._model = YOLO(self._model_path)
            logger.info(f"[engine] Modelo cargado desde: {self._model_path}")
        except Exception as exc:
            logger.error(f"[engine] Error cargando modelo: {exc}")
            raise

    # ------------------------------------------------------------------

    def run(self, batch: Any) -> BatchResult:
        """Ejecuta inferencia sobre un ``Batch`` de imágenes."""
        from batch_receiver import Batch, ProcessedImage  # local import

        assert isinstance(batch, Batch), "Se esperaba un objeto Batch"
        result = BatchResult()

        for processed_img in batch.images:
            image_result = self._run_single(processed_img)
            result.image_results.append(image_result)

        return result

    def _run_single(self, processed_img: Any) -> ImageResult:
        """Inferencia sobre una imagen, con timeout seguro para threads."""
        image_result = ImageResult(filename=processed_img.filename)
        ctx = _TimeoutContext(timeout_s=config.inference_timeout_s)

        t0 = perf_counter()
        try:
            with ctx:
                predictions = self._model.predict(
                    source=processed_img.array,
                    conf=config.confidence_threshold,
                    iou=config.iou_threshold,
                    imgsz=config.image_size,
                    verbose=False,
                )
                # Verificar timeout después de predict (puede tomar tiempo)
                if ctx.timed_out:
                    raise TimeoutError(
                        f"Inferencia excedió {config.inference_timeout_s}s."
                    )

                image_result.detections = self._parse_predictions(
                    predictions, processed_img
                )
        except TimeoutError as exc:
            logger.warning(f"[engine] Timeout en {processed_img.filename}: {exc}")
            image_result.error = str(exc)
        except Exception as exc:
            logger.error(
                f"[engine] Error en {processed_img.filename}: {exc}",
                exc_info=True,
            )
            image_result.error = str(exc)
        finally:
            image_result.inference_time_ms = (perf_counter() - t0) * 1000

        return image_result

    # ------------------------------------------------------------------

    def _parse_predictions(
        self, predictions: list[Any], processed_img: Any
    ) -> list[Detection]:
        """Extrae detecciones del output de YOLO en formato normalizado."""
        detections: list[Detection] = []
        if not predictions:
            return detections

        h, w = processed_img.original_shape
        result = predictions[0]  # YOLOv8 retorna lista con 1 elemento por imagen

        boxes = result.boxes
        masks = result.masks  # None si es modelo de detección puro

        if boxes is None:
            return detections

        for i, box in enumerate(boxes):
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            x1, y1, x2, y2 = box.xyxy[0].tolist()

            # Normalizar bounding box
            bbox = [x1 / w, y1 / h, x2 / w, y2 / h]

            # Extraer máscara si existe
            mask_pts: list[list[float]] = []
            if masks is not None and i < len(masks.xy):
                for pt in masks.xy[i]:
                    mask_pts.append([float(pt[0]) / w, float(pt[1]) / h])

            class_name = (
                config.class_names[cls_id]
                if cls_id < len(config.class_names)
                else str(cls_id)
            )

            detections.append(
                Detection(
                    class_name=class_name,
                    class_id=cls_id,
                    confidence=conf,
                    bbox=bbox,
                    mask_points=mask_pts,
                )
            )

        return detections
