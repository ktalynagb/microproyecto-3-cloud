"""ETAPA 2: Inferencia por lote (inference_engine).

Ejecuta el modelo YOLOv8n registrado sobre cada imagen del lote y retorna
detecciones con clase, bounding box, máscara/contorno y confidence.

Uso como módulo:
    from inference_engine import BatchInference
    engine = BatchInference(model_path="best.pt")
    results = engine.run(batch)
"""

from __future__ import annotations

import signal
import time
import types
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Generator

import numpy as np

from batch_receiver import Batch
from config import config
from logger import get_logger

logger = get_logger("batch_inference")


@dataclass
class Detection:
    """Una detección individual dentro de una imagen."""

    class_name: str
    class_id: int
    confidence: float
    bbox: list[float]  # [x1, y1, x2, y2] normalizados 0-1
    mask_points: list[list[float]] | None = None  # polígono normalizado o None


@dataclass
class ImageResult:
    """Resultado de inferencia para una imagen."""

    filename: str
    detections: list[Detection]
    inference_time_ms: float
    has_defects: bool = field(init=False)
    error: str | None = None

    def __post_init__(self) -> None:
        self.has_defects = len(self.detections) > 0


@dataclass
class BatchResult:
    """Resultado completo de un lote."""

    batch_id: str
    image_results: list[ImageResult]
    total_time_ms: float


@contextmanager
def _timeout(seconds: int) -> Generator[None, None, None]:
    """Context manager que lanza TimeoutError tras `seconds` segundos (Unix)."""
    def _handler(signum: int, frame: types.FrameType | None) -> None:
        raise TimeoutError(f"Inferencia excedió el tiempo límite de {seconds}s.")

    old = signal.signal(signal.SIGALRM, _handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old)


class BatchInference:
    """Ejecuta YOLOv8n sobre un Batch de imágenes."""

    def __init__(
        self,
        model_path: str = config.model_path,
        conf: float = config.conf_threshold,
        iou: float = config.iou_threshold,
        timeout_s: int = config.inference_timeout_s,
    ) -> None:
        from ultralytics import YOLO

        self.conf = conf
        self.iou = iou
        self.timeout_s = timeout_s
        logger.info(f"Cargando modelo desde: {model_path}")
        self.model = YOLO(str(model_path))
        self._names: dict[int, str] = self.model.names  # type: ignore[assignment]

    def _infer_single(self, img: np.ndarray, filename: str) -> ImageResult:
        """Ejecuta inferencia sobre una imagen numpy BGR 640×640."""
        t0 = time.perf_counter()
        try:
            with _timeout(self.timeout_s):
                results = self.model(
                    img,
                    conf=self.conf,
                    iou=self.iou,
                    verbose=False,
                )
        except TimeoutError as exc:
            logger.warning(str(exc), extra={"image": filename})
            elapsed = (time.perf_counter() - t0) * 1000
            return ImageResult(
                filename=filename,
                detections=[],
                inference_time_ms=round(elapsed, 2),
                error=str(exc),
            )
        except Exception as exc:  # noqa: BLE001
            logger.error(f"Error en inferencia: {exc}", extra={"image": filename})
            elapsed = (time.perf_counter() - t0) * 1000
            return ImageResult(
                filename=filename,
                detections=[],
                inference_time_ms=round(elapsed, 2),
                error=str(exc),
            )

        elapsed_ms = (time.perf_counter() - t0) * 1000
        detections: list[Detection] = []
        r = results[0]

        h, w = img.shape[:2]

        if r.boxes is not None:
            for i, box in enumerate(r.boxes):
                cls_id = int(box.cls.item())
                cls_name = self._names.get(cls_id, str(cls_id))
                conf_val = float(box.conf.item())
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                bbox_norm = [x1 / w, y1 / h, x2 / w, y2 / h]

                # Máscara de segmentación si está disponible
                mask_pts: list[list[float]] | None = None
                if r.masks is not None and i < len(r.masks):
                    xy = r.masks.xy[i]  # array (N, 2) en píxeles
                    mask_pts = [[float(p[0]) / w, float(p[1]) / h] for p in xy]

                detections.append(
                    Detection(
                        class_name=cls_name,
                        class_id=cls_id,
                        confidence=round(conf_val, 4),
                        bbox=bbox_norm,
                        mask_points=mask_pts,
                    )
                )

        logger.info(
            f"{filename}: {len(detections)} detección(es) en {elapsed_ms:.1f}ms",
            extra={"image": filename},
        )
        return ImageResult(
            filename=filename,
            detections=detections,
            inference_time_ms=round(elapsed_ms, 2),
        )

    def run(self, batch: Batch) -> BatchResult:
        """Ejecuta inferencia sobre todas las imágenes del lote."""
        t_start = time.perf_counter()
        image_results: list[ImageResult] = []

        for bi in batch.images:
            result = self._infer_single(bi.resized, bi.filename)
            image_results.append(result)

        total_ms = (time.perf_counter() - t_start) * 1000
        logger.info(
            f"Lote {batch.batch_id}: {len(image_results)} imágenes en {total_ms:.1f}ms total",
            extra={"batch_id": batch.batch_id, "stage": "inference"},
        )
        return BatchResult(
            batch_id=batch.batch_id,
            image_results=image_results,
            total_time_ms=round(total_ms, 2),
        )
