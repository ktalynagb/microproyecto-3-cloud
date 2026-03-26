"""ETAPA 3: Post-procesado y Visualización (post_processor).

Superpone máscaras de segmentación / contornos y bounding boxes sobre las
imágenes originales. Si ningún defecto supera el umbral genera la
notificación '✅ PCB sin defectos'.

Uso como módulo:
    from post_processor import PostProcessor
    processor = PostProcessor()
    annotated = processor.process(batch, batch_result)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import cv2
import numpy as np

from inference_engine import BatchResult, Detection, ImageResult
from batch_receiver import Batch
from config import config
from logger import get_logger
from utils import draw_bbox, image_to_bytes, load_image_from_bytes, overlay_mask

if TYPE_CHECKING:
    pass

logger = get_logger("post_processor")


@dataclass
class AnnotatedImage:
    """Imagen anotada lista para exportar."""

    filename: str
    annotated_bytes: bytes  # JPEG
    has_defects: bool
    detections_count: int
    no_defect_notification: str | None = None


class PostProcessor:
    """Genera imágenes anotadas a partir de los resultados de inferencia."""

    def __init__(
        self,
        conf_threshold: float = config.conf_threshold,
        class_colors: dict[str, tuple[int, int, int]] | None = None,
    ) -> None:
        self.conf_threshold = conf_threshold
        self.colors = class_colors or config.class_colors

    def _get_color(self, class_name: str) -> tuple[int, int, int]:
        return self.colors.get(class_name, (255, 0, 255))

    def _annotate_image(
        self,
        img: np.ndarray,
        image_result: ImageResult,
        w_orig: int,
        h_orig: int,
    ) -> tuple[np.ndarray, bool]:
        """Dibuja detecciones sobre la imagen y retorna (imagen_anotada, has_defects)."""
        annotated = img.copy()
        defects_above_threshold = [
            d for d in image_result.detections if d.confidence >= self.conf_threshold
        ]

        if not defects_above_threshold:
            # Sin defectos: añadir texto de notificación
            msg = "PCB sin defectos"
            (tw, th), _ = cv2.getTextSize(msg, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)
            x0 = max(0, (annotated.shape[1] - tw) // 2)
            y0 = max(th + 10, 50)
            cv2.rectangle(annotated, (x0 - 5, y0 - th - 10), (x0 + tw + 5, y0 + 5), (0, 200, 0), -1)
            cv2.putText(
                annotated,
                msg,
                (x0, y0),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            return annotated, False

        for det in defects_above_threshold:
            color = self._get_color(det.class_name)

            # Escalar bbox de coordenadas normalizadas a píxeles
            x1 = int(det.bbox[0] * annotated.shape[1])
            y1 = int(det.bbox[1] * annotated.shape[0])
            x2 = int(det.bbox[2] * annotated.shape[1])
            y2 = int(det.bbox[3] * annotated.shape[0])

            # Superponer máscara si está disponible
            if det.mask_points:
                pts = np.array(
                    [
                        [int(p[0] * annotated.shape[1]), int(p[1] * annotated.shape[0])]
                        for p in det.mask_points
                    ],
                    dtype=np.int32,
                )
                mask = np.zeros(annotated.shape[:2], dtype=np.uint8)
                cv2.fillPoly(mask, [pts], 255)
                annotated = overlay_mask(annotated, mask, color=color)
                # Dibujar contorno del polígono
                cv2.polylines(annotated, [pts], isClosed=True, color=color, thickness=2)

            label = f"{det.class_name} {det.confidence:.2f}"
            annotated = draw_bbox(annotated, x1, y1, x2, y2, label, color)

        return annotated, True

    def process(self, batch: Batch, batch_result: BatchResult) -> list[AnnotatedImage]:
        """Genera imágenes anotadas para todos los resultados del lote."""
        annotated_images: list[AnnotatedImage] = []

        img_map = {bi.filename: bi for bi in batch.images}

        for image_result in batch_result.image_results:
            bi = img_map.get(image_result.filename)
            if bi is None:
                logger.warning(f"Imagen no encontrada en el lote: {image_result.filename}")
                continue

            if image_result.error:
                logger.warning(
                    f"Saltando imagen con error: {image_result.filename}: {image_result.error}"
                )
                annotated_images.append(
                    AnnotatedImage(
                        filename=image_result.filename,
                        annotated_bytes=bi.original_bytes,
                        has_defects=False,
                        detections_count=0,
                        no_defect_notification=f"Error en inferencia: {image_result.error}",
                    )
                )
                continue

            img = load_image_from_bytes(bi.original_bytes)
            h_orig, w_orig = img.shape[:2]

            annotated_img, has_defects = self._annotate_image(
                img, image_result, w_orig, h_orig
            )
            annotated_bytes = image_to_bytes(annotated_img, ext=".jpg")

            notification = None if has_defects else "✅ PCB sin defectos"

            annotated_images.append(
                AnnotatedImage(
                    filename=image_result.filename,
                    annotated_bytes=annotated_bytes,
                    has_defects=has_defects,
                    detections_count=len(
                        [
                            d
                            for d in image_result.detections
                            if d.confidence >= self.conf_threshold
                        ]
                    ),
                    no_defect_notification=notification,
                )
            )
            logger.info(
                f"Anotada {image_result.filename}: defectos={has_defects}",
                extra={
                    "batch_id": batch_result.batch_id,
                    "image": image_result.filename,
                    "stage": "post_process",
                },
            )

        return annotated_images
