"""Post-procesamiento: anotación de imágenes y consolidación de resultados."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from config import config
from logger import get_logger

logger = get_logger("post_processor")

# Paleta de colores por clase (BGR para OpenCV)
_CLASS_COLORS: list[tuple[int, int, int]] = [
    (0, 0, 255),    # Missing_hole – rojo
    (0, 165, 255),  # Incorrect_installation – naranja
    (0, 255, 255),  # Copper_exposed – amarillo
    (255, 0, 0),    # Short_circuit – azul
    (0, 255, 0),    # Open_circuit – verde
    (255, 0, 255),  # Spur – magenta
]


@dataclass
class AnnotatedImage:
    filename: str
    has_defects: bool
    detections_count: int
    no_defect_notification: str
    annotated_bytes: bytes = b""  # JPEG anotado


class PostProcessor:
    """Genera imágenes anotadas y metadatos de resumen."""

    def process(self, batch: Any, batch_result: Any) -> list[AnnotatedImage]:
        """
        Genera ``AnnotatedImage`` para cada imagen del batch.

        Parámetros
        ----------
        batch : Batch
        batch_result : BatchResult
        """
        from batch_receiver import Batch  # local import
        from inference_engine import BatchResult  # local import

        assert isinstance(batch, Batch)
        assert isinstance(batch_result, BatchResult)

        img_map = {img.filename: img for img in batch.images}
        annotated: list[AnnotatedImage] = []

        for ir in batch_result.image_results:
            has_defects = len(ir.detections) > 0 and not ir.error
            ai = AnnotatedImage(
                filename=ir.filename,
                has_defects=has_defects,
                detections_count=len(ir.detections),
                no_defect_notification="✅ PCB sin defectos detectados",
            )

            raw_img = img_map.get(ir.filename)
            if raw_img is not None and has_defects:
                try:
                    ai.annotated_bytes = self._annotate(raw_img.array, ir)
                except Exception as exc:
                    logger.warning(
                        f"[post] Error anotando {ir.filename}: {exc}"
                    )

            annotated.append(ai)

        return annotated

    # ------------------------------------------------------------------

    def _annotate(self, img_array: np.ndarray, image_result: Any) -> bytes:
        """Dibuja bboxes y máscaras sobre la imagen y retorna JPEG bytes."""
        try:
            import cv2  # type: ignore[import]
        except ImportError:
            logger.warning("[post] OpenCV no disponible; sin anotaciones visuales.")
            return b""

        import io
        from PIL import Image

        img = img_array.copy()
        h, w = img.shape[:2]

        for det in image_result.detections:
            color = _CLASS_COLORS[det.class_id % len(_CLASS_COLORS)]

            # Bounding box (desnormalizar)
            x1 = int(det.bbox[0] * w)
            y1 = int(det.bbox[1] * h)
            x2 = int(det.bbox[2] * w)
            y2 = int(det.bbox[3] * h)

            cv2.rectangle(img, (x1, y1), (x2, y2), color, config.annotation_thickness)

            label = f"{det.class_name} {det.confidence:.2f}"
            cv2.putText(
                img,
                label,
                (x1, max(y1 - 5, 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                config.font_scale,
                color,
                1,
                cv2.LINE_AA,
            )

            # Máscara (si existe)
            if det.mask_points:
                pts = np.array(
                    [[int(p[0] * w), int(p[1] * h)] for p in det.mask_points],
                    dtype=np.int32,
                )
                overlay = img.copy()
                cv2.fillPoly(overlay, [pts], color)
                cv2.addWeighted(overlay, 0.3, img, 0.7, 0, img)

        # Convertir RGB → JPEG
        pil_img = Image.fromarray(img)
        buf = io.BytesIO()
        pil_img.save(buf, format="JPEG", quality=90)
        return buf.getvalue()
