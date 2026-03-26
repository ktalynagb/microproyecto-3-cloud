"""ETAPA 1: Recepción del lote (batch_receiver).

Recibe hasta 10 imágenes PCB, aplica resize 640×640, valida el formato
y retorna el ID del lote con estado 'En procesamiento'.

Uso como módulo:
    from batch_receiver import BatchReceiver
    receiver = BatchReceiver()
    batch = receiver.receive(image_files)  # lista de (nombre, bytes)
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Union

import numpy as np

from config import config
from logger import get_logger
from utils import (
    IMAGE_SIZE,
    load_image_from_bytes,
    resize_image,
    validate_image_format,
)

logger = get_logger("batch_receiver")


@dataclass
class BatchImage:
    """Representa una imagen dentro de un lote."""

    filename: str
    original_bytes: bytes
    resized: np.ndarray  # uint8 BGR, 640×640
    width_orig: int
    height_orig: int


@dataclass
class Batch:
    """Lote de imágenes listo para inferencia."""

    batch_id: str
    images: list[BatchImage]
    received_at: str = field(
        default_factory=lambda: datetime.now(tz=timezone.utc).isoformat()
    )
    status: str = "En procesamiento"

    @property
    def size(self) -> int:
        return len(self.images)


class BatchReceiver:
    """Valida, redimensiona y empaqueta imágenes en un lote."""

    def __init__(self, max_size: int = config.max_batch_size, image_size: int = IMAGE_SIZE):
        self.max_size = max_size
        self.image_size = image_size

    def receive(
        self,
        files: list[tuple[str, bytes]],
    ) -> Batch:
        """Procesa una lista de (nombre, bytes) y retorna un Batch.

        Args:
            files: Lista de tuplas ``(filename, raw_bytes)``.

        Returns:
            Batch con ID único y estado 'En procesamiento'.

        Raises:
            ValueError: si se supera el límite de imágenes o el formato es inválido.
        """
        if len(files) > self.max_size:
            raise ValueError(
                f"Se recibieron {len(files)} imágenes; el máximo es {self.max_size}."
            )
        if not files:
            raise ValueError("El lote está vacío.")

        batch_id = str(uuid.uuid4())
        images: list[BatchImage] = []

        for filename, raw_bytes in files:
            if not validate_image_format(filename):
                raise ValueError(
                    f"Formato inválido para '{filename}'. "
                    f"Formatos aceptados: {config.allowed_formats}"
                )
            img = load_image_from_bytes(raw_bytes)
            h_orig, w_orig = img.shape[:2]
            resized = resize_image(img, self.image_size)
            images.append(
                BatchImage(
                    filename=filename,
                    original_bytes=raw_bytes,
                    resized=resized,
                    width_orig=w_orig,
                    height_orig=h_orig,
                )
            )
            logger.info(
                "Imagen recibida",
                extra={"batch_id": batch_id, "image": filename, "stage": "receive"},
            )

        batch = Batch(batch_id=batch_id, images=images)
        logger.info(
            f"Lote {batch_id} creado con {batch.size} imagen(es).",
            extra={"batch_id": batch_id, "stage": "receive"},
        )
        return batch
