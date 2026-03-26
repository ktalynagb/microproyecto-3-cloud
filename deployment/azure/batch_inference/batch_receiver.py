"""Recepción y pre-procesamiento del lote de imágenes."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import NamedTuple

import numpy as np
from PIL import Image
import io

from config import config
from logger import get_logger

logger = get_logger("batch_receiver")


class ImageRecord(NamedTuple):
    filename: str
    image_bytes: bytes


@dataclass
class ProcessedImage:
    filename: str
    original_shape: tuple[int, int]  # (height, width)
    array: np.ndarray  # HWC, uint8, RGB


@dataclass
class Batch:
    images: list[ProcessedImage] = field(default_factory=list)


class BatchReceiver:
    """Recibe bytes de imagen, los decodifica y redimensiona."""

    def __init__(self, target_size: int | None = None) -> None:
        self._target_size = target_size or config.image_size

    def receive(self, image_files: list[tuple[str, bytes]]) -> Batch:
        """
        Parámetros
        ----------
        image_files : list of (filename, bytes)

        Retorna
        -------
        Batch con imágenes decodificadas y redimensionadas.
        """
        batch = Batch()
        for filename, raw_bytes in image_files:
            try:
                img = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
                original_shape = (img.height, img.width)
                # Redimensionar manteniendo aspecto
                img = img.resize(
                    (self._target_size, self._target_size), Image.LANCZOS
                )
                arr = np.asarray(img, dtype=np.uint8)
                batch.images.append(
                    ProcessedImage(
                        filename=filename,
                        original_shape=original_shape,
                        array=arr,
                    )
                )
                logger.info(
                    f"[receiver] {filename}: {original_shape} → "
                    f"({self._target_size}, {self._target_size})"
                )
            except Exception as exc:
                logger.error(f"[receiver] Error procesando {filename}: {exc}")
                # Crear imagen negra como fallback
                arr = np.zeros(
                    (self._target_size, self._target_size, 3), dtype=np.uint8
                )
                batch.images.append(
                    ProcessedImage(
                        filename=filename,
                        original_shape=(self._target_size, self._target_size),
                        array=arr,
                    )
                )
        return batch
