"""Funciones comunes para el pipeline de batch inference PCB.

Resize, normalización, validación de formatos y helpers de imagen.
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import Union

import cv2
import numpy as np


VALID_FORMATS = {".jpg", ".jpeg", ".png"}
IMAGE_SIZE = 640


def validate_image_format(filename: str) -> bool:
    """Verifica que el archivo tenga un formato de imagen válido."""
    return Path(filename).suffix.lower() in VALID_FORMATS


def resize_image(img: np.ndarray, size: int = IMAGE_SIZE) -> np.ndarray:
    """Redimensiona una imagen OpenCV a size×size manteniendo el aspecto."""
    return cv2.resize(img, (size, size), interpolation=cv2.INTER_LINEAR)


def normalize_image(img: np.ndarray) -> np.ndarray:
    """Normaliza una imagen BGR a float32 en [0, 1]."""
    return img.astype(np.float32) / 255.0


def load_image_from_bytes(data: bytes) -> np.ndarray:
    """Carga una imagen desde bytes (contenido de un archivo)."""
    arr = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("No se pudo decodificar la imagen desde los bytes proporcionados.")
    return img


def load_image_from_path(path: Union[str, Path]) -> np.ndarray:
    """Carga una imagen desde disco."""
    img = cv2.imread(str(path))
    if img is None:
        raise FileNotFoundError(f"No se pudo cargar la imagen: {path}")
    return img


def image_to_bytes(img: np.ndarray, ext: str = ".jpg", quality: int = 95) -> bytes:
    """Codifica una imagen OpenCV a bytes."""
    params = [cv2.IMWRITE_JPEG_QUALITY, quality] if ext.lower() in (".jpg", ".jpeg") else []
    success, buf = cv2.imencode(ext, img, params)
    if not success:
        raise RuntimeError("No se pudo codificar la imagen.")
    return buf.tobytes()


def overlay_mask(
    image: np.ndarray,
    mask: np.ndarray,
    color: tuple[int, int, int] = (0, 255, 0),
    alpha: float = 0.4,
) -> np.ndarray:
    """Superpone una máscara binaria sobre la imagen con transparencia."""
    overlay = image.copy()
    overlay[mask > 0] = color
    return cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)


def draw_bbox(
    image: np.ndarray,
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    label: str,
    color: tuple[int, int, int] = (0, 0, 255),
    thickness: int = 2,
) -> np.ndarray:
    """Dibuja un bounding box con etiqueta sobre la imagen."""
    cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    # Fondo para la etiqueta
    cv2.rectangle(image, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
    cv2.putText(
        image,
        label,
        (x1 + 2, y1 - 4),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return image
