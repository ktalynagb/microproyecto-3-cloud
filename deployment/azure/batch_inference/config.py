"""Configuración centralizada para el pipeline de batch inference PCB.

Todas las credenciales se leen desde variables de entorno; nunca se hardcodean.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class AppConfig:
    # ── Modelo ─────────────────────────────────────────────────────────────
    model_path: str = field(
        default_factory=lambda: os.environ.get("PCB_MODEL_PATH", "best.pt")
    )
    conf_threshold: float = field(
        default_factory=lambda: float(os.environ.get("PCB_CONF_THRESHOLD", "0.25"))
    )
    iou_threshold: float = field(
        default_factory=lambda: float(os.environ.get("PCB_IOU_THRESHOLD", "0.45"))
    )

    # ── Imágenes ───────────────────────────────────────────────────────────
    image_size: int = 640
    max_batch_size: int = 10
    allowed_formats: tuple[str, ...] = (".jpg", ".jpeg", ".png")

    # ── Azure Blob Storage ─────────────────────────────────────────────────
    azure_storage_account: str = field(
        default_factory=lambda: os.environ.get("AZURE_STORAGE_ACCOUNT", "")
    )
    azure_storage_key: str = field(
        default_factory=lambda: os.environ.get("AZURE_STORAGE_KEY", "")
    )
    azure_container_name: str = field(
        default_factory=lambda: os.environ.get("AZURE_CONTAINER_NAME", "pcb-results")
    )
    blob_ttl_hours: int = 24

    # ── Rutas locales ──────────────────────────────────────────────────────
    temp_dir: Path = field(default_factory=lambda: Path("/tmp/pcb_batches"))

    # ── Clases del modelo ──────────────────────────────────────────────────
    class_names: list[str] = field(
        default_factory=lambda: [
            "dry_joint",
            "incorrect_installation",
            "pcb_damage",
            "short_circuit",
        ]
    )

    # ── Colores por clase (BGR para OpenCV) ────────────────────────────────
    class_colors: dict[str, tuple[int, int, int]] = field(
        default_factory=lambda: {
            "dry_joint": (0, 0, 255),           # rojo
            "incorrect_installation": (0, 165, 255),  # naranja
            "pcb_damage": (0, 255, 255),         # amarillo
            "short_circuit": (0, 255, 0),        # verde
        }
    )

    # ── Timeout de inferencia (segundos) ──────────────────────────────────
    inference_timeout_s: int = field(
        default_factory=lambda: int(os.environ.get("PCB_INFERENCE_TIMEOUT", "30"))
    )


# Instancia global por defecto
config = AppConfig()
