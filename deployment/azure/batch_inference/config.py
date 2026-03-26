"""Configuración global para el pipeline de inferencia en Azure ML."""

from __future__ import annotations

import os
from dataclasses import dataclass, field


@dataclass
class Config:
    # Rutas del modelo
    model_path: str = os.environ.get("PCB_MODEL_PATH", "models/best.pt")

    # Clases del modelo YOLOv8
    class_names: list[str] = field(default_factory=lambda: [
        "Missing_hole",
        "Incorrect_installation",
        "Copper_exposed",
        "Short_circuit",
        "Open_circuit",
        "Spur",
    ])

    # Umbrales de confianza
    confidence_threshold: float = float(os.environ.get("PCB_CONF_THRESHOLD", "0.25"))
    iou_threshold: float = float(os.environ.get("PCB_IOU_THRESHOLD", "0.45"))

    # Tamaño de imagen para inferencia
    image_size: int = int(os.environ.get("PCB_IMAGE_SIZE", "640"))

    # Timeout de inferencia por imagen (segundos)
    inference_timeout_s: int = int(os.environ.get("PCB_INFERENCE_TIMEOUT", "60"))

    # Azure Blob Storage
    storage_account: str = os.environ.get(
        "AZURE_STORAGE_ACCOUNT", "pcbmlworstorage428505ef7"
    )
    storage_container_predictions: str = os.environ.get(
        "AZURE_STORAGE_CONTAINER",
        "azureml-blobstore-fa3e2152-1a09-4e81-acb4-3f701118ca5e",
    )
    storage_connection_string: str = os.environ.get(
        "AZURE_STORAGE_CONNECTION_STRING", ""
    )

    # Azure ML Batch Endpoint
    batch_endpoint_url: str = os.environ.get(
        "BATCH_ENDPOINT_URL",
        "https://pcb-batch-inference.centralus.inference.ml.azure.com/jobs",
    )

    # Anotaciones
    annotation_thickness: int = 2
    font_scale: float = 0.5


config = Config()
