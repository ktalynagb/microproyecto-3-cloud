"""Módulo de carga de modelos para el servicio de inferencia.

Proporciona funciones para inicializar los artefactos de inferencia
(modelo y procesador) desde Hugging Face y reportar el estado de
carga a MLflow.

Sin efectos secundarios al importarse: toda la lógica se ejecuta
únicamente cuando se llaman las funciones explícitamente.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class ModelLoadError(RuntimeError):
    """Error controlado al cargar modelo/processor (HF o MLflow)."""


def _wrap(msg: str, exc: Exception) -> ModelLoadError:
    raise ModelLoadError(
        f"{msg} | cause={type(exc).__name__}: {exc}"
    ) from exc


@dataclass(frozen=True)
class InferenceArtifacts:
    model: Any
    processor: Any
    device: str
    source: str
    model_id_or_uri: str


def init_inference_artifacts(
    *,
    hf_model_id: str,
    device: str = "cpu",
    hf_cache_dir: Optional[str] = None,
    hf_revision: Optional[str] = None,
    hf_token: Optional[str] = None,
) -> InferenceArtifacts:
    """Carga modelo + processor en CPU y modo eval.

    Sin side-effects: NO se ejecuta al importar el módulo,
    solo al llamar esta función.

    Nota: Usamos AutoModelForImageClassification para que funcione
    con modelos ViT/otros (por ejemplo
    Ateeqq/ai-vs-human-image-detector es ViT).
    """
    if not hf_model_id or not hf_model_id.strip():
        raise ModelLoadError(
            "hf_model_id está vacío. Define HF_MODEL_ID "
            "(variable de entorno) o pásalo al llamar."
        )

    try:
        import torch
        from transformers import (
            AutoImageProcessor,
            AutoModelForImageClassification,
        )
    except Exception as e:
        raise _wrap(
            "Faltan dependencias (torch/transformers). "
            "Revisa tu entorno.",
            e,
        )

    kwargs: Dict[str, Any] = {}
    if hf_cache_dir:
        kwargs["cache_dir"] = hf_cache_dir
    if hf_revision:
        kwargs["revision"] = hf_revision
    if hf_token:
        # HF token para repos privados o mejores rate limits
        kwargs["token"] = hf_token

    try:
        processor = AutoImageProcessor.from_pretrained(
            hf_model_id, **kwargs
        )
        model = AutoModelForImageClassification.from_pretrained(
            hf_model_id, **kwargs
        )
    except OSError as e:
        raise _wrap(
            f"Fallo al cargar desde Hugging Face '{hf_model_id}'. "
            "Verifica: internet, nombre del modelo, permisos/token, "
            "o cache.",
            e,
        )
    except Exception as e:
        raise _wrap(
            f"Fallo inesperado cargando desde HF '{hf_model_id}'.", e
        )

    try:
        model.to(device)  # cpu por defecto
        model.eval()      # criterio de aceptación
        torch.set_grad_enabled(False)
    except Exception as e:
        raise _wrap(
            "El modelo cargó, pero falló al mover a CPU o eval().", e
        )

    return InferenceArtifacts(
        model=model,
        processor=processor,
        device=device,
        source="hf",
        model_id_or_uri=hf_model_id,
    )


def report_loaded_to_mlflow(*, artifacts: InferenceArtifacts) -> None:
    """Registra tags en MLflow indicando que el servicio cargó el modelo.

    No se llama solo (sin side-effects).
    """
    try:
        import mlflow
    except Exception:
        logger.warning("MLflow no disponible; no se reportará health.")
        return

    try:
        mlflow.set_tag("service.inference_loaded", "true")
        mlflow.set_tag("service.model_source", artifacts.source)
        mlflow.set_tag(
            "service.model_id_or_uri", artifacts.model_id_or_uri
        )
        mlflow.set_tag("service.device", artifacts.device)
    except Exception as e:
        logger.warning("No se pudo reportar a MLflow: %s", e)
