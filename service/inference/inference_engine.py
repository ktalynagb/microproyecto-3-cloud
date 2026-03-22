"""
Modulo de inferencia central para el modelo Ateeqq/ai-vs-human-image-detector.

Funcion reutilizable que ejecuta el modelo sobre una imagen y retorna
la prediccion (clase ganadora), las probabilidades por clase, los tiempos
de ejecucion en milisegundos y el estado del proceso (ok/error), en formato
serializable (float/str), lista para ser usada desde gRPC o GUI.

La funcion NUNCA lanza excepciones al caller: cualquier error se captura
internamente y se retorna como respuesta estandarizada con status="error",
permitiendo el procesamiento continuo de lotes de imagenes.

Uso:
    from service.inference.inference_engine import run_inference
    result = run_inference(image, model, processor)
    if result["status"] == "ok":
        print(result["label"])
    else:
        print(result["error"]["message"])

Comando Make asociado:
    make test-inference  ->  Ejecuta los tests unitarios del modulo

Unidades de tiempo:
    Todos los tiempos se expresan en milisegundos (ms) como float
    redondeado a 3 decimales. Se usa time.perf_counter() para maxima
    precision en medicion de intervalos cortos.

Codigos de error estandarizados:
    INVALID_IMAGE   -> TypeError o ValueError en preprocesamiento
    INFERENCE_ERROR -> RuntimeError durante la inferencia del modelo
    UNKNOWN_ERROR   -> Cualquier otra excepcion inesperada
"""

import logging
import time
from typing import Union

import torch
from PIL import Image

from .preprocessing import preprocess_image

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Respuesta de error estandarizada (helper interno)
# ---------------------------------------------------------------------------

_EMPTY_TIMING = {
    "preprocessing_ms": 0.0,
    "inference_ms": 0.0,
    "total_ms": 0.0,
}


def _error_response(
    code: str,
    message: str,
    preprocessing_ms: float = 0.0,
) -> dict:
    """Construye una respuesta de error estandarizada y serializable."""
    timing = {
        "preprocessing_ms": preprocessing_ms,
        "inference_ms": 0.0,
        "total_ms": preprocessing_ms,
    }
    return {
        "status": "error",
        "label": None,
        "label_id": None,
        "scores": {},
        "timing": timing,
        "error": {
            "code": code,
            "message": message,
        },
    }


# ---------------------------------------------------------------------------
# Funcion principal
# ---------------------------------------------------------------------------

def run_inference(
    image: Union[Image.Image, bytes],
    model,
    processor,
) -> dict:
    """Ejecuta inferencia sobre una imagen usando el modelo ViT.

    Preprocesa la imagen, ejecuta el modelo en modo evaluacion, calcula
    softmax sobre los logits y retorna la prediccion con sus probabilidades
    por clase, tiempos de ejecucion y estado del proceso.

    Esta funcion NUNCA propaga excepciones al caller. Cualquier error se
    captura y se retorna como un dict con status="error", permitiendo el
    procesamiento continuo de lotes.

    Args:
        image: Imagen fuente. Puede ser:
            - PIL.Image.Image: usada directamente.
            - bytes: decodificada internamente antes del preprocesamiento.
        model: Modelo de Hugging Face ya cargado
            (AutoModelForImageClassification o compatible).
            Debe tener model.config.id2label.
        processor: Instancia de AutoImageProcessor (o compatible) ya cargada.

    Returns:
        Dict con la siguiente estructura en caso exitoso::

            {
                "status": "ok",
                "label": "AI",
                "label_id": 0,
                "scores": {"AI": 0.9741, "Real": 0.0259},
                "timing": {
                    "preprocessing_ms": 12.345,
                    "inference_ms": 45.678,
                    "total_ms": 58.023
                },
                "error": None
            }

        Dict con la siguiente estructura en caso de error::

            {
                "status": "error",
                "label": None,
                "label_id": None,
                "scores": {},
                "timing": {
                    "preprocessing_ms": 1.234,
                    "inference_ms": 0.0,
                    "total_ms": 1.234
                },
                "error": {
                    "code": "INVALID_IMAGE",
                    "message": "Descripcion del error."
                }
            }

    Example:
        >>> from PIL import Image
        >>> from transformers import (
        ...     AutoImageProcessor,
        ...     AutoModelForImageClassification,
        ... )
        >>> processor = AutoImageProcessor.from_pretrained(
        ...     "dima806/ai_vs_real_image_detection"
        ... )
        >>> model = AutoModelForImageClassification.from_pretrained(
        ...     "dima806/ai_vs_real_image_detection"
        ... )
        >>> img = Image.open("photo.jpg")
        >>> result = run_inference(img, model, processor)
        >>> if result["status"] == "ok":
        ...     print(result["label"])
        ...     print(result["timing"]["total_ms"])
        ... else:
        ...     print(result["error"]["code"])
    """
    preprocessing_ms = 0.0

    # 1. Preprocesar imagen -> inputs dict con pixel_values
    # (con medicion de tiempo)
    try:
        t0_preprocess = time.perf_counter()
        inputs = preprocess_image(image, processor)
        t1_preprocess = time.perf_counter()
        preprocessing_ms = round((t1_preprocess - t0_preprocess) * 1000, 3)
    except (TypeError, ValueError) as exc:
        logger.warning("Error de imagen invalida: %s", exc)
        return _error_response(
            code="INVALID_IMAGE",
            message=str(exc),
            preprocessing_ms=preprocessing_ms,
        )
    except Exception as exc:
        logger.error("Error inesperado en preprocesamiento: %s", exc)
        return _error_response(
            code="UNKNOWN_ERROR",
            message=f"Error inesperado en preprocesamiento: {exc}",
            preprocessing_ms=preprocessing_ms,
        )

    # 2. Inferencia en modo evaluacion sin gradientes (con medicion de tiempo)
    try:
        model.eval()
        t0_inference = time.perf_counter()
        with torch.no_grad():
            outputs = model(**inputs)
        t1_inference = time.perf_counter()
        inference_ms = round((t1_inference - t0_inference) * 1000, 3)
    except Exception as exc:
        logger.error("Error durante la inferencia: %s", exc)
        return _error_response(
            code="INFERENCE_ERROR",
            message=f"Error durante la inferencia del modelo: {exc}",
            preprocessing_ms=preprocessing_ms,
        )

    # 3. Validar que el modelo retorno logits
    if not hasattr(outputs, "logits"):
        logger.error("El modelo no retorno logits.")
        return _error_response(
            code="INFERENCE_ERROR",
            message=(
                "El modelo no retorno 'logits'. "
                "Verifica que sea un modelo de clasificacion de imagenes."
            ),
            preprocessing_ms=preprocessing_ms,
        )

    logits = outputs.logits  # shape: (1, num_classes)

    # 4. Clase predicha (argmax)
    label_id = int(torch.argmax(logits, dim=-1).item())

    # 5. Probabilidades por clase (softmax -> float)
    probs = torch.softmax(logits, dim=-1).squeeze(0)  # shape: (num_classes,)

    # 6. Mapear id2label desde la configuracion del modelo
    id2label = model.config.id2label  # {0: "AI", 1: "Real"} o similar

    scores = {
        id2label[i]: round(float(probs[i]), 6)
        for i in range(len(probs))
    }

    label = id2label[label_id]

    # 7. Calcular total_ms como suma explicita de ambos tiempos
    total_ms = round(preprocessing_ms + inference_ms, 3)

    result = {
        "status": "ok",
        "label": label,
        "label_id": label_id,
        "scores": scores,
        "timing": {
            "preprocessing_ms": preprocessing_ms,
            "inference_ms": inference_ms,
            "total_ms": total_ms,
        },
        "error": None,
    }

    logger.debug(
        "Inferencia exitosa. label=%s, label_id=%d, scores=%s, "
        "preprocessing_ms=%.3f, inference_ms=%.3f, total_ms=%.3f",
        label,
        label_id,
        scores,
        preprocessing_ms,
        inference_ms,
        total_ms,
    )

    return result
