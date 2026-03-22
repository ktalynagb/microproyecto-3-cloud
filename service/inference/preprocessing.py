"""
Modulo de preprocesamiento de imagen para el modelo
dima806/ai_vs_real_image_detection.

Funcion reutilizable que transforma una imagen PIL (o bytes) en tensores
compatibles con el modelo ViT de Hugging Face, lista para inferencia.

Uso:
    from service.inference.preprocessing import preprocess_image
    inputs = preprocess_image(pil_image, processor)

Comando Make asociado:
    make test-preprocessing  ->  Ejecuta los tests unitarios del modulo
"""

import io
import logging
from typing import Union

from PIL import Image

logger = logging.getLogger(__name__)


def preprocess_image(
    image: Union[Image.Image, bytes],
    processor,
) -> dict:
    """Preprocesa una imagen para ser usada como entrada al modelo ViT.

    Convierte una imagen PIL o bytes en un diccionario de tensores PyTorch
    compatibles con el modelo Ateeqq/ai-vs-human-image-detector.

    Args:
        image: Imagen fuente. Puede ser:
            - PIL.Image.Image: usada directamente.
            - bytes: decodificada a PIL antes del preprocesamiento.
        processor: Instancia de AutoImageProcessor (o compatible) de
            Hugging Face, ya cargada con el modelo correspondiente.

    Returns:
        Dict con al menos la clave "pixel_values" (tensor PyTorch),
        listo para pasarse al modelo con **inputs.

    Raises:
        TypeError: Si image no es PIL.Image.Image ni bytes.
        ValueError: Si los bytes no pueden decodificarse como imagen
            valida, o si el processor retorna un resultado
            vacio/inesperado.
        RuntimeError: Si ocurre un error inesperado durante el
            preprocesamiento.

    Example:
        >>> from PIL import Image
        >>> from transformers import AutoImageProcessor
        >>> processor = AutoImageProcessor.from_pretrained(
        ...     "Ateeqq/ai-vs-human-image-detector"
        ... )
        >>> img = Image.new("RGB", (224, 224))
        >>> inputs = preprocess_image(img, processor)
        >>> print(inputs["pixel_values"].shape)  # torch.Size([1, 3, 224, 224])
    """
    # 1. Validar y convertir a PIL
    if isinstance(image, bytes):
        logger.debug("Convirtiendo bytes (%d B) a PIL.Image.", len(image))
        try:
            pil_image = Image.open(io.BytesIO(image))
        except Exception as exc:
            raise ValueError(
                f"No se pudo decodificar los bytes como imagen: {exc}"
            ) from exc
    elif isinstance(image, Image.Image):
        pil_image = image
    else:
        raise TypeError(
            "Se esperaba PIL.Image.Image o bytes, "
            f"se recibio {type(image).__name__}."
        )

    # 2. Asegurar modo RGB
    if pil_image.mode != "RGB":
        logger.debug(
            "Convirtiendo imagen de modo '%s' a 'RGB'.", pil_image.mode
        )
        pil_image = pil_image.convert("RGB")

    # 3. Aplicar processor de Hugging Face
    try:
        inputs = processor(images=pil_image, return_tensors="pt")
    except Exception as exc:
        raise RuntimeError(
            "Error al procesar la imagen con el processor "
            f"de Hugging Face: {exc}"
        ) from exc

    # 4. Validar salida
    if not inputs or "pixel_values" not in inputs:
        raise ValueError(
            "El processor no retorno 'pixel_values'. "
            "Verifica que el processor sea compatible con el modelo."
        )

    logger.debug(
        "Preprocesamiento exitoso. pixel_values shape: %s",
        tuple(inputs["pixel_values"].shape),
    )
    return inputs
