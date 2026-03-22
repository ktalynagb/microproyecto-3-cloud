"""Módulo de errores personalizados para el servicio de inferencia."""


class PreprocessError(RuntimeError):
    """Error controlado para fallas de preprocesamiento.

    Se lanza cuando la imagen es inválida, tiene un formato
    no soportado u otro problema durante el preprocesamiento.
    """
