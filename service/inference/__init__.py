"""Módulo de inferencia para ImageAivsReal.

Expone las funciones de preprocesamiento e inferencia de imágenes
utilizadas por el servidor gRPC y el motor central del servicio.
"""
from .preprocessing import preprocess_image
from .inference_engine import run_inference

__all__ = ["preprocess_image", "run_inference"]
