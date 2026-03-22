"""Módulo de inferencia para detección de defectos en PCB.

Expone las funciones de inferencia del motor YOLOv8 utilizadas
por el servidor FastAPI.
"""
from .inference_engine import get_model, run_inference

__all__ = ["get_model", "run_inference"]
