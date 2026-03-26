"""Configuración de logging estructurado para Azure ML."""

from __future__ import annotations

import logging
import sys


def get_logger(name: str) -> logging.Logger:
    """Retorna un logger configurado con formato estructurado."""
    logger = logging.getLogger(name)

    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%dT%H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    logger.setLevel(logging.INFO)
    return logger
