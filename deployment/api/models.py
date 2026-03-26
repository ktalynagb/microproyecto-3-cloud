"""Pydantic models para el backend FastAPI de inferencia PCB.

Define los esquemas de request y response usados por los endpoints:
    POST /api/v1/infer
    GET  /api/v1/jobs/{job_id}
    GET  /api/v1/jobs/{job_id}/results
    GET  /api/v1/health
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


# ── Health ────────────────────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    status: str = "ok"
    version: str = "1.0.0"
    batch_endpoint: str
    blob_storage: str


# ── Infer (submit) ────────────────────────────────────────────────────────────

class InferResponse(BaseModel):
    job_id: str
    status: str = "submitted"
    message: str = "Job enviado al Batch Endpoint"


# ── Job status ────────────────────────────────────────────────────────────────

class JobStatusResponse(BaseModel):
    job_id: str
    status: str                      # submitted | running | completed | failed
    created_at: Optional[str] = None
    updated_at: Optional[str] = None
    message: Optional[str] = None


# ── Detección individual ──────────────────────────────────────────────────────

class DetectionOut(BaseModel):
    """Detección adaptada al formato esperado por el frontend."""

    class_name: str = Field(alias="class")
    confidence: float                # 0–100 (porcentaje)
    bbox: List[float]                # [x1, y1, x2, y2] normalizados
    mask: Optional[List[List[float]]] = None

    model_config = {"populate_by_name": True}


# ── Imagen con resultados ─────────────────────────────────────────────────────

class ImageResult(BaseModel):
    filename: str
    has_defects: bool
    detection_count: int
    confidence_avg: float
    download_url: Optional[str] = None
    detections: List[DetectionOut] = Field(default_factory=list)


# ── Resumen del lote ──────────────────────────────────────────────────────────

class BatchSummary(BaseModel):
    total_images: int
    defective_images: int
    total_defects: int


# ── Resultado completo ────────────────────────────────────────────────────────

class JobResultsResponse(BaseModel):
    job_id: str
    status: str
    timestamp: str
    processing_time_ms: Optional[float] = None
    images: List[ImageResult] = Field(default_factory=list)
    summary: BatchSummary


# ── Adaptador JSONL → JobResultsResponse ─────────────────────────────────────

def adapt_jsonl_record(record: Dict[str, Any]) -> Dict[str, Any]:
    """Convierte un registro JSONL del Batch Endpoint al formato del frontend.

    Entrada (batch endpoint)::

        {
            "filename": "image.jpg",
            "has_defects": true,
            "detections_count": 8,
            "detections": [
                {
                    "class_name": "Incorrect_installation",
                    "class_id": 1,
                    "confidence": 0.9996,
                    "bbox": [0.22, 0.46, 0.34, 0.70],
                    "mask_points": [[0.21, 0.45], ...]
                }
            ]
        }

    Salida (para frontend)::

        {
            "filename": "image.jpg",
            "has_defects": true,
            "detection_count": 8,
            "confidence_avg": 0.85,
            "detections": [
                {
                    "class": "Incorrect_installation",
                    "confidence": 99.96,
                    "bbox": [0.22, 0.46, 0.34, 0.70],
                    "mask": [[0.21, 0.45], ...]
                }
            ]
        }
    """
    detections_raw = record.get("detections", [])
    detections_out = []
    confidences = []

    for det in detections_raw:
        conf_raw = det.get("confidence", 0.0)
        confidences.append(conf_raw)
        detections_out.append(
            {
                "class": det.get("class_name", ""),
                "confidence": round(conf_raw * 100, 2),
                "bbox": det.get("bbox", []),
                "mask": det.get("mask_points"),
            }
        )

    avg_conf = round(sum(confidences) / len(confidences), 4) if confidences else 0.0

    return {
        "filename": record.get("filename", ""),
        "has_defects": record.get("has_defects", False),
        "detection_count": record.get("detections_count", len(detections_raw)),
        "confidence_avg": avg_conf,
        "detections": detections_out,
    }
