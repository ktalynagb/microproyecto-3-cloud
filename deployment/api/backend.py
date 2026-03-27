"""Backend FastAPI para el sistema de inspección de PCB.

Conecta el Frontend Streamlit con el Azure ML Batch Endpoint y Azure Blob Storage.

Endpoints:
    POST /api/v1/infer                  Subir imágenes (hasta 10) y lanzar batch job
    GET  /api/v1/jobs/{job_id}          Consultar estado del job (polling cada 5s)
    GET  /api/v1/jobs/{job_id}/results  Descargar resultados JSONL + SAS URLs
    GET  /api/v1/health                 Health check

Variables de entorno requeridas (ver .env.example):
    BACKEND_API_KEY             API Key para autenticar clientes
    AZURE_SUBSCRIPTION_ID       Suscripción de Azure
    AZURE_RESOURCE_GROUP        Grupo de recursos del workspace de Azure ML
    AZURE_WORKSPACE_NAME        Nombre del workspace de Azure ML
    AZURE_STORAGE_ACCOUNT       Nombre de la cuenta de storage
    AZURE_STORAGE_KEY           Clave de acceso (nunca se expone al frontend)
    AZURE_INPUT_CONTAINER       Container de entrada para el batch (workspaceblobstore)

Uso::

    uvicorn backend:app --host 0.0.0.0 --port 8080
"""

from __future__ import annotations

import logging
import logging.config
import os
import uuid
from datetime import datetime, timezone
from typing import Any, Dict

from fastapi import Depends, FastAPI, File, HTTPException, Security, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security.api_key import APIKeyHeader

from azure_ml_client import AzureMLBatchClient
from blob_manager import BlobManager
from models import (
    HealthResponse,
    InferResponse,
    JobResultsResponse,
    JobStatusResponse,
    BatchSummary,
    ImageResult,
    DetectionOut,
    adapt_jsonl_record,
)

# ── Logging JSON estructurado ─────────────────────────────────────────────────

_LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO").upper()

logging.config.dictConfig(
    {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "json": {
                "format": '{"time":"%(asctime)s","level":"%(levelname)s","logger":"%(name)s","msg":"%(message)s"}',
                "datefmt": "%Y-%m-%dT%H:%M:%SZ",
            }
        },
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "formatter": "json",
            }
        },
        "root": {"level": _LOG_LEVEL, "handlers": ["console"]},
    }
)
LOG = logging.getLogger(__name__)

# ── FastAPI app ───────────────────────────────────────────────────────────────

app = FastAPI(
    title="PCB Defect Inspection API",
    description="Backend para el sistema de inspección de calidad de PCB usando Azure ML Batch.",
    version="1.0.0",
)

# CORS – permite conexiones del frontend Streamlit (local y cloud)
_ALLOWED_ORIGINS = [
    "http://localhost:8501",
    "http://127.0.0.1:8501",
    "https://*.azurecontainerapps.io",
    "https://*.azurewebsites.net",
]
_EXTRA_ORIGINS = os.environ.get("CORS_EXTRA_ORIGINS", "")
if _EXTRA_ORIGINS:
    _ALLOWED_ORIGINS.extend(o.strip() for o in _EXTRA_ORIGINS.split(",") if o.strip())

app.add_middleware(
    CORSMiddleware,
    allow_origins=_ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── API Key auth ──────────────────────────────────────────────────────────────

_API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=False)
_BACKEND_API_KEY = os.environ.get("BACKEND_API_KEY", "")


def _require_api_key(api_key: str | None = Security(_API_KEY_HEADER)) -> str:
    """Valida el header X-API-Key.

    Si BACKEND_API_KEY no está configurado en el entorno, la autenticación
    queda deshabilitada (útil para desarrollo local).
    """
    if not _BACKEND_API_KEY:
        LOG.warning("BACKEND_API_KEY no configurada – autenticación deshabilitada")
        return ""
    if api_key != _BACKEND_API_KEY:
        raise HTTPException(status_code=403, detail="API Key inválida o ausente")
    return api_key


# ── Clientes ──────────────────────────────────────────────────────────────────

_batch_client = AzureMLBatchClient()
_blob_manager = BlobManager()

# ── Constantes ────────────────────────────────────────────────────────────────

_MAX_IMAGES = 10


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/api/v1/health", response_model=HealthResponse, tags=["system"])
def health_check() -> HealthResponse:
    """Verifica que el backend esté funcionando."""
    return HealthResponse(
        status="ok",
        batch_endpoint=_batch_client.endpoint_url,
        blob_storage=f"https://{_blob_manager.account}.blob.core.windows.net",
    )


@app.post(
    "/api/v1/infer",
    response_model=InferResponse,
    status_code=202,
    dependencies=[Depends(_require_api_key)],
    tags=["inference"],
)
async def submit_inference(
    files: list[UploadFile] = File(..., description="Imágenes PCB (máximo 10)"),
) -> InferResponse:
    """Sube imágenes al Blob Storage y lanza un Batch Endpoint job.

    - Acepta hasta 10 imágenes JPG/PNG.
    - Retorna un ``job_id`` para hacer polling con GET /api/v1/jobs/{job_id}.
    """
    if not files:
        raise HTTPException(status_code=400, detail="Se requiere al menos una imagen")
    if len(files) > _MAX_IMAGES:
        raise HTTPException(
            status_code=400,
            detail=f"Máximo {_MAX_IMAGES} imágenes por lote",
        )

    # Leer imágenes
    images: Dict[str, bytes] = {}
    for upload in files:
        if not _is_image(upload.filename or ""):
            raise HTTPException(
                status_code=422,
                detail=f"Formato no soportado: {upload.filename}. Use JPG o PNG.",
            )
        content = await upload.read()
        if not content:
            raise HTTPException(
                status_code=422,
                detail=f"El archivo {upload.filename} está vacío",
            )
        images[upload.filename or f"image_{uuid.uuid4().hex[:8]}.jpg"] = content

    # Generar job_id y run_id únicos
    job_id = str(uuid.uuid4())
    run_id = f"run_{uuid.uuid4().hex[:12]}"
    LOG.info("Subiendo %d imágenes para job %s (run_id=%s)", len(images), job_id, run_id)

    try:
        input_url = _blob_manager.upload_images(images, folder=job_id)
    except Exception as exc:
        LOG.error("Error subiendo imágenes: %s", exc)
        raise HTTPException(status_code=502, detail=f"Error subiendo imágenes: {exc}") from exc

    # Enviar job al Batch Endpoint con ruta de salida dinámica (run_id)
    try:
        az_job_id = _batch_client.submit_job(input_url, run_id)
    except Exception as exc:
        LOG.error("Error enviando job al Batch Endpoint: %s", exc)
        raise HTTPException(
            status_code=502,
            detail=f"Error enviando job al Batch Endpoint: {exc}",
        ) from exc

    LOG.info("Job %s → Azure job %s (run_id=%s)", job_id, az_job_id, run_id)

    # Guardar mapeo job_id → az_job_id + run_id en memoria
    _job_store[job_id] = {
        "az_job_id": az_job_id,
        "run_id": run_id,
        "created_at": datetime.now(tz=timezone.utc).isoformat(),
        "image_names": list(images.keys()),
    }

    return InferResponse(
        job_id=job_id,
        status="submitted",
        message=f"Job enviado. Azure job id: {az_job_id}",
    )


@app.get(
    "/api/v1/jobs/{job_id}",
    response_model=JobStatusResponse,
    dependencies=[Depends(_require_api_key)],
    tags=["inference"],
)
def get_job_status(job_id: str) -> JobStatusResponse:
    """Retorna el estado actual del job.

    El frontend hace polling cada 5 segundos hasta que ``status`` sea
    ``completed`` o ``failed``.
    """
    entry = _job_store.get(job_id)
    if entry is None:
        raise HTTPException(status_code=404, detail=f"Job no encontrado: {job_id}")

    az_job_id: str = entry["az_job_id"]
    try:
        info = _batch_client.get_job_status(az_job_id)
    except Exception as exc:
        LOG.error("Error consultando job %s: %s", az_job_id, exc)
        raise HTTPException(
            status_code=502,
            detail=f"Error consultando estado del job: {exc}",
        ) from exc

    return JobStatusResponse(
        job_id=job_id,
        status=info.get("status", "unknown"),
        created_at=entry.get("created_at"),
        updated_at=info.get("updated_at"),
        message=info.get("message"),
    )


@app.get(
    "/api/v1/jobs/{job_id}/results",
    response_model=JobResultsResponse,
    dependencies=[Depends(_require_api_key)],
    tags=["inference"],
)
def get_job_results(job_id: str) -> JobResultsResponse:
    """Descarga resultados JSONL del Batch Endpoint y genera SAS URLs.

    Solo disponible cuando el job está en estado ``completed``.
    """
    entry = _job_store.get(job_id)
    if entry is None:
        raise HTTPException(status_code=404, detail=f"Job no encontrado: {job_id}")

    az_job_id: str = entry["az_job_id"]
    run_id: str = entry["run_id"]

    # Verificar estado
    try:
        info = _batch_client.get_job_status(az_job_id)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    if info.get("status") != "completed":
        raise HTTPException(
            status_code=409,
            detail=f"Job aún no completado (estado: {info.get('status')})",
        )

    # Obtener el container del datastore por defecto de Azure ML
    try:
        ml_container = _batch_client.get_default_container()
    except Exception as exc:
        LOG.error("Error obteniendo container del datastore: %s", exc)
        raise HTTPException(
            status_code=502,
            detail=f"Error obteniendo container del datastore: {exc}",
        ) from exc

    # Descargar JSONL desde Blob Storage usando la clave de almacenamiento
    try:
        records = _blob_manager.download_jsonl(run_id, ml_container)
    except Exception as exc:
        LOG.error("Error descargando resultados JSONL: %s", exc)
        raise HTTPException(
            status_code=502,
            detail=f"Error descargando resultados: {exc}",
        ) from exc

    # Generar SAS URLs para las imágenes anotadas (opcional, no bloquea)
    try:
        sas_urls = _blob_manager.generate_sas_urls_for_folder(job_id)
    except Exception as exc:
        LOG.warning("No se pudieron generar SAS URLs: %s. Resultados sin URLs de descarga.", exc)
        sas_urls = {}

    # Adaptar registros JSONL al formato del frontend
    t_start = datetime.now(tz=timezone.utc)
    images_out: list[ImageResult] = []
    total_defects = 0
    defective_count = 0

    for rec in records:
        adapted = adapt_jsonl_record(rec)
        filename = adapted["filename"]
        detections = [
            DetectionOut(**{**d, "class": d["class"]}) for d in adapted["detections"]
        ]
        if adapted["has_defects"]:
            defective_count += 1
        total_defects += adapted["detection_count"]

        images_out.append(
            ImageResult(
                filename=filename,
                has_defects=adapted["has_defects"],
                detection_count=adapted["detection_count"],
                confidence_avg=adapted["confidence_avg"],
                download_url=sas_urls.get(filename),
                detections=detections,
            )
        )

    processing_ms = (datetime.now(tz=timezone.utc) - t_start).total_seconds() * 1000

    return JobResultsResponse(
        job_id=job_id,
        status="completed",
        timestamp=t_start.isoformat(),
        processing_time_ms=round(processing_ms, 2),
        images=images_out,
        summary=BatchSummary(
            total_images=len(images_out),
            defective_images=defective_count,
            total_defects=total_defects,
        ),
    )


# ── In-memory job store ───────────────────────────────────────────────────────
# Para producción se reemplazaría por Redis o Azure Table Storage.

_job_store: Dict[str, Any] = {}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _is_image(filename: str) -> bool:
    """Verifica que el archivo tenga una extensión de imagen soportada."""
    return filename.lower().rsplit(".", 1)[-1] in {"jpg", "jpeg", "png"}


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "backend:app",
        host=os.environ.get("API_HOST", "0.0.0.0"),
        port=int(os.environ.get("API_PORT", "8080")),
        log_level=_LOG_LEVEL.lower(),
    )
