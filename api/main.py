"""FastAPI backend para el pipeline PCB Defect Detection.

Expone endpoints REST para:
  - Enviar imágenes al Batch Endpoint de Azure ML
  - Hacer polling del estado del job
  - Generar SAS URLs para descargar resultados sin exponer credenciales
  - Adaptar la salida JSONL al formato que espera el frontend Streamlit

Autenticación: API Key en el header ``X-API-Key``.
CORS habilitado para Streamlit local y cloud.
"""

from __future__ import annotations

import asyncio
import base64
import io
import json
import logging
import os
import time
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import httpx
from azure.storage.blob import (
    BlobServiceClient,
    BlobSasPermissions,
    generate_blob_sas,
)
from fastapi import Depends, FastAPI, File, HTTPException, Security, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.security.api_key import APIKeyHeader
from pydantic import BaseModel

# ---------------------------------------------------------------------------
# Configuración
# ---------------------------------------------------------------------------

API_KEY = os.environ.get("PCB_API_KEY", "changeme-secret-key")
BATCH_ENDPOINT_URL = os.environ.get(
    "BATCH_ENDPOINT_URL",
    "https://pcb-batch-inference.centralus.inference.ml.azure.com/jobs",
)
AZURE_ML_TOKEN = os.environ.get("AZURE_ML_TOKEN", "")  # Bearer token de Azure ML
STORAGE_ACCOUNT = os.environ.get("STORAGE_ACCOUNT", "pcbmlworstorage428505ef7")
STORAGE_KEY = os.environ.get("STORAGE_KEY", "")
STORAGE_CONNECTION_STRING = os.environ.get("AZURE_STORAGE_CONNECTION_STRING", "")
PREDICTIONS_CONTAINER = os.environ.get(
    "PREDICTIONS_CONTAINER",
    "azureml-blobstore-fa3e2152-1a09-4e81-acb4-3f701118ca5e",
)
INPUT_CONTAINER = os.environ.get("INPUT_CONTAINER", "pcb-inference-input")

POLL_INTERVAL_S = int(os.environ.get("POLL_INTERVAL_S", "10"))
MAX_POLL_ATTEMPTS = int(os.environ.get("MAX_POLL_ATTEMPTS", "36"))  # ~6 min

CORS_ORIGINS = os.environ.get(
    "CORS_ORIGINS",
    "http://localhost:8501,http://127.0.0.1:8501,https://*.streamlit.app",
).split(",")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("pcb_api")

# ---------------------------------------------------------------------------
# Aplicación FastAPI
# ---------------------------------------------------------------------------

app = FastAPI(
    title="PCB Defect Detection API",
    description="API REST para detección de defectos en PCBs usando Azure ML.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_origin_regex=r"https://.*\.streamlit\.app",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Autenticación
# ---------------------------------------------------------------------------

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


async def verify_api_key(api_key: str = Security(api_key_header)) -> str:
    if api_key != API_KEY:
        raise HTTPException(status_code=403, detail="API Key inválida o ausente.")
    return api_key


# ---------------------------------------------------------------------------
# Modelos Pydantic
# ---------------------------------------------------------------------------

class Detection(BaseModel):
    class_name: str
    class_id: int
    confidence: float
    bbox: list[float]
    mask_points: list[list[float]]


class ImageResult(BaseModel):
    filename: str
    has_defects: bool
    detections_count: int
    detections: list[Detection]
    inference_time_ms: float
    error: str | None = None
    no_defect_notification: str = "✅ PCB sin defectos detectados"
    annotated_image_url: str | None = None
    timestamp: str = ""


class BatchJobStatus(BaseModel):
    job_id: str
    status: str  # "running" | "completed" | "failed"
    progress: float  # 0.0 – 1.0
    message: str
    results: list[ImageResult] | None = None
    output_url: str | None = None


class InferenceResponse(BaseModel):
    job_id: str
    status: str
    message: str


# ---------------------------------------------------------------------------
# Almacén en memoria de jobs (producción usaría Redis o DB)
# ---------------------------------------------------------------------------

_jobs: dict[str, dict[str, Any]] = {}


# ---------------------------------------------------------------------------
# Helpers Azure Blob Storage
# ---------------------------------------------------------------------------

def _get_blob_service_client() -> BlobServiceClient:
    if STORAGE_CONNECTION_STRING:
        return BlobServiceClient.from_connection_string(STORAGE_CONNECTION_STRING)
    if STORAGE_ACCOUNT and STORAGE_KEY:
        url = f"https://{STORAGE_ACCOUNT}.blob.core.windows.net"
        return BlobServiceClient(account_url=url, credential=STORAGE_KEY)
    raise RuntimeError(
        "Configure AZURE_STORAGE_CONNECTION_STRING o STORAGE_ACCOUNT+STORAGE_KEY."
    )


def _generate_sas_url(container: str, blob_name: str, expiry_hours: int = 1) -> str:
    """Genera una SAS URL de solo lectura para un blob."""
    if not STORAGE_KEY:
        # Sin clave, retornar URL directa (solo para cuentas públicas)
        return (
            f"https://{STORAGE_ACCOUNT}.blob.core.windows.net/"
            f"{container}/{blob_name}"
        )

    expiry = datetime.now(timezone.utc) + timedelta(hours=expiry_hours)
    sas_token = generate_blob_sas(
        account_name=STORAGE_ACCOUNT,
        container_name=container,
        blob_name=blob_name,
        account_key=STORAGE_KEY,
        permission=BlobSasPermissions(read=True),
        expiry=expiry,
    )
    return (
        f"https://{STORAGE_ACCOUNT}.blob.core.windows.net/"
        f"{container}/{blob_name}?{sas_token}"
    )


async def _upload_images_to_blob(
    files: list[UploadFile], job_id: str
) -> list[str]:
    """Sube imágenes al blob de entrada y retorna sus rutas."""
    try:
        client = _get_blob_service_client()
        container_client = client.get_container_client(INPUT_CONTAINER)
        try:
            container_client.create_container()
        except Exception:
            pass  # ya existe

        blob_paths: list[str] = []
        for f in files:
            data = await f.read()
            blob_name = f"{job_id}/{f.filename}"
            container_client.upload_blob(blob_name, data, overwrite=True)
            blob_paths.append(blob_name)
            logger.info(f"Subido: {blob_name}")
        return blob_paths
    except Exception as exc:
        logger.error(f"Error subiendo imágenes: {exc}")
        raise


async def _submit_batch_job(blob_paths: list[str], job_id: str) -> str:
    """Envía el job al Batch Endpoint de Azure ML."""
    if not AZURE_ML_TOKEN:
        raise HTTPException(
            status_code=503,
            detail="AZURE_ML_TOKEN no configurado. No es posible enviar jobs.",
        )

    input_data = {
        "properties": {
            "InputData": {
                "images": {
                    "JobInputType": "UriFolder",
                    "Uri": (
                        f"https://{STORAGE_ACCOUNT}.blob.core.windows.net/"
                        f"{INPUT_CONTAINER}/{job_id}/"
                    ),
                }
            }
        }
    }

    headers = {
        "Authorization": f"Bearer {AZURE_ML_TOKEN}",
        "Content-Type": "application/json",
    }

    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.post(
            BATCH_ENDPOINT_URL, json=input_data, headers=headers
        )
        if resp.status_code not in (200, 201, 202):
            raise HTTPException(
                status_code=502,
                detail=f"Azure ML respondió {resp.status_code}: {resp.text}",
            )
        data = resp.json()
        # El ID del job viene en distintos campos según la versión de la API
        azure_job_id: str = (
            data.get("name")
            or data.get("id", "")
            or data.get("properties", {}).get("jobId", job_id)
        )
        return azure_job_id


async def _poll_job_status(azure_job_id: str) -> dict[str, Any]:
    """Consulta el estado de un job en Azure ML."""
    headers = {"Authorization": f"Bearer {AZURE_ML_TOKEN}"}
    url = f"{BATCH_ENDPOINT_URL}/{azure_job_id}"
    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.get(url, headers=headers)
        if resp.status_code != 200:
            return {"status": "Unknown", "error": resp.text}
        return resp.json()


async def _fetch_results(azure_job_id: str, job_id: str) -> list[ImageResult]:
    """Descarga y parsea el JSONL de resultados desde Blob Storage."""
    try:
        client = _get_blob_service_client()
        # Buscar archivos de salida del job
        container_client = client.get_container_client(PREDICTIONS_CONTAINER)
        prefix = f"azureml/{azure_job_id}/predictions/"
        blobs = list(container_client.list_blobs(name_starts_with=prefix))

        results: list[ImageResult] = []
        for blob in blobs:
            if not blob.name.endswith(".jsonl"):
                continue
            blob_client = container_client.get_blob_client(blob.name)
            content = blob_client.download_blob().readall().decode("utf-8")
            for line in content.strip().splitlines():
                if not line:
                    continue
                try:
                    record = json.loads(line)
                    results.append(_record_to_image_result(record))
                except json.JSONDecodeError:
                    continue
        return results
    except Exception as exc:
        logger.error(f"Error descargando resultados: {exc}")
        return []


def _record_to_image_result(record: dict[str, Any]) -> ImageResult:
    """Convierte un registro JSONL al modelo ImageResult del frontend."""
    detections = [
        Detection(
            class_name=d.get("class_name", "Unknown"),
            class_id=d.get("class_id", 0),
            confidence=d.get("confidence", 0.0),
            bbox=d.get("bbox", [0, 0, 0, 0]),
            mask_points=d.get("mask_points", []),
        )
        for d in record.get("detections", [])
    ]
    return ImageResult(
        filename=record.get("filename", "unknown"),
        has_defects=record.get("has_defects", False),
        detections_count=record.get("detections_count", len(detections)),
        detections=detections,
        inference_time_ms=record.get("inference_time_ms", 0.0),
        error=record.get("error"),
        no_defect_notification=record.get(
            "no_defect_notification", "✅ PCB sin defectos detectados"
        ),
        timestamp=datetime.now(timezone.utc).isoformat(),
    )


# ---------------------------------------------------------------------------
# Background task: polling
# ---------------------------------------------------------------------------

async def _run_batch_job(job_id: str, azure_job_id: str) -> None:
    """Tarea asíncrona que hace polling hasta que el job termine."""
    _jobs[job_id]["azure_job_id"] = azure_job_id
    _jobs[job_id]["status"] = "running"
    _jobs[job_id]["progress"] = 0.05

    terminal_statuses = {"Completed", "Failed", "Canceled", "NotStarted"}
    running_statuses = {"Running", "Starting", "Provisioning", "Preparing"}

    for attempt in range(MAX_POLL_ATTEMPTS):
        await asyncio.sleep(POLL_INTERVAL_S)

        try:
            status_data = await _poll_job_status(azure_job_id)
        except Exception as exc:
            logger.warning(f"[poll] Error en intento {attempt}: {exc}")
            continue

        status_str: str = (
            status_data.get("status")
            or status_data.get("properties", {}).get("status", "Unknown")
        )

        progress = min(0.05 + (attempt + 1) / MAX_POLL_ATTEMPTS * 0.9, 0.95)
        _jobs[job_id]["progress"] = progress
        _jobs[job_id]["raw_status"] = status_str
        logger.info(f"[poll] job={job_id} status={status_str} attempt={attempt}")

        if status_str in terminal_statuses:
            if status_str == "Completed":
                results = await _fetch_results(azure_job_id, job_id)
                _jobs[job_id]["status"] = "completed"
                _jobs[job_id]["progress"] = 1.0
                _jobs[job_id]["results"] = [r.model_dump() for r in results]
                _jobs[job_id]["message"] = (
                    f"Completado. {len(results)} imágenes procesadas."
                )
            else:
                _jobs[job_id]["status"] = "failed"
                _jobs[job_id]["progress"] = 1.0
                _jobs[job_id]["message"] = f"Job falló con estado: {status_str}"
            return

    # Timeout
    _jobs[job_id]["status"] = "failed"
    _jobs[job_id]["progress"] = 1.0
    _jobs[job_id]["message"] = "Timeout esperando el job de Azure ML."


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health", tags=["meta"])
async def health() -> dict[str, str]:
    """Health check para ACI y load balancers."""
    return {"status": "ok", "version": "1.0.0"}


@app.post(
    "/inference/batch",
    response_model=InferenceResponse,
    tags=["inference"],
    dependencies=[Depends(verify_api_key)],
)
async def submit_batch_inference(
    files: list[UploadFile] = File(...),
) -> InferenceResponse:
    """
    Envía imágenes al Batch Endpoint de Azure ML.

    - Sube las imágenes a Blob Storage.
    - Envía el job al endpoint.
    - Inicia el polling en background.
    - Retorna el ``job_id`` para consultar el estado.
    """
    if not files:
        raise HTTPException(status_code=400, detail="No se enviaron imágenes.")

    job_id = str(uuid.uuid4())
    _jobs[job_id] = {
        "job_id": job_id,
        "status": "submitting",
        "progress": 0.0,
        "message": "Subiendo imágenes…",
        "results": None,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }

    try:
        await _upload_images_to_blob(files, job_id)
        _jobs[job_id]["message"] = "Enviando job a Azure ML…"
        _jobs[job_id]["progress"] = 0.02

        azure_job_id = await _submit_batch_job(
            [f.filename or f"image_{i}" for i, f in enumerate(files)], job_id
        )
        _jobs[job_id]["message"] = f"Job enviado: {azure_job_id}"

        # Iniciar polling en background
        asyncio.create_task(_run_batch_job(job_id, azure_job_id))

    except HTTPException:
        _jobs[job_id]["status"] = "failed"
        _jobs[job_id]["message"] = "Error enviando el job."
        raise
    except Exception as exc:
        _jobs[job_id]["status"] = "failed"
        _jobs[job_id]["message"] = str(exc)
        raise HTTPException(status_code=500, detail=str(exc))

    return InferenceResponse(
        job_id=job_id,
        status="submitting",
        message="Job enviado. Usa /inference/status/{job_id} para monitorear.",
    )


@app.get(
    "/inference/status/{job_id}",
    response_model=BatchJobStatus,
    tags=["inference"],
    dependencies=[Depends(verify_api_key)],
)
async def get_job_status(job_id: str) -> BatchJobStatus:
    """Consulta el estado de un job de inferencia."""
    if job_id not in _jobs:
        raise HTTPException(status_code=404, detail=f"Job {job_id} no encontrado.")

    j = _jobs[job_id]
    results = None
    if j.get("results"):
        results = [ImageResult(**r) for r in j["results"]]

    return BatchJobStatus(
        job_id=job_id,
        status=j.get("status", "unknown"),
        progress=j.get("progress", 0.0),
        message=j.get("message", ""),
        results=results,
    )


@app.get(
    "/inference/results/{job_id}",
    tags=["inference"],
    dependencies=[Depends(verify_api_key)],
)
async def get_job_results(job_id: str) -> JSONResponse:
    """Retorna los resultados completos de un job completado."""
    if job_id not in _jobs:
        raise HTTPException(status_code=404, detail=f"Job {job_id} no encontrado.")

    j = _jobs[job_id]
    if j.get("status") != "completed":
        raise HTTPException(
            status_code=409,
            detail=f"Job aún no completado. Estado: {j.get('status')}",
        )

    return JSONResponse(content={"job_id": job_id, "results": j.get("results", [])})


@app.get(
    "/storage/sas",
    tags=["storage"],
    dependencies=[Depends(verify_api_key)],
)
async def get_sas_url(
    container: str, blob: str, expiry_hours: int = 1
) -> dict[str, str]:
    """Genera una SAS URL temporal para descargar un blob sin exponer credenciales."""
    try:
        url = _generate_sas_url(container, blob, expiry_hours)
        return {"url": url, "expires_in_hours": str(expiry_hours)}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/jobs", tags=["meta"], dependencies=[Depends(verify_api_key)])
async def list_jobs() -> dict[str, Any]:
    """Lista todos los jobs conocidos por esta instancia."""
    return {
        "jobs": [
            {
                "job_id": j["job_id"],
                "status": j.get("status"),
                "progress": j.get("progress"),
                "created_at": j.get("created_at"),
            }
            for j in _jobs.values()
        ]
    }


@app.delete(
    "/jobs/{job_id}",
    tags=["meta"],
    dependencies=[Depends(verify_api_key)],
)
async def delete_job(job_id: str) -> dict[str, str]:
    """Elimina un job del registro en memoria."""
    if job_id not in _jobs:
        raise HTTPException(status_code=404, detail="Job no encontrado.")
    del _jobs[job_id]
    return {"message": f"Job {job_id} eliminado."}
