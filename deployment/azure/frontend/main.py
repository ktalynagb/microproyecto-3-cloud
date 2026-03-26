"""Frontend FastAPI para el pipeline de batch inference de defectos en PCB.

Endpoints:
    POST /api/upload-batch    - Recibe hasta 10 imágenes y lanza el pipeline.
    GET  /api/status/{batch_id} - Consulta el estado del procesamiento.
    GET  /api/download/{batch_id} - Descarga links SAS de resultados.
    POST /api/export-pdf      - Genera el PDF del reporte localmente.

Las credenciales de Azure se leen desde variables de entorno:
    AZURE_STORAGE_ACCOUNT, AZURE_STORAGE_KEY, AZURE_CONTAINER_NAME, PCB_MODEL_PATH
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import uvicorn
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi import Request

# Añadir el directorio de componentes al path
_COMPONENTS_DIR = Path(__file__).resolve().parent.parent / "components"
sys.path.insert(0, str(_COMPONENTS_DIR))

from batch_inference import BatchInference
from batch_receiver import BatchReceiver
from blob_exporter import BlobExporter
from config import config
from delivery import DeliveryService
from logger import get_logger
from post_processor import PostProcessor

logger = get_logger("frontend")

# ── Estado en memoria de los lotes (en producción usar Redis/DB) ──────────
_batch_status: dict[str, dict[str, Any]] = {}
_batch_results: dict[str, Any] = {}

# ── Instancias de servicios ────────────────────────────────────────────────
_receiver = BatchReceiver()
_exporter = BlobExporter()
_delivery = DeliveryService()
_processor = PostProcessor()
_engine: BatchInference | None = None


def _get_engine() -> BatchInference:
    global _engine
    if _engine is None:
        _engine = BatchInference(model_path=config.model_path)
    return _engine


# ── App ────────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Iniciando servidor FastAPI PCB Batch Inference...")
    # Pre-cargar el modelo en background si existe
    model_path = Path(config.model_path)
    if model_path.exists():
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, _get_engine)
        logger.info("Modelo cargado: %s", config.model_path)
    else:
        logger.warning("Modelo no encontrado en: %s", config.model_path)
    yield
    logger.info("Cerrando servidor.")


app = FastAPI(
    title="PCB Defect Inspector - Batch Inference API",
    description=(
        "Pipeline de inspección de defectos en PCB usando YOLOv8n. "
        "Soporta lotes de hasta 10 imágenes con exportación temporal a Azure Blob Storage."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# Templates HTML
_TEMPLATES_DIR = Path(__file__).resolve().parent / "templates"
if _TEMPLATES_DIR.exists():
    templates = Jinja2Templates(directory=str(_TEMPLATES_DIR))


# ── Endpoints ─────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """Interfaz web principal."""
    if _TEMPLATES_DIR.exists() and ((_TEMPLATES_DIR / "index.html").exists()):
        return templates.TemplateResponse("index.html", {"request": request})
    return HTMLResponse(
        "<h1>PCB Defect Inspector</h1>"
        "<p>API disponible en <a href='/docs'>/docs</a></p>"
    )


@app.post("/api/upload-batch")
async def upload_batch(files: list[UploadFile] = File(...)) -> JSONResponse:
    """Recibe hasta 10 imágenes PCB y lanza el pipeline de inferencia.

    Returns:
        JSON con batch_id y estado 'En procesamiento'.
    """
    if not files:
        raise HTTPException(status_code=400, detail="No se enviaron archivos.")
    if len(files) > config.max_batch_size:
        raise HTTPException(
            status_code=400,
            detail=f"Máximo {config.max_batch_size} imágenes por lote.",
        )

    # Leer contenido de los archivos
    image_data: list[tuple[str, bytes]] = []
    for upload in files:
        content = await upload.read()
        filename = upload.filename or "image.jpg"
        image_data.append((filename, content))

    # Recibir lote
    try:
        batch = _receiver.receive(image_data)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    _batch_status[batch.batch_id] = {
        "status": "En procesamiento",
        "batch_id": batch.batch_id,
        "total_images": batch.size,
        "received_at": batch.received_at,
    }

    # Ejecutar pipeline en background
    asyncio.create_task(_run_pipeline(batch))

    return JSONResponse(
        {
            "batch_id": batch.batch_id,
            "status": "En procesamiento",
            "total_images": batch.size,
            "message": "Lote recibido. Consulta el estado en /api/status/{batch_id}",
        }
    )


async def _run_pipeline(batch: Any) -> None:  # noqa: ANN401
    """Ejecuta el pipeline completo de inferencia en background."""
    batch_id = batch.batch_id
    try:
        loop = asyncio.get_event_loop()

        # Etapa 2: Inferencia
        engine = _get_engine()
        batch_result = await loop.run_in_executor(None, engine.run, batch)

        # Etapa 3: Post-procesado
        annotated = await loop.run_in_executor(None, _processor.process, batch, batch_result)

        # Etapa 4: Exportación
        export_result = await loop.run_in_executor(
            None, _exporter.export, batch, batch_result, annotated
        )

        # Etapa 5: Entrega
        links = await loop.run_in_executor(
            None, _delivery.generate_download_links, export_result
        )

        _batch_results[batch_id] = {
            "export_result": export_result,
            "links": links,
            "annotated": annotated,
        }
        _batch_status[batch_id] = {
            **_batch_status.get(batch_id, {}),
            "status": "Completado",
            "images_with_defects": sum(1 for ai in annotated if ai.has_defects),
            "total_detections": sum(ai.detections_count for ai in annotated),
            "total_time_ms": batch_result.total_time_ms,
        }
        logger.info("Pipeline completado para lote %s", batch_id)

    except Exception as exc:  # noqa: BLE001
        logger.error("Error en pipeline del lote %s: %s", batch_id, exc)
        _batch_status[batch_id] = {
            **_batch_status.get(batch_id, {}),
            "status": "Error",
            "error": str(exc),
        }


@app.get("/api/status/{batch_id}")
async def get_status(batch_id: str) -> JSONResponse:
    """Consulta el estado del procesamiento de un lote.

    Returns:
        JSON con estado actual, número de defectos y tiempo de procesamiento.
    """
    if batch_id not in _batch_status:
        raise HTTPException(status_code=404, detail=f"Lote {batch_id!r} no encontrado.")
    return JSONResponse(_batch_status[batch_id])


@app.get("/api/download/{batch_id}")
async def download_results(batch_id: str) -> JSONResponse:
    """Retorna las URLs SAS para descarga temporal de resultados.

    Returns:
        JSON con URLs de imágenes anotadas, resumen JSON y PDF.
    """
    if batch_id not in _batch_results:
        status = _batch_status.get(batch_id, {}).get("status", "Desconocido")
        if status == "En procesamiento":
            raise HTTPException(status_code=202, detail="El lote aún está en procesamiento.")
        if status == "Error":
            raise HTTPException(
                status_code=500,
                detail=_batch_status[batch_id].get("error", "Error desconocido."),
            )
        raise HTTPException(status_code=404, detail=f"Lote {batch_id!r} no encontrado.")

    result = _batch_results[batch_id]
    links = result["links"]
    return JSONResponse(
        {
            "batch_id": batch_id,
            "image_links": links.image_links,
            "summary_link": links.summary_link,
            "pdf_link": links.pdf_link,
            "local_pdf_path": links.local_pdf_path,
            "expires_at": links.expires_at,
        }
    )


@app.post("/api/export-pdf")
async def export_pdf(batch_id: str) -> FileResponse:
    """Descarga el PDF de reporte generado localmente para un lote procesado.

    El PDF se almacena temporalmente en el servidor y se elimina después de
    la descarga (sin persistencia permanente en la nube).
    """
    if batch_id not in _batch_results:
        raise HTTPException(status_code=404, detail=f"Lote {batch_id!r} no encontrado.")

    links = _batch_results[batch_id]["links"]
    if not links.local_pdf_path or not Path(links.local_pdf_path).exists():
        raise HTTPException(
            status_code=404,
            detail="PDF no disponible. Puede que reportlab no esté instalado.",
        )

    return FileResponse(
        path=links.local_pdf_path,
        media_type="application/pdf",
        filename=f"pcb_report_{batch_id}.pdf",
    )


@app.get("/health")
async def health() -> JSONResponse:
    """Endpoint de health check."""
    return JSONResponse(
        {
            "status": "ok",
            "model_loaded": _engine is not None,
            "model_path": config.model_path,
        }
    )


# ── Punto de entrada ───────────────────────────────────────────────────────

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=int(os.environ.get("PORT", "8000")),
        reload=False,
        log_level="info",
    )
