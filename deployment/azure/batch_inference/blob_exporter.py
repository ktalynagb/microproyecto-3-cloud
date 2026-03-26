"""ETAPA 4: Exportación Efímera (blob_exporter).

Sube imágenes anotadas a Azure Blob Storage en un contenedor temporal (TTL=24h),
genera el resumen diagnóstico en JSON y un PDF con las anotaciones.
No hay persistencia permanente; todos los blobs se borran tras TTL.

Uso como módulo:
    from blob_exporter import BlobExporter
    exporter = BlobExporter()
    export_result = exporter.export(batch_result, annotated_images)
"""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from inference_engine import BatchResult
from batch_receiver import Batch
from config import config
from logger import get_logger
from post_processor import AnnotatedImage

logger = get_logger("blob_exporter")


@dataclass
class ExportResult:
    """Resultado de la exportación de un lote."""

    batch_id: str
    blob_urls: dict[str, str]   # {filename: blob_url}
    summary_url: str | None
    pdf_url: str | None
    summary: dict[str, Any]
    local_pdf_path: str | None = None
    expires_at: str = field(
        default_factory=lambda: (
            datetime.now(tz=timezone.utc) + timedelta(hours=config.blob_ttl_hours)
        ).isoformat()
    )


def _generate_pdf(
    annotated_images: list[AnnotatedImage],
    batch_result: BatchResult,
    output_path: Path,
) -> None:
    """Genera un PDF con las imágenes anotadas y el resumen diagnóstico."""
    try:
        from reportlab.lib import colors
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.units import cm
        from reportlab.platypus import (
            Image as RLImage,
            Paragraph,
            SimpleDocTemplate,
            Spacer,
            Table,
            TableStyle,
        )
        from reportlab.lib.styles import getSampleStyleSheet
    except ImportError:
        logger.warning("reportlab no disponible; PDF no generado.")
        return

    import io
    from PIL import Image as PILImage

    styles = getSampleStyleSheet()
    doc = SimpleDocTemplate(str(output_path), pagesize=A4)
    story = []

    story.append(Paragraph(f"Reporte de Inspección PCB - Lote {batch_result.batch_id}", styles["Title"]))
    story.append(Paragraph(
        f"Generado: {datetime.now(tz=timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}",
        styles["Normal"],
    ))
    story.append(Spacer(1, 0.4 * cm))

    # Tabla resumen
    total_imgs = len(annotated_images)
    defective = sum(1 for ai in annotated_images if ai.has_defects)
    table_data = [
        ["Total imágenes", str(total_imgs)],
        ["Con defectos", str(defective)],
        ["Sin defectos", str(total_imgs - defective)],
        ["Tiempo total (ms)", str(batch_result.total_time_ms)],
    ]
    t = Table(table_data, colWidths=[7 * cm, 5 * cm])
    t.setStyle(
        TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.grey),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.black),
            ("FONTNAME", (0, 0), (-1, -1), "Helvetica"),
        ])
    )
    story.append(t)
    story.append(Spacer(1, 0.5 * cm))

    # Imágenes anotadas
    for ai in annotated_images:
        story.append(Paragraph(ai.filename, styles["Heading3"]))
        try:
            pil_img = PILImage.open(io.BytesIO(ai.annotated_bytes))
            # Escalar para caber en la página
            max_w, max_h = 14 * cm, 10 * cm
            w, h = pil_img.size
            ratio = min(max_w / w, max_h / h)
            rl_img = RLImage(io.BytesIO(ai.annotated_bytes), width=w * ratio, height=h * ratio)
            story.append(rl_img)
        except Exception as exc:  # noqa: BLE001
            story.append(Paragraph(f"[Error al renderizar imagen: {exc}]", styles["Normal"]))

        if ai.no_defect_notification:
            story.append(Paragraph(ai.no_defect_notification, styles["Normal"]))
        else:
            story.append(Paragraph(f"Defectos detectados: {ai.detections_count}", styles["Normal"]))
        story.append(Spacer(1, 0.3 * cm))

    doc.build(story)
    logger.info(f"PDF generado: {output_path}")


class BlobExporter:
    """Exporta resultados a Azure Blob Storage con TTL de 24 horas."""

    def __init__(
        self,
        account: str = config.azure_storage_account,
        key: str = config.azure_storage_key,
        container: str = config.azure_container_name,
        ttl_hours: int = config.blob_ttl_hours,
    ) -> None:
        self.account = account
        self.key = key
        self.container = container
        self.ttl_hours = ttl_hours
        self._client: Any = None

    def _get_client(self) -> Any:
        if self._client is None:
            from azure.storage.blob import BlobServiceClient
            conn_str = (
                f"DefaultEndpointsProtocol=https;"
                f"AccountName={self.account};"
                f"AccountKey={self.key};"
                "EndpointSuffix=core.windows.net"
            )
            self._client = BlobServiceClient.from_connection_string(conn_str)
        return self._client

    def _upload_blob(self, name: str, data: bytes, content_type: str) -> str:
        """Sube un blob y retorna su URL."""
        client = self._get_client()
        container_client = client.get_container_client(self.container)
        try:
            container_client.create_container()
        except Exception:
            pass  # Ya existe

        blob_client = container_client.get_blob_client(name)
        from azure.storage.blob import ContentSettings
        blob_client.upload_blob(
            data,
            overwrite=True,
            content_settings=ContentSettings(content_type=content_type),
        )
        return blob_client.url

    def export(
        self,
        batch: Batch,
        batch_result: BatchResult,
        annotated_images: list[AnnotatedImage],
    ) -> ExportResult:
        """Sube imágenes anotadas, JSON de resumen y PDF al Blob Storage."""
        batch_id = batch_result.batch_id
        blob_urls: dict[str, str] = {}
        summary_url: str | None = None
        pdf_url: str | None = None
        local_pdf_path: str | None = None

        # ── Subir imágenes anotadas ────────────────────────────────────────
        for ai in annotated_images:
            blob_name = f"{batch_id}/images/{ai.filename}"
            try:
                url = self._upload_blob(blob_name, ai.annotated_bytes, "image/jpeg")
                blob_urls[ai.filename] = url
                logger.info(f"Imagen subida: {blob_name}", extra={"batch_id": batch_id})
            except Exception as exc:  # noqa: BLE001
                logger.error(f"Error subiendo {blob_name}: {exc}")

        # ── Generar y subir resumen JSON ───────────────────────────────────
        summary = _build_summary(batch_result, annotated_images)
        summary_json = json.dumps(summary, indent=2, ensure_ascii=False).encode("utf-8")
        summary_blob = f"{batch_id}/summary.json"
        try:
            summary_url = self._upload_blob(summary_blob, summary_json, "application/json")
        except Exception as exc:  # noqa: BLE001
            logger.error(f"Error subiendo summary.json: {exc}")

        # ── Generar PDF y subirlo ──────────────────────────────────────────
        with tempfile.TemporaryDirectory() as tmpdir:
            pdf_path = Path(tmpdir) / f"{batch_id}_report.pdf"
            _generate_pdf(annotated_images, batch_result, pdf_path)
            local_pdf_path = str(pdf_path)
            if pdf_path.exists():
                pdf_bytes = pdf_path.read_bytes()
                pdf_blob = f"{batch_id}/report.pdf"
                try:
                    pdf_url = self._upload_blob(pdf_blob, pdf_bytes, "application/pdf")
                except Exception as exc:  # noqa: BLE001
                    logger.error(f"Error subiendo PDF: {exc}")
                # Guardar una copia temporal local para descarga inmediata
                local_tmp = config.temp_dir / batch_id
                local_tmp.mkdir(parents=True, exist_ok=True)
                local_pdf_copy = local_tmp / "report.pdf"
                import shutil
                shutil.copy2(pdf_path, local_pdf_copy)
                local_pdf_path = str(local_pdf_copy)

        return ExportResult(
            batch_id=batch_id,
            blob_urls=blob_urls,
            summary_url=summary_url,
            pdf_url=pdf_url,
            summary=summary,
            local_pdf_path=local_pdf_path,
        )


def _build_summary(
    batch_result: BatchResult,
    annotated_images: list[AnnotatedImage],
) -> dict[str, Any]:
    """Construye el dict de resumen diagnóstico."""
    defect_counts: dict[str, int] = {}
    all_scores: list[float] = []

    for ir in batch_result.image_results:
        for det in ir.detections:
            defect_counts[det.class_name] = defect_counts.get(det.class_name, 0) + 1
            all_scores.append(det.confidence)

    return {
        "timestamp": datetime.now(tz=timezone.utc).isoformat(),
        "batch_id": batch_result.batch_id,
        "total_images": len(batch_result.image_results),
        "images_with_defects": sum(1 for ai in annotated_images if ai.has_defects),
        "total_detections": sum(len(ir.detections) for ir in batch_result.image_results),
        "defect_counts_by_class": defect_counts,
        "avg_confidence": round(sum(all_scores) / len(all_scores), 4) if all_scores else 0.0,
        "total_inference_time_ms": batch_result.total_time_ms,
    }
