"""ETAPA 5: Entrega (delivery).

Genera URLs SAS temporales para que el técnico descargue imágenes anotadas
y el PDF de reporte directamente desde Azure Blob Storage.
Todo el ciclo (ingesta → predicción → visualización → descarga) ocurre en ≤1 min.

Uso como módulo:
    from delivery import DeliveryService
    svc = DeliveryService()
    links = svc.generate_download_links(export_result)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any

from blob_exporter import ExportResult
from config import config
from logger import get_logger

logger = get_logger("delivery")


@dataclass
class DownloadLinks:
    """URLs SAS para descarga de resultados de un lote."""

    batch_id: str
    image_links: dict[str, str]   # {filename: sas_url}
    summary_link: str | None
    pdf_link: str | None
    local_pdf_path: str | None
    expires_at: str


class DeliveryService:
    """Genera URLs SAS para descarga temporal desde Azure Blob Storage."""

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

    def _generate_sas(self, blob_name: str, expiry: datetime) -> str:
        """Genera una URL SAS para un blob específico."""
        from azure.storage.blob import (
            BlobSasPermissions,
            BlobServiceClient,
            generate_blob_sas,
        )

        sas_token = generate_blob_sas(
            account_name=self.account,
            container_name=self.container,
            blob_name=blob_name,
            account_key=self.key,
            permission=BlobSasPermissions(read=True),
            expiry=expiry,
        )
        return (
            f"https://{self.account}.blob.core.windows.net"
            f"/{self.container}/{blob_name}?{sas_token}"
        )

    def generate_download_links(self, export_result: ExportResult) -> DownloadLinks:
        """Genera URLs SAS para todas las imágenes, el JSON y el PDF del lote."""
        batch_id = export_result.batch_id
        expiry = datetime.now(tz=timezone.utc) + timedelta(hours=self.ttl_hours)

        image_links: dict[str, str] = {}
        for filename in export_result.blob_urls:
            blob_name = f"{batch_id}/images/{filename}"
            try:
                url = self._generate_sas(blob_name, expiry)
                image_links[filename] = url
            except Exception as exc:  # noqa: BLE001
                logger.error(f"Error generando SAS para {filename}: {exc}")
                image_links[filename] = export_result.blob_urls.get(filename, "")

        summary_link: str | None = None
        if export_result.summary_url:
            try:
                summary_link = self._generate_sas(f"{batch_id}/summary.json", expiry)
            except Exception as exc:  # noqa: BLE001
                logger.error(f"Error generando SAS para summary.json: {exc}")
                summary_link = export_result.summary_url

        pdf_link: str | None = None
        if export_result.pdf_url:
            try:
                pdf_link = self._generate_sas(f"{batch_id}/report.pdf", expiry)
            except Exception as exc:  # noqa: BLE001
                logger.error(f"Error generando SAS para report.pdf: {exc}")
                pdf_link = export_result.pdf_url

        links = DownloadLinks(
            batch_id=batch_id,
            image_links=image_links,
            summary_link=summary_link,
            pdf_link=pdf_link,
            local_pdf_path=export_result.local_pdf_path,
            expires_at=expiry.isoformat(),
        )
        logger.info(
            f"URLs SAS generadas para lote {batch_id}, expiran: {expiry.isoformat()}",
            extra={"batch_id": batch_id, "stage": "delivery"},
        )
        return links
