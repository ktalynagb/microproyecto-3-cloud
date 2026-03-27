"""Gestión de Azure Blob Storage y generación de SAS URLs.

Sube imágenes al container de entrada para el Batch Endpoint, genera
SAS URLs temporales para las imágenes anotadas de salida, y descarga
el archivo JSONL de resultados producido por el Batch Endpoint.

Variables de entorno:
    AZURE_STORAGE_ACCOUNT       Nombre de la cuenta de storage
    AZURE_STORAGE_KEY           Clave de acceso (nunca se expone al frontend)
    AZURE_INPUT_CONTAINER       Container de entrada (default: azureml-blobstore-...)
    AZURE_OUTPUT_CONTAINER      Container de salida con resultados anotados
    AZURE_SAS_TTL_HOURS         TTL de SAS URLs en horas (default: 24)
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timedelta, timezone
from io import BytesIO
from typing import Any, Dict, List, Optional

LOG = logging.getLogger(__name__)

_STORAGE_ACCOUNT = os.environ.get("AZURE_STORAGE_ACCOUNT", "pcbmlworstorage428505ef7")
_INPUT_CONTAINER = os.environ.get(
    "AZURE_INPUT_CONTAINER",
    "azureml-blobstore-fa3e2152-1a09-4e81-acb4-3f701118ca5e",
)
_OUTPUT_CONTAINER = os.environ.get("AZURE_OUTPUT_CONTAINER", "pcb-results")
_SAS_TTL_HOURS = int(os.environ.get("AZURE_SAS_TTL_HOURS", "24"))


class BlobManager:
    """Gestiona uploads y SAS URLs en Azure Blob Storage.

    Parameters
    ----------
    account:
        Nombre de la cuenta de storage.
    key:
        Clave de acceso. **Nunca se expone al frontend.**
    input_container:
        Container donde se suben las imágenes de entrada.
    output_container:
        Container donde el Batch Endpoint deja los resultados.
    sas_ttl_hours:
        Tiempo de vida de las SAS URLs en horas.
    """

    def __init__(
        self,
        account: Optional[str] = None,
        key: Optional[str] = None,
        input_container: Optional[str] = None,
        output_container: Optional[str] = None,
        sas_ttl_hours: int = _SAS_TTL_HOURS,
    ) -> None:
        self.account = account or _STORAGE_ACCOUNT
        self.key = key or os.environ.get("AZURE_STORAGE_KEY", "")
        self.input_container = input_container or _INPUT_CONTAINER
        self.output_container = output_container or _OUTPUT_CONTAINER
        self.sas_ttl_hours = sas_ttl_hours
        self._service_client = None

    # ── Internal client ────────────────────────────────────────────────────

    def _get_service_client(self):
        """Lazy-init del BlobServiceClient."""
        if self._service_client is None:
            from azure.storage.blob import BlobServiceClient

            conn_str = (
                f"DefaultEndpointsProtocol=https;"
                f"AccountName={self.account};"
                f"AccountKey={self.key};"
                "EndpointSuffix=core.windows.net"
            )
            self._service_client = BlobServiceClient.from_connection_string(conn_str)
        return self._service_client

    # ── Upload ─────────────────────────────────────────────────────────────

    def upload_images(
        self,
        images: Dict[str, bytes],
        folder: str,
    ) -> str:
        """Sube múltiples imágenes a un sub-folder del container de entrada.

        Args:
            images: Dict de {filename: image_bytes}.
            folder: Nombre del sub-folder (normalmente el job_id).

        Returns:
            URL del folder en Blob Storage (sin SAS) para pasarle al Batch Endpoint.
        """
        client = self._get_service_client()
        container_client = client.get_container_client(self.input_container)

        for filename, data in images.items():
            blob_name = f"{folder}/{filename}"
            blob_client = container_client.get_blob_client(blob_name)
            ext = filename.rsplit(".", 1)[-1].lower()
            content_type = "image/png" if ext == "png" else "image/jpeg"
            from azure.storage.blob import ContentSettings

            blob_client.upload_blob(
                BytesIO(data),
                overwrite=True,
                content_settings=ContentSettings(content_type=content_type),
            )
            LOG.info("Imagen subida: %s/%s", self.input_container, blob_name)

        folder_url = (
            f"https://{self.account}.blob.core.windows.net"
            f"/{self.input_container}/{folder}"
        )
        return folder_url

    # ── SAS URLs ───────────────────────────────────────────────────────────

    def generate_sas_url(self, blob_name: str, container: Optional[str] = None) -> str:
        """Genera una SAS URL con TTL para un blob en el container de output.

        La Storage Key **nunca** se incluye en la URL ni se envía al frontend;
        solo se usa internamente para firmar el token SAS.

        Args:
            blob_name: Nombre del blob (ruta relativa dentro del container).
            container: Container destino. Usa output_container si no se indica.

        Returns:
            URL firmada con SAS válida por ``sas_ttl_hours`` horas.
        """
        from azure.storage.blob import (
            BlobSasPermissions,
            generate_blob_sas,
        )

        container_name = container or self.output_container
        expiry = datetime.now(tz=timezone.utc) + timedelta(hours=self.sas_ttl_hours)

        sas_token = generate_blob_sas(
            account_name=self.account,
            container_name=container_name,
            blob_name=blob_name,
            account_key=self.key,
            permission=BlobSasPermissions(read=True),
            expiry=expiry,
        )

        url = (
            f"https://{self.account}.blob.core.windows.net"
            f"/{container_name}/{blob_name}?{sas_token}"
        )
        LOG.debug("SAS URL generada para %s (expira %s)", blob_name, expiry.isoformat())
        return url

    def generate_sas_urls_for_folder(
        self,
        folder: str,
        container: Optional[str] = None,
    ) -> Dict[str, str]:
        """Genera SAS URLs para todos los blobs en un folder.

        Args:
            folder: Prefijo del folder dentro del container.
            container: Container a listar. Usa output_container si no se indica.

        Returns:
            Dict de {filename: sas_url}.
        """
        container_name = container or self.output_container
        client = self._get_service_client()
        container_client = client.get_container_client(container_name)

        urls: Dict[str, str] = {}
        try:
            for blob in container_client.list_blobs(name_starts_with=folder):
                filename = blob.name.split("/")[-1]
                urls[filename] = self.generate_sas_url(blob.name, container_name)
        except Exception as exc:  # noqa: BLE001
            LOG.error("Error listando blobs en %s/%s: %s", container_name, folder, exc)
            raise

        return urls

    # ── Download JSONL results ──────────────────────────────────────────────

    def download_jsonl(self, run_id: str, container: str) -> List[Dict[str, Any]]:
        """Descarga y parsea el archivo JSONL de resultados del Batch Endpoint.

        Azure ML guarda las predicciones en el container interno del workspace
        (workspaceblobstore) bajo la ruta ``pcb-inference-output/{run_id}/predictions.jsonl``.
        El container real se obtiene con ``ml_client.datastores.get_default().container_name``.

        Args:
            run_id: Identificador de la corrida (generado en submit_inference).
            container: Nombre del container del datastore por defecto de Azure ML.

        Returns:
            Lista de dicts, uno por imagen procesada.

        Raises:
            azure.core.exceptions.ResourceNotFoundError: Si el blob no existe aún.
            Exception: Cualquier otro error de red o storage.
        """
        from azure.storage.blob import BlobServiceClient

        blob_path = f"pcb-inference-output/{run_id}/predictions.jsonl"
        LOG.info("Descargando JSONL desde %s/%s", container, blob_path)

        conn_str = (
            f"DefaultEndpointsProtocol=https;"
            f"AccountName={self.account};"
            f"AccountKey={self.key};"
            "EndpointSuffix=core.windows.net"
        )
        blob_service = BlobServiceClient.from_connection_string(conn_str)
        blob_client = blob_service.get_blob_client(container=container, blob=blob_path)

        content = blob_client.download_blob().readall()

        records: List[Dict[str, Any]] = []
        for line in content.decode("utf-8").splitlines():
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    LOG.warning("Línea JSONL inválida ignorada: %s", exc)

        LOG.info("JSONL descargado: %d registros encontrados", len(records))
        return records
