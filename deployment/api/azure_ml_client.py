"""Cliente para Azure ML Managed Batch Endpoint.

Envía trabajos de inferencia al Batch Endpoint y consulta su estado.

Variables de entorno:
    AZURE_ML_BATCH_ENDPOINT_URL   URL completa del endpoint (con /jobs)
    AZURE_ML_API_KEY              API Key del endpoint
"""

from __future__ import annotations

import json
import logging
import os
import time
from typing import Any, Dict, List, Optional

import requests

LOG = logging.getLogger(__name__)

_DEFAULT_BATCH_URL = (
    "https://pcb-batch-inference.centralus.inference.ml.azure.com/jobs"
)
_MAX_RETRIES = 3
_RETRY_BACKOFF = 2  # segundos


class AzureMLBatchClient:
    """Cliente para el Batch Endpoint de Azure ML.

    Parameters
    ----------
    endpoint_url:
        URL del endpoint de jobs.  Toma AZURE_ML_BATCH_ENDPOINT_URL si no
        se proporciona.
    api_key:
        API Key.  Toma AZURE_ML_API_KEY si no se proporciona.
    timeout:
        Timeout HTTP en segundos.
    """

    def __init__(
        self,
        endpoint_url: Optional[str] = None,
        api_key: Optional[str] = None,
        timeout: int = 30,
    ) -> None:
        self.endpoint_url: str = endpoint_url or os.environ.get(
            "AZURE_ML_BATCH_ENDPOINT_URL", _DEFAULT_BATCH_URL
        )
        self.api_key: str = api_key or os.environ.get("AZURE_ML_API_KEY", "")
        self.timeout = timeout

    # ── Headers ────────────────────────────────────────────────────────────

    @property
    def _headers(self) -> Dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    # ── Submit job ─────────────────────────────────────────────────────────

    def submit_job(self, input_data_url: str) -> str:
        """Envía un trabajo al Batch Endpoint y retorna el job_id.

        Args:
            input_data_url: URL del blob con las imágenes de entrada.

        Returns:
            ID del job creado.

        Raises:
            requests.HTTPError: Si el endpoint devuelve un error HTTP.
        """
        payload = {
            "properties": {
                "InputData": {
                    "input_data": {
                        "uri": input_data_url,
                        "job_input_type": "UriFolder",
                    }
                }
            }
        }

        for attempt in range(1, _MAX_RETRIES + 1):
            try:
                resp = requests.post(
                    self.endpoint_url,
                    headers=self._headers,
                    json=payload,
                    timeout=self.timeout,
                )
                resp.raise_for_status()
                data = resp.json()
                job_id: str = data.get("name", data.get("id", ""))
                LOG.info("Job enviado: %s", job_id)
                return job_id
            except requests.RequestException as exc:
                LOG.warning("Intento %d/%d fallido: %s", attempt, _MAX_RETRIES, exc)
                if attempt == _MAX_RETRIES:
                    raise
                time.sleep(_RETRY_BACKOFF * attempt)

        return ""  # unreachable

    # ── Job status ─────────────────────────────────────────────────────────

    def get_job_status(self, job_id: str) -> Dict[str, Any]:
        """Consulta el estado de un job.

        Args:
            job_id: ID del job.

        Returns:
            Dict con keys: job_id, status, created_at, updated_at, message.
        """
        url = f"{self.endpoint_url}/{job_id}"

        for attempt in range(1, _MAX_RETRIES + 1):
            try:
                resp = requests.get(
                    url,
                    headers=self._headers,
                    timeout=self.timeout,
                )
                resp.raise_for_status()
                data = resp.json()
                props = data.get("properties", data)
                raw_status: str = props.get("status", props.get("jobStatus", "unknown"))
                normalized = _normalize_status(raw_status)
                return {
                    "job_id": job_id,
                    "status": normalized,
                    "created_at": props.get("creationContext", {}).get("createdAt"),
                    "updated_at": props.get("creationContext", {}).get("lastModifiedAt"),
                    "message": props.get("statusMessage"),
                }
            except requests.RequestException as exc:
                LOG.warning("Intento %d/%d fallido: %s", attempt, _MAX_RETRIES, exc)
                if attempt == _MAX_RETRIES:
                    raise
                time.sleep(_RETRY_BACKOFF * attempt)

        return {}  # unreachable

    # ── Get output URL ─────────────────────────────────────────────────────

    def get_output_url(self, job_id: str) -> Optional[str]:
        """Retorna la URL del output folder del job (si está disponible).

        Args:
            job_id: ID del job completado.

        Returns:
            URL del folder de output o None si aún no está disponible.
        """
        url = f"{self.endpoint_url}/{job_id}"
        try:
            resp = requests.get(url, headers=self._headers, timeout=self.timeout)
            resp.raise_for_status()
            data = resp.json()
            props = data.get("properties", data)
            outputs = props.get("outputs", {})
            # El output default se llama "score" o "default"
            for key in ("score", "default"):
                if key in outputs:
                    return outputs[key].get("uri")
        except requests.RequestException as exc:
            LOG.warning("Error obteniendo output URL: %s", exc)
        return None

    # ── Download JSONL results ─────────────────────────────────────────────

    def download_results(self, results_url: str) -> List[Dict[str, Any]]:
        """Descarga y parsea el archivo JSONL de resultados.

        Args:
            results_url: URL directa al archivo .jsonl.

        Returns:
            Lista de dicts, uno por imagen.
        """
        for attempt in range(1, _MAX_RETRIES + 1):
            try:
                resp = requests.get(
                    results_url,
                    headers=self._headers,
                    timeout=60,
                )
                resp.raise_for_status()
                records = []
                for line in resp.text.splitlines():
                    line = line.strip()
                    if line:
                        try:
                            records.append(json.loads(line))
                        except json.JSONDecodeError as exc:
                            LOG.warning("Línea JSONL inválida: %s", exc)
                return records
            except requests.RequestException as exc:
                LOG.warning("Intento %d/%d fallido: %s", attempt, _MAX_RETRIES, exc)
                if attempt == _MAX_RETRIES:
                    raise
                time.sleep(_RETRY_BACKOFF * attempt)

        return []  # unreachable


# ── Helpers ───────────────────────────────────────────────────────────────────

def _normalize_status(raw: str) -> str:
    """Normaliza el estado devuelto por Azure ML a un valor canónico."""
    mapping = {
        "notstarted": "submitted",
        "queued": "submitted",
        "preparing": "running",
        "running": "running",
        "finalizing": "running",
        "completed": "completed",
        "succeeded": "completed",
        "failed": "failed",
        "canceled": "failed",
        "cancelled": "failed",
    }
    return mapping.get(raw.lower(), raw.lower())
