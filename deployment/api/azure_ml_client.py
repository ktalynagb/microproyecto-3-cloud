"""Cliente para Azure ML Managed Batch Endpoint.

Envía trabajos de inferencia al Batch Endpoint y consulta su estado.

Autenticación (en orden de prioridad):
    1. IMDS – Azure Instance Metadata Service (Managed Identity en ACI)
    2. Azure CLI – credenciales montadas en ~/.azure (desarrollo local)
    3. AZURE_ML_API_KEY – clave estática como fallback

Variables de entorno:
    AZURE_ML_BATCH_ENDPOINT_URL   URL completa del endpoint (con /jobs)
    AZURE_ML_API_KEY              API Key estática (fallback)
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import time
from typing import Any, Dict, List, Optional

import requests

LOG = logging.getLogger(__name__)

_DEFAULT_BATCH_URL = (
    "https://pcb-batch-inference.centralus.inference.ml.azure.com/jobs"
)
_MAX_RETRIES = 3
_RETRY_BACKOFF = 2  # segundos
_TOKEN_BUFFER_SECS = 300  # refrescar el token 5 min antes de que expire
_IMDS_URL = "http://169.254.169.254/metadata/identity/oauth2/token"
_AZURE_RESOURCE = "https://management.azure.com/"


class AzureMLBatchClient:
    """Cliente para el Batch Endpoint de Azure ML.

    Parameters
    ----------
    endpoint_url:
        URL del endpoint de jobs.  Toma AZURE_ML_BATCH_ENDPOINT_URL si no
        se proporciona.
    api_key:
        API Key estática de fallback.  Toma AZURE_ML_API_KEY si no se
        proporciona.
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
        # ✅ Intentar obtener el token del .env primero
        self.api_key: str = api_key or os.environ.get("AZURE_ML_API_KEY", "")
        self.timeout = timeout
        self._token_cache: Optional[str] = None
        self._token_expiry: float = 0.0

    # ── Token management ───────────────────────────────────────────────────

    def _get_fresh_token(self) -> str:
        """Devuelve un token Bearer válido, refrescándolo si es necesario.

        Orden de prioridad:
        1. IMDS (Managed Identity en Azure Container Instances)
        2. Azure CLI (desarrollo local con ~/.azure montado)
        3. AZURE_ML_API_KEY estática (fallback)

        Returns:
            Token Bearer válido.

        Raises:
            RuntimeError: Si no hay ninguna fuente de credenciales disponible.
        """
        now = time.time()
        if self._token_cache and now < self._token_expiry - _TOKEN_BUFFER_SECS:
            return self._token_cache

        token = self._fetch_imds_token() or self._fetch_cli_token()
        if token:
            return token

        if self.api_key:
            LOG.debug("Usando AZURE_ML_API_KEY estática")
            return self.api_key

        raise RuntimeError(
            "No hay credenciales de Azure disponibles: "
            "IMDS, Azure CLI y AZURE_ML_API_KEY han fallado"
        )

    def _fetch_imds_token(self) -> Optional[str]:
        """Obtiene un token desde Azure IMDS (Managed Identity en ACI)."""
        try:
            resp = requests.get(
                _IMDS_URL,
                params={
                    "api-version": "2018-02-01",
                    "resource": _AZURE_RESOURCE,
                },
                headers={"Metadata": "true"},
                timeout=2,
            )
            resp.raise_for_status()
            data = resp.json()
            token: Optional[str] = data.get("access_token")
            expires_in = int(data.get("expires_in", 3600))
            if token:
                self._token_cache = token
                self._token_expiry = time.time() + expires_in
                LOG.info("Token obtenido desde IMDS (Managed Identity)")
                return token
        except Exception as exc:  # noqa: BLE001
            LOG.debug("IMDS no disponible: %s", exc)
        return None

    def _fetch_cli_token(self) -> Optional[str]:
        """Obtiene un token desde Azure CLI (desarrollo local)."""
        try:
            result = subprocess.run(
                [
                    "az", "account", "get-access-token",
                    "--resource", _AZURE_RESOURCE,
                    "--query", "accessToken",
                    "-o", "tsv",
                ],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode == 0 and result.stdout.strip():
                token = result.stdout.strip()
                self._token_cache = token
                self._token_expiry = time.time() + 3600
                LOG.info("Token obtenido desde Azure CLI")
                return token
            if result.stderr.strip():
                LOG.debug("Azure CLI stderr: %s", result.stderr.strip())
        except Exception as exc:  # noqa: BLE001
            LOG.debug("Azure CLI no disponible: %s", exc)
        return None

    # ── Headers ────────────────────────────────────────────────────────────

    @property
    def _headers(self) -> Dict[str, str]:
        return {
            "Authorization": f"Bearer {self._get_fresh_token()}",
            "Content-Type": "application/json",
        }

    # ── Submit job ────────────────────────────────────��────────────────────

    def submit_job(self, input_data_url: str) -> str:
        """Envía un trabajo al Batch Endpoint y retorna el job_id."""
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

        return ""

    # ── Job status ─────────────────────────────────────────────────────────

    def get_job_status(self, job_id: str) -> Dict[str, Any]:
        """Consulta el estado de un job."""
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

        return {}

    # ── Get output URL ─────────────────────────────────────────���───────────

    def get_output_url(self, job_id: str) -> Optional[str]:
        """Retorna la URL del output folder del job."""
        url = f"{self.endpoint_url}/{job_id}"
        try:
            resp = requests.get(url, headers=self._headers, timeout=self.timeout)
            resp.raise_for_status()
            data = resp.json()
            props = data.get("properties", data)
            outputs = props.get("outputs", {})
            for key in ("score", "default"):
                if key in outputs:
                    return outputs[key].get("uri")
        except requests.RequestException as exc:
            LOG.warning("Error obteniendo output URL: %s", exc)
        return None

    # ── Download JSONL results ─────────────────────────────────────────────

    def download_results(self, results_url: str) -> List[Dict[str, Any]]:
        """Descarga y parsea el archivo JSONL de resultados."""
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

        return []


def _normalize_status(raw: str) -> str:
    """Normaliza el estado devuelto por Azure ML."""
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