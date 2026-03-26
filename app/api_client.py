"""Cliente HTTP para la PCB Defect Detection API.

Encapsula todas las llamadas REST a la API FastAPI con:
  - Retry con backoff exponencial
  - Polling con barra de progreso
  - Manejo de errores
"""

from __future__ import annotations

import time
from typing import Any, Callable

import requests


class APIError(Exception):
    """Error de la API con código HTTP."""

    def __init__(self, status_code: int, detail: str) -> None:
        self.status_code = status_code
        self.detail = detail
        super().__init__(f"HTTP {status_code}: {detail}")


class PCBApiClient:
    """Cliente para el backend FastAPI de detección de defectos en PCB."""

    def __init__(
        self,
        base_url: str = "http://localhost:8000",
        api_key: str = "changeme-secret-key",
        timeout: int = 30,
        max_retries: int = 3,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._headers = {"X-API-Key": api_key}
        self._timeout = timeout
        self._max_retries = max_retries

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _request(
        self,
        method: str,
        path: str,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Realiza una petición HTTP con reintentos y backoff exponencial."""
        url = f"{self._base_url}{path}"
        last_exc: Exception | None = None

        for attempt in range(self._max_retries):
            try:
                resp = requests.request(
                    method,
                    url,
                    headers=self._headers,
                    timeout=self._timeout,
                    **kwargs,
                )
                if resp.status_code == 403:
                    raise APIError(403, "API Key inválida.")
                if resp.status_code == 404:
                    raise APIError(404, "Recurso no encontrado.")
                if resp.status_code >= 500:
                    raise APIError(resp.status_code, resp.text)
                resp.raise_for_status()
                return resp.json()
            except APIError:
                raise
            except requests.exceptions.ConnectionError as exc:
                last_exc = exc
                wait = 2 ** attempt
                time.sleep(wait)
            except Exception as exc:
                raise APIError(0, str(exc)) from exc

        raise APIError(0, f"No se pudo conectar tras {self._max_retries} intentos: {last_exc}")

    # ------------------------------------------------------------------
    # Endpoints
    # ------------------------------------------------------------------

    def health(self) -> dict[str, str]:
        """Verifica que la API esté disponible."""
        return self._request("GET", "/health")

    def submit_batch(self, image_paths: list[str]) -> str:
        """
        Envía imágenes al pipeline batch.

        Parámetros
        ----------
        image_paths : list[str]
            Rutas locales a los archivos de imagen.

        Retorna
        -------
        str
            job_id para hacer polling.
        """
        files = []
        try:
            for path in image_paths:
                with open(path, "rb") as f:
                    import os
                    filename = os.path.basename(path)
                    files.append(("files", (filename, f.read(), "image/jpeg")))

            resp = self._request_multipart("/inference/batch", files=files)
            return resp["job_id"]
        except APIError:
            raise
        except Exception as exc:
            raise APIError(0, str(exc)) from exc

    def _request_multipart(
        self, path: str, files: list[tuple[Any, ...]]
    ) -> dict[str, Any]:
        """POST multipart/form-data con reintentos."""
        url = f"{self._base_url}{path}"
        last_exc: Exception | None = None

        for attempt in range(self._max_retries):
            try:
                resp = requests.post(
                    url,
                    headers=self._headers,
                    files=files,
                    timeout=120,  # más tiempo para subir imágenes
                )
                if resp.status_code == 403:
                    raise APIError(403, "API Key inválida.")
                if resp.status_code >= 500:
                    raise APIError(resp.status_code, resp.text)
                resp.raise_for_status()
                return resp.json()
            except APIError:
                raise
            except requests.exceptions.ConnectionError as exc:
                last_exc = exc
                time.sleep(2 ** attempt)
            except Exception as exc:
                raise APIError(0, str(exc)) from exc

        raise APIError(0, f"Error de conexión: {last_exc}")

    def get_status(self, job_id: str) -> dict[str, Any]:
        """Consulta el estado de un job."""
        return self._request("GET", f"/inference/status/{job_id}")

    def get_results(self, job_id: str) -> list[dict[str, Any]]:
        """Descarga los resultados de un job completado."""
        data = self._request("GET", f"/inference/results/{job_id}")
        return data.get("results", [])

    def get_sas_url(
        self, container: str, blob: str, expiry_hours: int = 1
    ) -> str:
        """Genera una SAS URL temporal para un blob."""
        data = self._request(
            "GET",
            "/storage/sas",
            params={"container": container, "blob": blob, "expiry_hours": expiry_hours},
        )
        return data["url"]

    def list_jobs(self) -> list[dict[str, Any]]:
        """Lista todos los jobs registrados en la API."""
        data = self._request("GET", "/jobs")
        return data.get("jobs", [])

    # ------------------------------------------------------------------
    # Polling con callback de progreso
    # ------------------------------------------------------------------

    def wait_for_job(
        self,
        job_id: str,
        poll_interval: float = 5.0,
        timeout: float = 600.0,
        on_progress: Callable[[float, str], None] | None = None,
    ) -> list[dict[str, Any]]:
        """
        Hace polling hasta que el job termine.

        Parámetros
        ----------
        job_id : str
        poll_interval : float
            Segundos entre consultas.
        timeout : float
            Segundos máximos de espera.
        on_progress : callable(progress: float, message: str) -> None
            Callback llamado en cada polling con el progreso (0.0–1.0).

        Retorna
        -------
        list[dict]
            Lista de resultados de imágenes.

        Lanza
        -----
        TimeoutError
            Si el job no termina en ``timeout`` segundos.
        APIError
            Si el job falla.
        """
        deadline = time.time() + timeout
        while time.time() < deadline:
            status_data = self.get_status(job_id)
            status = status_data.get("status", "unknown")
            progress = float(status_data.get("progress", 0.0))
            message = status_data.get("message", "")

            if on_progress:
                on_progress(progress, message)

            if status == "completed":
                return self.get_results(job_id)
            if status == "failed":
                raise APIError(0, f"Job fallido: {message}")

            time.sleep(poll_interval)

        raise TimeoutError(f"Job {job_id} no terminó en {timeout}s.")
