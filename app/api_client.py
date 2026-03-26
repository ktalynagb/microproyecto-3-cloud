"""Clientes REST para el sistema de detección de defectos en PCB.

Contiene dos clientes:

APIClient
    Cliente directo para el servidor FastAPI de inferencia local
    (endpoint /predict).  Lee API_HOST, API_PORT, API_TIMEOUT desde .env.

AzureMLClient
    Cliente para el backend FastAPI que conecta con el Azure ML Batch
    Endpoint.  Lee AZURE_BACKEND_URL y BACKEND_API_KEY desde .env.

    Métodos principales:
        submit_inference(images)     Envía hasta 10 imágenes y retorna job_id.
        poll_results(job_id)         Consulta el estado del job (polling).
        get_download_links(job_id)   Retorna resultados completos con SAS URLs.

Estrategia de manejo de errores
---------------------------------
* analyze_image: lanza APIClientError en caso de fallo HTTP o de red.
* analyze_image_safe: nunca lanza; retorna un dict con status="error" y
  error_message poblado. Usar este método en bucles de procesamiento en lote
  para evitar que una imagen fallida aborte el lote completo.

Uso::

    from app.api_client import APIClient, APIClientError, AzureMLClient

    # Cliente local
    client = APIClient()
    result = client.analyze_image(image_bytes, filename="pcb.jpg")

    # Cliente Azure ML Batch
    az_client = AzureMLClient()
    job_id = az_client.submit_inference([img1_bytes, img2_bytes])
    status = az_client.poll_results(job_id)
    results = az_client.get_download_links(job_id)
"""
from __future__ import annotations

import logging
import os
from typing import Any, Dict, Optional

import requests
from dotenv import load_dotenv

load_dotenv()

LOG = logging.getLogger(__name__)

_EMPTY_RESULT: Dict[str, Any] = {
    "status": "error",
    "has_defects": False,
    "defects_summary": [],
    "processed_image_base64": "",
    "error_message": "",
}


class APIClientError(Exception):
    """Raised when the API client cannot connect or an HTTP call fails."""


class APIClient:
    """Cliente REST para el endpoint /predict del servidor FastAPI.

    Parameters
    ----------
    host:
        Hostname del servidor. Cae en API_HOST o 'localhost'.
    port:
        Puerto del servidor. Cae en API_PORT o 8000.
    timeout:
        Segundos para el timeout de la petición. Cae en API_TIMEOUT o 30.
    """

    def __init__(
        self,
        host: Optional[str] = None,
        port: Optional[int] = None,
        timeout: Optional[int] = None,
    ) -> None:
        """Inicializa el cliente con host, puerto y timeout.

        Args:
            host: Hostname del servidor.
            port: Puerto del servidor.
            timeout: Timeout en segundos para cada petición.
        """
        self.host: str = host or os.getenv("API_HOST", "localhost")
        self.port: int = port or int(os.getenv("API_PORT", "8000"))
        self.timeout: int = (
            timeout
            if timeout is not None
            else int(os.getenv("API_TIMEOUT", "30"))
        )
        self.base_url = f"http://{self.host}:{self.port}"
        LOG.info("APIClient configurado en %s", self.base_url)

    def analyze_image(
        self,
        image_bytes: bytes,
        filename: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Envía la imagen al endpoint /predict y retorna el dict de respuesta.

        Args:
            image_bytes: Bytes crudos de la imagen (JPG o PNG).
            filename: Nombre original del archivo (informativo).

        Returns:
            Dict con status, has_defects, defects_summary,
            processed_image_base64 y error_message.

        Raises:
            APIClientError: Si ocurre un error de red o HTTP.
        """
        url = f"{self.base_url}/predict"
        fname = filename or "image.jpg"
        # Detect MIME type from file extension
        ext = fname.rsplit(".", 1)[-1].lower() if "." in fname else ""
        mime_type = "image/png" if ext == "png" else "image/jpeg"
        files = {"file": (fname, image_bytes, mime_type)}

        try:
            response = requests.post(url, files=files, timeout=self.timeout)
            response.raise_for_status()
            return response.json()
        except requests.Timeout as exc:
            msg = (
                f"La petición al servidor excedió el timeout "
                f"({self.timeout}s). "
                "El servidor puede estar sobrecargado o la red es lenta."
            )
            LOG.error("Timeout en /predict: %s", exc)
            raise APIClientError(msg) from exc
        except requests.ConnectionError as exc:
            msg = (
                f"No se pudo conectar al servidor en {self.base_url}. "
                "Verifica que el servidor FastAPI esté activo."
            )
            LOG.error("ConnectionError en /predict: %s", exc)
            raise APIClientError(msg) from exc
        except requests.HTTPError as exc:
            msg = f"Error HTTP {exc.response.status_code}: {exc}"
            LOG.error("HTTPError en /predict: %s", exc)
            raise APIClientError(msg) from exc
        except Exception as exc:
            msg = f"Error inesperado al llamar /predict: {exc}"
            LOG.error("Error inesperado en APIClient: %s", exc)
            raise APIClientError(msg) from exc

    def analyze_image_safe(
        self,
        image_bytes: bytes,
        filename: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Versión segura de analyze_image: nunca lanza excepciones.

        Atrapa todos los errores posibles y los convierte en un dict con
        status="error" para uso en bucles de procesamiento en lote.

        Args:
            image_bytes: Bytes crudos de la imagen (JPG o PNG).
            filename: Nombre original del archivo (informativo).

        Returns:
            Dict con status, has_defects, defects_summary,
            processed_image_base64 y error_message. Si hubo un error,
            status="error" y error_message describe el problema.
        """
        try:
            return self.analyze_image(image_bytes, filename)
        except APIClientError as exc:
            return {**_EMPTY_RESULT, "error_message": str(exc)}
        except Exception as exc:
            LOG.exception("Error inesperado en analyze_image_safe: %s", exc)
            return {**_EMPTY_RESULT, "error_message": str(exc)}


# ── AzureMLClient ─────────────────────────────────────────────────────────────


class AzureMLClient:
    """Cliente para el backend FastAPI que conecta con Azure ML Batch Endpoint.

    Permite al frontend Streamlit:
      1. Enviar imágenes para inferencia (submit_inference)
      2. Consultar el estado del job asíncronamente (poll_results)
      3. Obtener los resultados con SAS URLs (get_download_links)

    Parameters
    ----------
    backend_url:
        URL base del backend FastAPI (sin barra final).
        Lee AZURE_BACKEND_URL o usa http://localhost:8080.
    api_key:
        API Key para el header X-API-Key.  Lee BACKEND_API_KEY.
    timeout:
        Timeout HTTP en segundos para cada petición.
    """

    def __init__(
        self,
        backend_url: Optional[str] = None,
        api_key: Optional[str] = None,
        timeout: int = 60,
    ) -> None:
        self.base_url: str = (
            (backend_url or os.getenv("AZURE_BACKEND_URL", "http://localhost:8080"))
            .rstrip("/")
        )
        self.api_key: str = api_key or os.getenv("BACKEND_API_KEY", "")
        self.timeout = timeout
        LOG.info("AzureMLClient configurado en %s", self.base_url)

    @property
    def _headers(self) -> Dict[str, str]:
        h: Dict[str, str] = {}
        if self.api_key:
            h["X-API-Key"] = self.api_key
        return h

    # ── submit_inference ───────────────────────────────────────────────────

    def submit_inference(self, images: list[bytes], filenames: Optional[list[str]] = None) -> str:
        """Envía hasta 10 imágenes al backend y retorna el job_id.

        Args:
            images: Lista de bytes de cada imagen (JPG o PNG).
            filenames: Nombres de archivo opcionales para cada imagen.

        Returns:
            job_id (str) para usar en poll_results / get_download_links.

        Raises:
            APIClientError: Si ocurre un error de red o HTTP.
        """
        if not images:
            raise APIClientError("Se requiere al menos una imagen")
        if len(images) > 10:
            raise APIClientError("Máximo 10 imágenes por lote")

        names = filenames or [f"image_{i + 1}.jpg" for i in range(len(images))]
        files = []
        for i, (data, name) in enumerate(zip(images, names)):
            ext = name.rsplit(".", 1)[-1].lower() if "." in name else ""
            mime = "image/png" if ext == "png" else "image/jpeg"
            files.append(("files", (name, data, mime)))

        url = f"{self.base_url}/api/v1/infer"
        try:
            resp = requests.post(
                url,
                headers=self._headers,
                files=files,
                timeout=self.timeout,
            )
            resp.raise_for_status()
            return resp.json()["job_id"]
        except requests.Timeout as exc:
            raise APIClientError(f"Timeout enviando imágenes ({self.timeout}s)") from exc
        except requests.ConnectionError as exc:
            raise APIClientError(
                f"No se pudo conectar al backend en {self.base_url}"
            ) from exc
        except requests.HTTPError as exc:
            detail = ""
            try:
                detail = exc.response.json().get("detail", "")
            except Exception:
                pass
            raise APIClientError(
                f"Error HTTP {exc.response.status_code}: {detail or exc}"
            ) from exc
        except Exception as exc:
            raise APIClientError(f"Error inesperado al enviar imágenes: {exc}") from exc

    # ── poll_results ───────────────────────────────────────────────────────

    def poll_results(self, job_id: str) -> Dict[str, Any]:
        """Consulta el estado actual del job.

        Args:
            job_id: ID retornado por submit_inference.

        Returns:
            Dict con keys: job_id, status, created_at, updated_at, message.
            El campo ``status`` puede ser: submitted | running | completed | failed.

        Raises:
            APIClientError: Si ocurre un error de red o HTTP.
        """
        url = f"{self.base_url}/api/v1/jobs/{job_id}"
        try:
            resp = requests.get(url, headers=self._headers, timeout=self.timeout)
            resp.raise_for_status()
            return resp.json()
        except requests.Timeout as exc:
            raise APIClientError(f"Timeout consultando job {job_id}") from exc
        except requests.ConnectionError as exc:
            raise APIClientError(
                f"No se pudo conectar al backend en {self.base_url}"
            ) from exc
        except requests.HTTPError as exc:
            raise APIClientError(
                f"Error HTTP {exc.response.status_code} al consultar job"
            ) from exc
        except Exception as exc:
            raise APIClientError(f"Error inesperado al consultar job: {exc}") from exc

    # ── get_download_links ─────────────────────────────────────────────────

    def get_download_links(self, job_id: str) -> Dict[str, Any]:
        """Descarga los resultados completos con SAS URLs cuando el job está completo.

        Args:
            job_id: ID retornado por submit_inference.

        Returns:
            Dict con la estructura::

                {
                    "job_id": "...",
                    "status": "completed",
                    "timestamp": "...",
                    "processing_time_ms": 398,
                    "images": [
                        {
                            "filename": "image.jpg",
                            "has_defects": true,
                            "detection_count": 8,
                            "confidence_avg": 0.85,
                            "download_url": "https://...?sig=...",
                            "detections": [...]
                        }
                    ],
                    "summary": {
                        "total_images": 4,
                        "defective_images": 3,
                        "total_defects": 11
                    }
                }

        Raises:
            APIClientError: Si el job aún no está completo o hay un error.
        """
        url = f"{self.base_url}/api/v1/jobs/{job_id}/results"
        try:
            resp = requests.get(url, headers=self._headers, timeout=self.timeout)
            resp.raise_for_status()
            return resp.json()
        except requests.Timeout as exc:
            raise APIClientError(f"Timeout descargando resultados del job {job_id}") from exc
        except requests.ConnectionError as exc:
            raise APIClientError(
                f"No se pudo conectar al backend en {self.base_url}"
            ) from exc
        except requests.HTTPError as exc:
            detail = ""
            try:
                detail = exc.response.json().get("detail", "")
            except Exception:
                pass
            raise APIClientError(
                f"Error HTTP {exc.response.status_code}: {detail or exc}"
            ) from exc
        except Exception as exc:
            raise APIClientError(f"Error inesperado al obtener resultados: {exc}") from exc

