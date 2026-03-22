"""Cliente REST para el servicio de detección de defectos en PCB.

Lee la configuración de conexión desde .env (API_HOST, API_PORT, API_TIMEOUT)
y permite sobreescribir los valores por parámetro de constructor.

Estrategia de manejo de errores
---------------------------------
* analyze_image: lanza APIClientError en caso de fallo HTTP o de red.
* analyze_image_safe: nunca lanza; retorna un dict con status="error" y
  error_message poblado. Usar este método en bucles de procesamiento en lote
  para evitar que una imagen fallida aborte el lote completo.

Uso::

    from app.api_client import APIClient, APIClientError

    client = APIClient()
    result = client.analyze_image(image_bytes, filename="pcb.jpg")

    # Variante segura para uso en lote (nunca lanza):
    result = client.analyze_image_safe(image_bytes, filename="pcb.jpg")
    if result["status"] == "error":
        print(result["error_message"])
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
