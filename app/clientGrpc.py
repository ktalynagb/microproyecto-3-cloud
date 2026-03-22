"""gRPC client for the AiVsReal image classification service.

Reads connection settings from .env (GRPC_SERVER_HOST, GRPC_SERVER_PORT,
GRPC_TIMEOUT) and allows overriding them via constructor parameters.

Error handling strategy
-----------------------
* Connection errors (server unreachable, channel readiness timeout) raise
  :class:`GRPCClientError` immediately from ``__init__`` so the caller knows
  the client is unusable.
* Per-RPC failures are mapped to friendly human-readable messages depending
  on the gRPC status code:

  - ``DEADLINE_EXCEEDED`` → timeout message
  - ``UNAVAILABLE``       → server unreachable message
  - ``INVALID_ARGUMENT``  → bad image payload message
  - ``INTERNAL``          → unexpected server-side error
  - other codes           → generic gRPC error message with code name

* :meth:`classify_image` raises :class:`GRPCClientError` on failure so the
  caller can handle it explicitly.
* :meth:`classify_image_safe` never raises; instead it returns a result dict
  with ``status="error"`` and ``error_message`` populated.  Use this method
  in batch-processing loops to prevent a single failed image from aborting the
  entire batch.

Usage::

    from app.clientGrpc import GRPCClient, GRPCClientError

    client = GRPCClient()
    result = client.classify_image(image_bytes, filename="photo.jpg")
    client.close()

    # Safe variant for batch use (never raises):
    result = client.classify_image_safe(image_bytes, filename="photo.jpg")
    if result["status"] == "error":
        print(result["error_message"])
"""
from __future__ import annotations

import logging
import os
import sys
import uuid
from typing import Any, Dict, Optional

import grpc
from dotenv import load_dotenv

# Add proto/generated to sys.path so the generated stubs can be imported
# regardless of the working directory.
_PROTO_GENERATED = os.path.join(
    os.path.dirname(__file__), "..", "proto", "generated"
)
if _PROTO_GENERATED not in sys.path:
    sys.path.insert(0, _PROTO_GENERATED)

try:
    import inference_pb2
    import inference_pb2_grpc
except ImportError as _exc:
    raise ImportError(
        "gRPC stubs not found. Run 'make proto-gen' to generate them from "
        "proto/inference.proto."
    ) from _exc

load_dotenv()

LOG = logging.getLogger(__name__)

# Friendly messages per gRPC status code (used by _grpc_error_message).
_GRPC_STATUS_MESSAGES: Dict[grpc.StatusCode, str] = {
    grpc.StatusCode.DEADLINE_EXCEEDED: (
        "La llamada excedió el timeout configurado. "
        "El servidor puede estar sobrecargado o la red es lenta."
    ),
    grpc.StatusCode.UNAVAILABLE: (
        "El servidor gRPC no está disponible. "
        "Verifique que el servicio de inferencia esté activo."
    ),
    grpc.StatusCode.INVALID_ARGUMENT: (
        "El payload de la imagen es inválido. "
        "Asegúrese de que el archivo sea JPG o PNG y no esté corrupto."
    ),
    grpc.StatusCode.INTERNAL: (
        "El servidor encontró un error interno inesperado. "
        "Revise los logs del servicio de inferencia."
    ),
    grpc.StatusCode.CANCELLED: (
        "La llamada gRPC fue cancelada antes de completarse."
    ),
    grpc.StatusCode.RESOURCE_EXHAUSTED: (
        "El servidor no tiene recursos suficientes para procesar la solicitud."
    ),
}


def _grpc_error_message(rpc_err: grpc.RpcError) -> str:
    """Return a human-readable message for a :class:`grpc.RpcError`.

    If the status code has a predefined friendly message it is returned;
    otherwise a generic message with the code name is produced.
    """
    try:
        code: grpc.StatusCode = rpc_err.code()  # type: ignore[attr-defined]
    except Exception:
        return f"Error gRPC desconocido: {rpc_err}"

    friendly = _GRPC_STATUS_MESSAGES.get(code)
    if friendly:
        return friendly

    try:
        details = rpc_err.details()  # type: ignore[attr-defined]
    except Exception:
        details = str(rpc_err)

    return f"Error gRPC [{code.name}]: {details}"


class GRPCClientError(Exception):
    """Raised when the gRPC client cannot connect or an RPC call fails."""


class GRPCClient:
    """Reusable gRPC client for the AiVsRealClassifier inference service.

    Parameters
    ----------
    host:
        Server hostname. Falls back to ``GRPC_SERVER_HOST`` env var or
        ``"localhost"``.
    port:
        Server port. Falls back to ``GRPC_SERVER_PORT`` env var or ``50051``.
    timeout:
        Seconds to wait for channel readiness and per-RPC deadline. Falls back
        to ``GRPC_TIMEOUT`` env var or ``5``.
    """

    def __init__(
        self,
        host: Optional[str] = None,
        port: Optional[int] = None,
        timeout: Optional[int] = None,
    ) -> None:
        """Connect to the gRPC server using provided or environment settings.

        Args:
            host: Server hostname. Falls back to GRPC_SERVER_HOST env var
                or 'localhost'.
            port: Server port. Falls back to GRPC_SERVER_PORT env var
                or 50051.
            timeout: Seconds for channel readiness and per-RPC deadline.
                Falls back to GRPC_TIMEOUT env var or 5.

        Raises:
            GRPCClientError: If the server is not reachable within timeout.
        """
        self.host: str = host or os.getenv("GRPC_SERVER_HOST", "localhost")
        self.port: int = port or int(os.getenv("GRPC_SERVER_PORT", "50051"))
        self.timeout: int = (
            timeout
            if timeout is not None
            else int(os.getenv("GRPC_TIMEOUT", "5"))
        )
        self._channel: Optional[grpc.Channel] = None
        self._stub = None
        self._connect()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    # Maximum message size accepted/sent by the channel (50 MB).
    # The default gRPC limit is 4 MB which causes RESOURCE_EXHAUSTED for
    # large image payloads.
    _MAX_MSG_BYTES = 50 * 1024 * 1024

    def _connect(self) -> None:
        """Establish the gRPC channel and create the stub.

        Raises:
            GRPCClientError: If the channel cannot be established within
                the configured timeout, or if a connection error occurs.
        """
        target = f"{self.host}:{self.port}"
        try:
            self._channel = grpc.insecure_channel(
                target,
                options=[
                    ("grpc.max_send_message_length", self._MAX_MSG_BYTES),
                    (
                        "grpc.max_receive_message_length",
                        self._MAX_MSG_BYTES,
                    ),
                ],
            )
            ready_future = grpc.channel_ready_future(self._channel)
            ready_future.result(timeout=self.timeout)
            self._stub = inference_pb2_grpc.AiVsRealClassifierStub(
                self._channel
            )
            LOG.info("Connected to gRPC server at %s", target)
        except grpc.FutureTimeoutError as exc:
            LOG.error(
                "Timeout waiting for gRPC channel to become ready at %s "
                "(timeout=%ss)",
                target,
                self.timeout,
            )
            raise GRPCClientError(
                f"Timeout conectando al servidor gRPC en {target} "
                f"(timeout={self.timeout}s). "
                "Verifique que el servidor esté activo."
            ) from exc
        except grpc.RpcError as exc:
            LOG.error("gRPC error connecting to %s: %s", target, exc)
            raise GRPCClientError(
                f"Error gRPC al conectar con {target}: "
                f"{_grpc_error_message(exc)}"
            ) from exc
        except Exception as exc:
            LOG.exception("Could not connect to gRPC server at %s", target)
            raise GRPCClientError(
                f"Error connecting to gRPC server at {target}: {exc}"
            ) from exc

    def _parse_response(self, response: Any) -> Dict[str, Any]:
        """Convert a ``ClassificationResponse`` proto message to a plain dict.

        The returned dict uses the same field names expected by
        :class:`app.batch_upload.BatchImage` and
        :class:`app.result_table.ResultsTableBuilder` so it can be fed
        directly into the GUI layer or a CSV export.
        """
        metrics = response.metrics
        error_msg = response.error_message if response.error_message else None
        return {
            "image_id": response.image_id,
            "status": "ok" if response.status == inference_pb2.OK else "error",
            "predicted_label": response.predicted_label or None,
            "confidence": float(response.confidence),
            "prob_ai": float(response.prob_ai),
            # GUI / CSV schema uses "prob_real"; server returns "prob_human"
            "prob_real": float(response.prob_human),
            "preprocess_time_ms": (
                metrics.preprocess_time_ms if metrics else None
            ),
            "inference_time_ms": (
                metrics.inference_time_ms if metrics else None
            ),
            "error_message": error_msg,
        }

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def classify_image(
        self,
        image_bytes: bytes,
        filename: Optional[str] = None,
        image_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Send image bytes to the server and return a parsed result dict.

        Parameters
        ----------
        image_bytes:
            Raw bytes of a JPG or PNG image.
        filename:
            Original filename (optional, informational).
        image_id:
            Unique identifier for this image. A UUID is generated if omitted.

        Returns
        -------
        dict with keys: ``image_id``, ``status``, ``predicted_label``,
        ``confidence``, ``prob_ai``, ``prob_real``, ``preprocess_time_ms``,
        ``inference_time_ms``, ``error_message``.

        Raises
        ------
        GRPCClientError
            If the client is not connected or if the RPC call fails.  The
            exception message is human-readable and suitable for display in
            the GUI.
        """
        if self._stub is None:
            raise GRPCClientError("Client is not connected")

        img_id = image_id or str(uuid.uuid4())
        try:
            request = inference_pb2.ImageRequest(
                image_id=img_id,
                filename=filename or "",
                image_data=image_bytes,
            )
            response = self._stub.ClassifyImage(request, timeout=self.timeout)
            return self._parse_response(response)
        except grpc.RpcError as rpc_err:
            friendly = _grpc_error_message(rpc_err)
            LOG.error(
                "gRPC RpcError classifying image_id=%s filename=%s: %s",
                img_id,
                filename,
                friendly,
            )
            raise GRPCClientError(friendly) from rpc_err
        except GRPCClientError:
            raise
        except Exception as exc:
            LOG.exception(
                "Unexpected error sending image image_id=%s: %s",
                img_id,
                exc,
            )
            raise GRPCClientError(
                f"Error inesperado al enviar imagen: {exc}"
            ) from exc

    def classify_image_safe(
        self,
        image_bytes: bytes,
        filename: Optional[str] = None,
        image_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Send image bytes to the server without raising on failure.

        This is the recommended method for batch-processing loops in the GUI:
        a failure for one image produces a result dict with ``status="error"``
        instead of aborting the entire batch.

        Parameters
        ----------
        image_bytes:
            Raw bytes of a JPG or PNG image.
        filename:
            Original filename (optional, informational).
        image_id:
            Unique identifier. A UUID is generated if omitted.

        Returns
        -------
        dict with keys: ``image_id``, ``status``, ``predicted_label``,
        ``confidence``, ``prob_ai``, ``prob_real``, ``preprocess_time_ms``,
        ``inference_time_ms``, ``error_message``.

        On error, ``status`` is ``"error"`` and ``error_message`` contains a
        human-readable description suitable for display in the GUI.  All
        numeric fields will be ``None``.
        """
        img_id = image_id or str(uuid.uuid4())
        try:
            return self.classify_image(
                image_bytes, filename=filename, image_id=img_id
            )
        except GRPCClientError as err:
            LOG.warning(
                "classify_image_safe: error for image_id=%s filename=%s: %s",
                img_id,
                filename,
                err,
            )
            return {
                "image_id": img_id,
                "status": "error",
                "predicted_label": None,
                "confidence": None,
                "prob_ai": None,
                "prob_real": None,
                "preprocess_time_ms": None,
                "inference_time_ms": None,
                "error_message": str(err),
            }

    def close(self) -> None:
        """Close the underlying gRPC channel and release resources."""
        if self._channel is not None:
            self._channel.close()
            self._channel = None
            self._stub = None
