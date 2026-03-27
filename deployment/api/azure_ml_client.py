"""Cliente para Azure ML Managed Batch Endpoint (SDK-based).

Envía trabajos de inferencia al Batch Endpoint y consulta su estado
usando la Azure ML Python SDK, imitando la lógica probada en
end_to_end_test.py.

Autenticación (en orden de prioridad):
    1. ManagedIdentityCredential – Managed Identity en ACI
    2. AzureCliCredential         – credenciales ~/.azure (desarrollo local)

Variables de entorno:
    AZURE_SUBSCRIPTION_ID   Subscription de Azure
    AZURE_RESOURCE_GROUP    Grupo de recursos del workspace
    AZURE_WORKSPACE_NAME    Nombre del workspace de Azure ML
    AZURE_ML_ENDPOINT_NAME  Nombre del Batch Endpoint (default: pcb-batch-inference)
    AZURE_ML_DEPLOYMENT     Nombre del deployment (default: pcb-yolov8n-deployment)
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any, Dict, Optional

LOG = logging.getLogger(__name__)

_ENDPOINT_NAME = os.environ.get("AZURE_ML_ENDPOINT_NAME", "pcb-batch-inference")
_DEPLOYMENT_NAME = os.environ.get("AZURE_ML_DEPLOYMENT", "pcb-yolov8n-deployment")
_BLOB_OUTPUT_PREFIX = "pcb-inference-output"


def _normalize_status(raw: str) -> str:
    """Normaliza el estado devuelto por Azure ML al contrato del frontend."""
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


class AzureMLBatchClient:
    """Cliente SDK para el Batch Endpoint de Azure ML.

    Parameters
    ----------
    subscription_id:
        ID de suscripción. Lee AZURE_SUBSCRIPTION_ID si no se proporciona.
    resource_group:
        Grupo de recursos. Lee AZURE_RESOURCE_GROUP si no se proporciona.
    workspace_name:
        Nombre del workspace. Lee AZURE_WORKSPACE_NAME si no se proporciona.
    endpoint_name:
        Nombre del Batch Endpoint. Lee AZURE_ML_ENDPOINT_NAME o usa el default.
    deployment_name:
        Nombre del deployment. Lee AZURE_ML_DEPLOYMENT o usa el default.
    """

    def __init__(
        self,
        subscription_id: Optional[str] = None,
        resource_group: Optional[str] = None,
        workspace_name: Optional[str] = None,
        endpoint_name: Optional[str] = None,
        deployment_name: Optional[str] = None,
    ) -> None:
        self.subscription_id = subscription_id or os.environ.get(
            "AZURE_SUBSCRIPTION_ID", ""
        )
        self.resource_group = resource_group or os.environ.get(
            "AZURE_RESOURCE_GROUP", ""
        )
        self.workspace_name = workspace_name or os.environ.get(
            "AZURE_WORKSPACE_NAME", ""
        )
        self.endpoint_name = endpoint_name or _ENDPOINT_NAME
        self.deployment_name = deployment_name or _DEPLOYMENT_NAME
        # endpoint_url is used by backend.py for health check display
        self.endpoint_url = (
            f"https://{self.endpoint_name}.centralus.inference.ml.azure.com/jobs"
        )
        self._ml_client = None

    # ── MLClient (lazy init) ────────────────────────────────────────────────

    def _get_ml_client(self):
        """Crea (o reutiliza) el MLClient con credenciales en cadena.

        Orden de credenciales:
            1. ManagedIdentityCredential (ACI con Managed Identity)
            2. AzureCliCredential (desarrollo local con ~/.azure)

        Raises:
            RuntimeError: Si faltan las variables de workspace.
        """
        if self._ml_client is not None:
            return self._ml_client

        if not all([self.subscription_id, self.resource_group, self.workspace_name]):
            raise RuntimeError(
                "Faltan variables de entorno del workspace de Azure ML: "
                "AZURE_SUBSCRIPTION_ID, AZURE_RESOURCE_GROUP, AZURE_WORKSPACE_NAME"
            )

        from azure.identity import ChainedTokenCredential, ManagedIdentityCredential, AzureCliCredential
        from azure.ai.ml import MLClient

        credential = ChainedTokenCredential(
            ManagedIdentityCredential(),
            AzureCliCredential(),
        )

        self._ml_client = MLClient(
            credential=credential,
            subscription_id=self.subscription_id,
            resource_group_name=self.resource_group,
            workspace_name=self.workspace_name,
        )
        LOG.info(
            "MLClient inicializado para workspace %s/%s",
            self.resource_group,
            self.workspace_name,
        )
        return self._ml_client

    # ── Submit job ──────────────────────────────────────────────────────────

    def submit_job(self, input_data_url: str, run_id: str) -> str:
        """Envía un trabajo al Batch Endpoint con una ruta de salida dinámica.

        El output se dirige a:
            azureml://datastores/workspaceblobstore/paths/
            pcb-inference-output/{run_id}/predictions.jsonl

        Args:
            input_data_url: URL del folder de entrada en Blob Storage.
            run_id: Identificador único de esta corrida (usado para la ruta de salida).

        Returns:
            Nombre del job de Azure ML (para hacer polling con get_job_status).

        Raises:
            RuntimeError: Si el envío falla.
        """
        from azure.ai.ml import Input, Output
        from azure.ai.ml.constants import AssetTypes

        blob_path = f"{_BLOB_OUTPUT_PREFIX}/{run_id}/predictions.jsonl"
        output_path = f"azureml://datastores/workspaceblobstore/paths/{blob_path}"

        ml = self._get_ml_client()

        job = ml.batch_endpoints.invoke(
            endpoint_name=self.endpoint_name,
            deployment_name=self.deployment_name,
            input=Input(type=AssetTypes.URI_FOLDER, path=input_data_url),
            outputs={"output": Output(type=AssetTypes.URI_FILE, path=output_path)},
        )

        LOG.info("Job enviado: %s → carpeta de salida: %s", job.name, run_id)
        return job.name

    # ── Job status ──────────────────────────────────────────────────────────

    def get_job_status(self, az_job_name: str) -> Dict[str, Any]:
        """Consulta el estado de un job usando la SDK.

        Args:
            az_job_name: Nombre del job devuelto por submit_job.

        Returns:
            Dict con status normalizado, created_at, updated_at y message.
        """
        ml = self._get_ml_client()
        try:
            job = ml.jobs.get(name=az_job_name)
        except Exception as exc:
            LOG.error("Error consultando job %s: %s", az_job_name, exc)
            raise

        raw_status: str = getattr(job, "status", "unknown") or "unknown"
        creation_ctx = getattr(job, "creation_context", None)
        created_at = str(getattr(creation_ctx, "created_at", "") or "") if creation_ctx else ""
        updated_at = str(getattr(creation_ctx, "last_modified_at", "") or "") if creation_ctx else ""
        return {
            "job_id": az_job_name,
            "status": _normalize_status(raw_status),
            "created_at": created_at,
            "updated_at": updated_at,
            "message": getattr(job, "status_message", None),
        }

    # ── Default datastore container ─────────────────────────────────────────

    def get_default_container(self) -> str:
        """Obtiene el nombre del contenedor del datastore por defecto.

        Azure ML almacena las salidas en el container interno del workspace
        (workspaceblobstore).  Este método devuelve ese nombre para que
        blob_manager pueda acceder directamente a los resultados.

        Returns:
            Nombre del container de Azure Blob Storage del datastore por defecto.
        """
        ml = self._get_ml_client()
        default_ds = ml.datastores.get_default()
        container: str = default_ds.container_name
        LOG.info("Container por defecto del workspace: %s", container)
        return container
