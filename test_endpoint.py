import json
from pathlib import Path
from azure.ai.ml import MLClient, Input
from azure.ai.ml.constants import AssetTypes
from azure.identity import AzureCliCredential

ml_client = MLClient.from_config(AzureCliCredential())

job = ml_client.batch_endpoints.invoke(
    endpoint_name="pcb-batch-inference",
    deployment_name="pcb-yolov8n-deployment",
    input=Input(
        type=AssetTypes.URI_FOLDER,
        path=r"C:\Users\Katalina Garcia\Downloads\test",
    ),
)
print(f"Job de inferencia enviado: {job.name}")
print("Monitorea en: https://ml.azure.com")
