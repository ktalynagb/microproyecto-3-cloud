from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential

ml_client = MLClient.from_config(DefaultAzureCredential())

job_name = "batchjob-ac850679-01e8-48d3-aa77-e326bceab368"
ml_client.jobs.stream(job_name)

outputs = ml_client.jobs.get(job_name).outputs
print("Salida:", outputs)
