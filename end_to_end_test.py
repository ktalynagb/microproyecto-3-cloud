import os
import time
import json
import cv2
from pathlib import Path
from dotenv import load_dotenv

from azure.ai.ml import MLClient, Input, Output
from azure.ai.ml.constants import AssetTypes
from azure.identity import AzureCliCredential
from azure.storage.blob import BlobServiceClient

# ==========================================
# ⚙️ 1. CONFIGURACIÓN GENERAL
# ==========================================
load_dotenv()

# Credenciales de Storage
STORAGE_ACCOUNT = os.getenv("AZURE_STORAGE_ACCOUNT")
STORAGE_KEY = os.getenv("AZURE_STORAGE_KEY")

# Parámetros de Azure ML
ENDPOINT_NAME = "pcb-batch-inference"
DEPLOYMENT_NAME = "pcb-yolov8n-deployment"

# Rutas locales
LOCAL_IMAGES_DIR = r"C:\Users\Katalina Garcia\Downloads\test"
LOCAL_OUTPUT_DIR = "./resultados_visuales"
LOCAL_JSON_FILE = "./predictions_actual.jsonl"

# Colores BGR para OpenCV según el defecto
COLORS = {
    "short_circuit": (0, 0, 255),            # Rojo
    "dry_joint": (255, 0, 0),                # Azul
    "incorrect_installation": (0, 255, 255), # Amarillo
    "pcb_damage": (255, 0, 255)              # Magenta
}

def main():
    print("="*50)
    print("🚀 PIPELINE END-TO-END: INSPECCIÓN DE PCBs")
    print("="*50)

    # 1. Autenticación con Azure
    print("\n[1/5] Autenticando servicios de Azure...")
    ml_client = MLClient.from_config(AzureCliCredential())
    
    # ==========================================
    # 📤 2. LANZAMIENTO DEL TRABAJO (DINÁMICO)
    # ==========================================
    print("\n[2/5] Configurando y enviando lote de imágenes...")
    
    # Creamos un ID único para esta corrida basado en la fecha y hora
    run_id = f"run_{time.strftime('%Y%m%d_%H%M%S')}"
    ruta_blob_interna = f"pcb-inference-output/{run_id}/predictions.jsonl"
    
    output_path = f"azureml://datastores/workspaceblobstore/paths/{ruta_blob_interna}"
    output_data = Output(type=AssetTypes.URI_FILE, path=output_path)

    job = ml_client.batch_endpoints.invoke(
        endpoint_name=ENDPOINT_NAME,
        deployment_name=DEPLOYMENT_NAME,
        input=Input(
            type=AssetTypes.URI_FOLDER,
            path=LOCAL_IMAGES_DIR,
        ),
        outputs={"output": output_data}
    )
    
    print(f"✅ Trabajo enviado exitosamente.")
    print(f"ID del Job: {job.name}")
    print(f"Carpeta de destino en nube: {run_id}")

    # ==========================================
    # ⏳ 3. ESPERA ACTIVA (POLLING)
    # ==========================================
    print("\n[3/5] Esperando a que el clúster procese las PCBs (~3 min)...")
    while True:
        job_status = ml_client.jobs.get(name=job.name).status
        if job_status in ["Completed", "Failed", "Canceled"]:
            print(f"\n✅ ¡Trabajo finalizado con estado: {job_status}!")
            if job_status != "Completed":
                print("❌ El trabajo falló. Revisa los logs en el portal.")
                return
            break
        print(f"   Estado: {job_status}... comprobando en 30s.")
        time.sleep(30)

    # ==========================================
    # 📥 4. RESCATE QUIRÚRGICO DEL JSONL
    # ==========================================
    print("\n[4/5] Rescatando el archivo de resultados...")
    
    # Obtenemos el nombre del contenedor interno de Azure ML
    default_datastore = ml_client.datastores.get_default()
    real_container = default_datastore.container_name
    
    # Conectamos a Blob Storage
    conn_str = f"DefaultEndpointsProtocol=https;AccountName={STORAGE_ACCOUNT};AccountKey={STORAGE_KEY};EndpointSuffix=core.windows.net"
    blob_service_client = BlobServiceClient.from_connection_string(conn_str)
    
    blob_client = blob_service_client.get_blob_client(container=real_container, blob=ruta_blob_interna)
    
    with open(LOCAL_JSON_FILE, "wb") as f:
        f.write(blob_client.download_blob().readall())
    
    print(f"✅ Archivo JSONL guardado en: {LOCAL_JSON_FILE}")

if __name__ == "__main__":
    main()