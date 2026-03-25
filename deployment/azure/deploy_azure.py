from azure.ai.ml import MLClient
from azure.ai.ml.entities import AmlCompute
from azure.identity import DefaultAzureCredential
from dotenv import load_dotenv

# 1. Cargar variables del archivo .env (el de la raíz)
load_dotenv()

# 2. Conexión con Azure (usa el config.json)
credential = DefaultAzureCredential()
try:
    ml_client = MLClient.from_config(credential)
except Exception:
    print("Error: No se encontró config.json. Créalo en la raíz del proyecto.")
    exit()

# 3. Configuración del clúster DS3 v2
cpu_compute_name = "cpu-cluster-ds3"

try:
    cpu_cluster = ml_client.compute.get(cpu_compute_name)
    print(f"El clúster {cpu_compute_name} ya existe.")
except Exception:
    print("Creando nuevo clúster DS3 v2 (4 Cores, 14GB RAM)...")
    cpu_cluster = AmlCompute(
        name=cpu_compute_name,
        type="amlcompute",
        size="STANDARD_DS3_V2", 
        min_instances=0,     
        max_instances=2,
        idle_time_before_scale_down=120, # Se apaga tras 2 min de inactividad
    )
    ml_client.compute.begin_create_or_update(cpu_cluster).result()
    print("Clúster creado con éxito.")

print("Infraestructura de cómputo lista para la sustentación.")