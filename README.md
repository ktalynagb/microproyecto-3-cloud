# Microproyecto 3 – PCB Defect Inspection · Flux Solutions Cali

Sistema de inspección de calidad basada en visión artificial para PCB
(Printed Circuit Boards) usando **YOLOv8n** – Flux Solutions Cali.

> **Entorno de referencia:** Windows 11, PowerShell 7+, gestor de paquetes **uv**,
> Python 3.11, cuenta **Azure for Students** (tenant `uao.edu.co`).

---

## Índice

1. [Prerrequisitos](#1-prerrequisitos)
2. [Instalación local (uv)](#2-instalación-local-uv)
3. [Ejecución local](#3-ejecución-local)
4. [Docker (local)](#4-docker-local)
5. [Azure – Configuración paso a paso](#5-azure--configuración-paso-a-paso)
   - 5.1 [Crear el Workspace en Azure Portal](#51-crear-el-workspace-en-azure-portal)
   - 5.2 [Actualizar config.json](#52-actualizar-configjson)
   - 5.3 [Instalar la extensión Azure ML CLI en Windows](#53-instalar-la-extensión-azure-ml-cli-en-windows)
   - 5.4 [Crear la infraestructura de cómputo](#54-crear-la-infraestructura-de-cómputo)
   - 5.5 [Ejecutar el pipeline modular de entrenamiento](#55-ejecutar-el-pipeline-modular-de-entrenamiento)
   - 5.6 [Desplegar el pipeline de inferencia en lote](#56-desplegar-el-pipeline-de-inferencia-en-lote)
   - 5.7 [Estructura de directorios (Azure)](#57-estructura-de-directorios-azure)
6. [Despliegue – Azure Container Apps (Frontend Streamlit)](#6-despliegue--azure-container-apps-frontend-streamlit)
7. [Despliegue – AWS ECS Fargate (Backend API)](#7-despliegue--aws-ecs-fargate-backend-api)
8. [Variables de entorno](#8-variables-de-entorno)

---

## 1. Prerrequisitos

| Herramienta | Versión mínima | Instalación |
|---|---|---|
| Python | 3.11 | <https://www.python.org/downloads/> |
| uv | última | `winget install astral-sh.uv` |
| Docker Desktop | 4.x | <https://www.docker.com/products/docker-desktop/> |
| Azure CLI | 2.x | `winget install Microsoft.AzureCLI` |
| Git | 2.x | `winget install Git.Git` |

> **Nota:** Si `winget` no está disponible, descarga los instaladores
> directamente desde las páginas enlazadas.

---

## 2. Instalación local (uv)

Todos los comandos se ejecutan en **PowerShell**.

```powershell
# Clonar el repositorio (si aún no lo has hecho)
git clone https://github.com/ktalynagb/microproyecto-3-cloud.git
Set-Location microproyecto-3-cloud

# Instalar dependencias con uv (reemplaza pip + venv)
uv sync
```

uv crea automáticamente el entorno virtual en `.venv/` y lo gestiona por ti.
No necesitas activarlo manualmente para ejecutar comandos con `uv run`.

---

## 3. Ejecución local

### 3.1 Backend FastAPI (servidor de inferencia)

```powershell
# Iniciar el servidor en http://localhost:8000
uv run service/inference_server.py

# O usando make:
make api-server
```

### 3.2 Frontend Streamlit

En otra terminal:

```powershell
# Iniciar la interfaz en http://localhost:8501
uv run -m streamlit run app/streamlit_app.py

# O usando make:
make gui
```

### 3.3 Tests

```powershell
# Todos los tests
uv run -m pytest tests/ -v

# Con reporte de cobertura HTML
uv run -m pytest tests/ --cov --cov-report=html
```

---

## 4. Docker (local)

```powershell
# Construir y levantar todos los servicios
docker-compose up --build

# Solo el backend FastAPI (puerto 8000)
docker build -t pcb-api -f Dockerfile-api .
docker run --rm -p 8000:8000 --env-file .env pcb-api

# Solo el frontend Streamlit (puerto 8501)
docker build -t pcb-frontend -f Dockerfile .
docker run --rm -p 8501:8501 pcb-frontend
```

---

## 5. Azure – Configuración paso a paso

### 5.1 Crear el Workspace en Azure Portal

El Resource Group **pcb-ml-rg** ya existe en la suscripción
`2a088410-37ec-472a-ae7e-09126fba02a6` (UAO Azure for Students).

1. Ve a <https://portal.azure.com> e inicia sesión con tu cuenta `@uao.edu.co`.
2. Busca **"Azure Machine Learning"** en la barra de búsqueda.
3. Clic en **"+ Crear"**.
4. Completa el formulario:
   - **Suscripción:** `Azure for Students`
   - **Grupo de recursos:** `pcb-ml-rg` *(seleccionar el existente)*
   - **Nombre del área de trabajo:** `pcb-ml-workspace`
   - **Región:** `East US` (o la más cercana disponible)
5. Clic en **"Revisar y crear"** → **"Crear"**.
6. Espera ~3 minutos hasta que el deployment finalice.

### 5.2 Actualizar config.json

El archivo `config.json` en la raíz del proyecto ya tiene los valores correctos:

```json
{
  "subscription_id": "2a088410-37ec-472a-ae7e-09126fba02a6",
  "resource_group": "pcb-ml-rg",
  "workspace_name": "pcb-ml-workspace"
}
```

Si necesitas descargarlo directamente desde el portal:

1. Abre el Workspace recién creado en Azure Portal.
2. En la esquina superior derecha, haz clic en el ícono **"⬇ Descargar config.json"**.
3. Reemplaza el archivo `config.json` en la raíz del proyecto con el descargado.

### 5.3 Instalar la extensión Azure ML CLI en Windows

> **Problema conocido:** en Windows, `az extension add -n ml` puede fallar
> con errores de `setuptools` si la versión del sistema es incompatible.

Solución recomendada:

```powershell
# 1. Iniciar sesión en Azure
az login

# 3. Instalar la extensión ml en un entorno uv aislado (evita conflictos)
uv pip install azure-ai-ml azure-identity

# 4. Verificar la instalación
uv run python -c "from azure.ai.ml import MLClient; print('azure-ai-ml OK')"

# Alternativa: instalar la extensión CLI directamente
az extension add -n ml --upgrade
az ml --version
```

Si el paso 4 (`az extension add`) falla con errores de `setuptools`:

```powershell
# Forzar la reinstalación aislada de setuptools
uv pip install --upgrade setuptools wheel
az extension add -n ml --upgrade --debug
```

### 5.4 Crear la infraestructura de cómputo

El script `deploy_azure.py` crea el clúster de cómputo DS3 v2:

```powershell
# Autenticarse primero
az login

# Crear el clúster (se crea solo si no existe)
uv run python deployment/azure/deploy_azure.py
```

Salida esperada:
```
Creando nuevo clúster DS3 v2 (4 Cores, 14GB RAM)...
Clúster creado con éxito.
Infraestructura de cómputo lista para la sustentación.
```

### 5.5 Ejecutar el pipeline modular de entrenamiento

```powershell
# REQUERIDO: Exportar ROBOFLOW_API_KEY antes de ejecutar (ver sección 8.2)
$env:ROBOFLOW_API_KEY = "<tu_api_key>"

# Ejecutar el pipeline completo (YOLOv8n fine-tuning, descarga Roboflow automática)
uv run python deployment/azure/pipeline_azure.py
```

> **Dataset Roboflow (recomendado):** el pipeline descarga automáticamente el
> dataset [`diplom-qz7q6/defects-2q87r v8`](https://universe.roboflow.com/diplom-qz7q6/defects-2q87r/dataset/8)
> desde Roboflow con imágenes + labels YOLO correctos para las 4 clases:
> `Dry_joint`, `Incorrect_installation`, `PCB_damage`, `Short_circuit`.
>
> **Sin `ROBOFLOW_API_KEY`:** el pipeline usa fallback Hugging Face
> (`keremberke/pcb-defect-segmentation`) que puede obtener 0 imágenes
> útiles para entrenamiento (ver sección 9 Troubleshooting).

El pipeline ejecuta los siguientes 4 códigos en Azure ML que corresponde a los siguientes pasos:

| # | Nombre | Descripción |
|---|--------|-------------|
| 1 | **Import Data** | Ingesta de imágenes PCB desde Azure Blob Storage |
| 2 | **Convert to Image Directory** | Conversión al formato ImageDirectory |
| 3 | **Init Image Transformation** | Resize 640×640 + normalización ImageNet |
| 4 | **Apply Transformation** | Aplica las transformaciones al dataset |
| 5 | **Split Image Directory** | Partición 80% train / 20% test (seed=42) |
| 6 | **Execute Python Script** | Carga YOLOv8n desde HuggingFace y configura fine-tuning |
| 7 | **Train PyTorch Model** | Fine-tuning del modelo sobre el dataset PCB |
| 8 | **Score Image Model** | Predicciones (bounding boxes + máscaras) sobre test |
| 9 | **Evaluate Model** | Métricas: mAP, precisión y recall |
| 10 | **Export Data** | Exporta modelo y resultados a Blob Storage |

| # | Componente | Script | Descripción |
|---|-----------|--------|-------------|
| 1 | **Ingest Data** | `components/ingest_data.py` | Descarga el dataset PCB desde Hugging Face |
| 2 | **Preprocess & Split** | `components/preprocess_split.py` | Resize 640×640 + partición 80% train / 20% test (seed=42) |
| 3 | **Train YOLOv8n** | `components/train_yolo.py` | Fine-tuning del modelo con registro MLflow |
| 4 | **Evaluate Model** | `components/evaluate_model.py` | Inferencia, métricas (mAP) y exportación a Blob Storage |

Monitorea el progreso en **Azure ML Studio**:
<https://ml.azure.com>

### 5.6 Desplegar el pipeline de inferencia en lote

Una vez completado el pipeline de entrenamiento (sección 5.5), ejecuta el
siguiente script para **registrar el modelo** en Azure ML Model Registry y
**crear el Batch Endpoint** que el frontend Streamlit invoca:

```powershell
# (Opcional) Especificar la ruta del artefacto best.pt si difiere del default
# Por defecto usa la salida del pipeline de entrenamiento en Blob Storage
uv run python deployment/azure/inference_pipeline.py
```

Si el modelo está en una ruta local o en un URI distinto, pásala como argumento:

```powershell
uv run python deployment/azure/inference_pipeline.py `
  --model_path "azureml://datastores/workspaceblobstore/paths/pcb-results/"
```

Salida esperada al finalizar:

```
[INFO] Modelo registrado: pcb-yolov8n (versión 1)
[INFO] Batch Endpoint 'pcb-batch-inference' listo.
[INFO] Batch Deployment 'pcb-yolov8n-deployment' desplegado en endpoint 'pcb-batch-inference'.
[INFO] Pipeline de inferencia desplegado con éxito.
[INFO] Scoring URI: https://pcb-batch-inference.<region>.inference.ml.azure.com/...
```

#### Verificar el Batch Endpoint

```powershell
# Listar endpoints disponibles
az ml batch-endpoint list --workspace-name pcb-ml-workspace --resource-group pcb-ml-rg

# Ver el estado del deployment
az ml batch-deployment show `
  --name pcb-yolov8n-deployment `
  --endpoint-name pcb-batch-inference `
  --workspace-name pcb-ml-workspace `
  --resource-group pcb-ml-rg
```

#### Invocar el endpoint manualmente (prueba rápida)

Puedes probar el endpoint con un lote de imágenes directamente desde PowerShell
usando el SDK de Python:

```powershell
# Ejecutar inferencia sobre un directorio local de imágenes de prueba
uv run python - <<'EOF'
import json
from pathlib import Path
from azure.ai.ml import MLClient, Input
from azure.ai.ml.constants import AssetTypes
from azure.identity import DefaultAzureCredential

ml_client = MLClient.from_config(DefaultAzureCredential())

job = ml_client.batch_endpoints.invoke(
    endpoint_name="pcb-batch-inference",
    input=Input(
        type=AssetTypes.URI_FOLDER,
        path="<ruta_local_o_uri_azureml_con_imagenes>",
    ),
)
print(f"Job de inferencia enviado: {job.name}")
print("Monitorea en: https://ml.azure.com")
EOF
```

Reemplaza `<ruta_local_o_uri_azureml_con_imagenes>` por la ruta a un directorio
con imágenes JPG/PNG.

#### Obtener los resultados del lote

```powershell
# Esperar a que finalice y descargar el JSONL de predicciones
uv run python - <<'EOF'
from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential

ml_client = MLClient.from_config(DefaultAzureCredential())

job_name = "<nombre_del_job_devuelto_arriba>"
ml_client.jobs.stream(job_name)          # Sigue los logs en tiempo real

# El archivo predictions.jsonl se genera en el output del job
outputs = ml_client.jobs.get(job_name).outputs
print("Salida:", outputs)
EOF
```

El archivo `predictions.jsonl` contiene una línea JSON por imagen con la
siguiente estructura:

```json
{
  "filename": "pcb_001.jpg",
  "has_defects": true,
  "no_defect_notification": null,
  "detections_count": 2,
  "inference_time_ms": 87.4,
  "error": null,
  "detections": [
    {
      "class_name": "dry_joint",
      "class_id": 0,
      "confidence": 0.83,
      "bbox": [0.12, 0.34, 0.45, 0.67],
      "mask_points": [[0.12, 0.34], [0.45, 0.34], [0.45, 0.67], [0.12, 0.67]]
    }
  ]
}
```

#### Arquitectura del pipeline de inferencia

El script de scoring (`deployment/azure/batch_inference/score.py`) ejecuta
las siguientes 5 etapas de forma automática por cada lote:

| Etapa | Módulo | Descripción |
|-------|--------|-------------|
| 1 | `batch_receiver.py` | Recibe y redimensiona imágenes a 640×640 |
| 2 | `inference_engine.py` | Ejecuta YOLOv8n (clase, bbox, máscara, confidence) |
| 3 | `post_processor.py` | Superpone máscaras; notifica "PCB sin defectos" si no hay detecciones |
| 4 | `blob_exporter.py` | Exporta imágenes anotadas + PDF a Blob Storage (TTL 24 h) |
| 5 | `delivery.py` | Genera URLs SAS temporales para descarga |

#### Integración con el frontend Streamlit

El frontend Streamlit (`app/streamlit_app.py`) invoca el endpoint de inferencia
a través de `app/api_client.py`. Configura las variables de entorno para
apuntar al servidor de inferencia (ver sección 8):

```env
API_HOST=<hostname_del_inference_server>
API_PORT=8000
API_TIMEOUT=60
```

El cliente llama a `/predict` con cada imagen y recibe la respuesta con:
`status`, `has_defects`, `defects_summary`, `processed_image_base64`.

---

### 5.7 Estructura de directorios (Azure)

```
deployment/azure/
├── pipeline_azure.py          # Orquestador del pipeline de entrenamiento (@dsl.pipeline)
├── inference_pipeline.py      # Orquestador del pipeline de inferencia (Batch Endpoint)
├── deploy_azure.py            # Crea la infraestructura de cómputo
├── Dockerfile                 # Imagen Docker para el entorno Azure ML
├── conda.yml                  # Entorno Conda registrado en Azure ML
├── components/                # Componentes del pipeline de entrenamiento
│   ├── ingest_data.py         # Paso 1: descarga dataset desde Hugging Face
│   ├── preprocess_split.py    # Paso 2: transformación + partición 80/20
│   ├── train_yolo.py          # Paso 3: fine-tuning YOLOv8n con MLflow
│   └── evaluate_model.py      # Paso 4: inferencia, mAP y exportación
└── batch_inference/           # Módulos del pipeline de inferencia en lote
    ├── __init__.py
    ├── score.py               # Script de scoring para Azure ML Batch Endpoint
    ├── batch_receiver.py      # Etapa 1: recepción y resize de imágenes
    ├── inference_engine.py    # Etapa 2: inferencia YOLOv8n
    ├── post_processor.py      # Etapa 3: anotaciones y visualización
    ├── blob_exporter.py       # Etapa 4: exportación efímera a Blob Storage
    ├── delivery.py            # Etapa 5: URLs SAS de descarga
    ├── config.py              # Configuración centralizada
    ├── utils.py               # Utilidades de imagen (resize, draw, etc.)
    └── logger.py              # Logging estructurado en JSON

# El frontend PERMANECE en app/ (Streamlit, no FastAPI):
app/
├── streamlit_app.py           # Frontend Streamlit (desplegado en Azure Container Apps)
├── api_client.py              # Cliente REST para el servidor de inferencia
├── batch_runner.py            # Orquestador de inferencia en lote desde el frontend
└── ...
```

Cada componente en `components/` acepta argumentos `--input_data` y
`--output_data` que Azure ML inyecta dinámicamente al conectar los pasos.

---

## 6. Despliegue – Azure Container Apps (Frontend Streamlit)

Ejecuta los siguientes comandos en **PowerShell** para definir los parámetros
del proyecto.

```powershell
$LOCATION     = "canadacentral"
$PROJECT      = "pcb-defect-inspection"
$RG_APP       = "$PROJECT-app"
$APP_NAME     = "$PROJECT-app"
$ENV_APP      = "$APP_NAME-env"
$ACR_NAME     = "uaopcbdefect"
```

Crear el grupo de recursos, el registro de contenedores y extraer las
credenciales de acceso:

```powershell
az group create --name $RG_APP --location $LOCATION
az acr create --resource-group $RG_APP --name $ACR_NAME --sku Basic --admin-enabled true

$ACR_SERVER = az acr show --name $ACR_NAME --resource-group $RG_APP --query loginServer -o tsv
$ACR_USER   = az acr credential show --name $ACR_NAME --resource-group $RG_APP --query username -o tsv
$ACR_PASS   = az acr credential show --name $ACR_NAME --resource-group $RG_APP --query "passwords[0].value" -o tsv
```

Construir la imagen Docker y desplegar el frontend (puerto 8501):

```powershell
az acr login -n $ACR_USER
docker build -t "${APP_NAME}:latest" -f Dockerfile .
docker tag "${APP_NAME}:latest" "${ACR_SERVER}/${APP_NAME}:latest"
docker push "${ACR_SERVER}/${APP_NAME}:latest"

az acr repository list --name $ACR_NAME --output table

az containerapp env create `
  --name $ENV_APP `
  --resource-group $RG_APP `
  --location $LOCATION

az containerapp create `
  --name $APP_NAME `
  --resource-group $RG_APP `
  --environment $ENV_APP `
  --image "${ACR_SERVER}/${APP_NAME}:latest" `
  --target-port 8501 `
  --ingress external `
  --registry-server $ACR_SERVER `
  --registry-username $ACR_USER `
  --registry-password $ACR_PASS
```

---

## 7. Despliegue – AWS ECS Fargate (Backend API)

Ejecuta los siguientes comandos en **PowerShell**:

```powershell
$REGION     = "us-east-2"
$PROJECT    = "pcb-defect-inspection"
$APP_NAME   = "$PROJECT-app"
$ACCOUNT_ID = aws sts get-caller-identity --query Account --output text
$ECR_URI    = "${ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com/${APP_NAME}"
```

Crear el repositorio ECR, autenticar Docker y subir la imagen:

```powershell
aws ecr create-repository --repository-name $APP_NAME --region $REGION

aws ecr get-login-password --region $REGION |
  docker login --username AWS --password-stdin "${ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com"

docker build -t $APP_NAME -f Dockerfile .
docker tag "${APP_NAME}:latest" "${ECR_URI}:latest"
docker push "${ECR_URI}:latest"
```

Crear el rol IAM y configurar el Grupo de Seguridad:

```powershell
aws iam create-role `
  --role-name ecsTaskExecutionRole `
  --assume-role-policy-document file://.aws/ecs-tasks-trust-policy.json

aws iam attach-role-policy `
  --role-name ecsTaskExecutionRole `
  --policy-arn arn:aws:iam::aws:policy/service-role/AmazonECSTaskExecutionRolePolicy

$VPC_ID    = aws ec2 describe-vpcs --filters Name=isDefault,Values=true --query "Vpcs[0].VpcId" --output text
$SUBNET_ID = aws ec2 describe-subnets --filters Name=vpc-id,Values=$VPC_ID --query "Subnets[0].SubnetId" --output text
$SG_ID     = aws ec2 create-security-group `
               --group-name "${APP_NAME}-sg" `
               --description "Frontend SG" `
               --vpc-id $VPC_ID `
               --query "GroupId" --output text

aws ec2 authorize-security-group-ingress `
  --group-id $SG_ID --protocol tcp --port 8501 --cidr 0.0.0.0/0
```

Crear el balanceador de carga y el target group:

```powershell
$SUBNET_ID_2 = aws ec2 describe-subnets `
  --filters Name=vpc-id,Values=$VPC_ID `
  --query "Subnets[1].SubnetId" --output text

$ALB_ARN = aws elbv2 create-load-balancer `
  --name "${APP_NAME}-alb" `
  --subnets $SUBNET_ID $SUBNET_ID_2 `
  --security-groups $SG_ID `
  --query "LoadBalancers[0].LoadBalancerArn" --output text

$TG_ARN = aws elbv2 create-target-group `
  --name "${APP_NAME}-tg" `
  --protocol HTTP --port 8501 `
  --vpc-id $VPC_ID `
  --target-type ip `
  --query "TargetGroups[0].TargetGroupArn" --output text

aws elbv2 create-listener `
  --load-balancer-arn $ALB_ARN `
  --protocol HTTP --port 80 `
  --default-actions "Type=forward,TargetGroupArn=$TG_ARN"
```

Crear la definición de la tarea y el servicio ECS:

```powershell
# Generar la definición de tarea
$taskDef = @{
  family               = $APP_NAME
  networkMode          = "awsvpc"
  containerDefinitions = @(@{
    name         = $APP_NAME
    image        = "${ECR_URI}:latest"
    portMappings = @(@{ containerPort = 8501; hostPort = 8501; protocol = "tcp" })
  })
  requiresCompatibilities = @("FARGATE")
  cpu                  = "1024"
  memory               = "2048"
  executionRoleArn     = "arn:aws:iam::${ACCOUNT_ID}:role/ecsTaskExecutionRole"
} | ConvertTo-Json -Depth 10

$taskDef | Out-File -Encoding utf8 .aws/ecs-task-definition.json

aws ecs register-task-definition --cli-input-json file://.aws/ecs-task-definition.json
aws ecs create-cluster --cluster-name $APP_NAME
aws ecs create-service `
  --cluster $APP_NAME `
  --service-name $APP_NAME `
  --task-definition $APP_NAME `
  --desired-count 1 `
  --launch-type FARGATE `
  --network-configuration "awsvpcConfiguration={subnets=[$SUBNET_ID],securityGroups=[$SG_ID],assignPublicIp=ENABLED}" `
  --load-balancers "targetGroupArn=$TG_ARN,containerName=$APP_NAME,containerPort=8501"
```

Obtener la URL del load balancer:

```powershell
$ALB_URL = aws elbv2 describe-load-balancers `
  --load-balancer-arns $ALB_ARN `
  --query "LoadBalancers[0].DNSName" --output text
Write-Host "Frontend URL: http://$ALB_URL"
```

---

## 8. Variables de entorno

Crea un archivo `.env` en la raíz del proyecto basándote en `.env.example`
(el `.env` real está en `.gitignore` y **nunca** debe commitearse):

```powershell
Copy-Item .env.example .env
# Editar .env con los valores reales
```

### 8.1 Configuración local (`.env`)

```env
# Roboflow API (REQUERIDO para pipeline de entrenamiento)
ROBOFLOW_API_KEY=<tu_api_key_de_roboflow>

# Backend FastAPI (servidor de inferencia local o en contenedor)
API_HOST=localhost
API_PORT=8000
API_TIMEOUT=60
LOG_LEVEL=INFO

# Modelo YOLOv8 (Hugging Face - fallback)
HF_MODEL_ID=keremberke/yolov8n-pcb-defect-segmentation

# Pipeline de batch inference – modelo
PCB_MODEL_PATH=best.pt
PCB_CONF_THRESHOLD=0.25
PCB_IOU_THRESHOLD=0.45
PCB_INFERENCE_TIMEOUT=30

# Azure Blob Storage (exportación efímera de resultados)
AZURE_STORAGE_ACCOUNT=<nombre_cuenta_storage>
AZURE_STORAGE_KEY=<clave_de_acceso>
AZURE_CONTAINER_NAME=pcb-results
```

### 8.2 Obtener `ROBOFLOW_API_KEY` paso a paso

1. Ve a <https://app.roboflow.com/settings/account> e inicia sesión (o crea una cuenta gratuita).
2. En la sección **"Roboflow API"**, copia tu **Private API Key**.
3. Agrega la clave a tu `.env` local o expórtala como variable de entorno:
   ```powershell
   # PowerShell (sesión actual)
   $env:ROBOFLOW_API_KEY = "<tu_api_key>"

   # O en .env
   ROBOFLOW_API_KEY=<tu_api_key>
   ```
4. El pipeline de entrenamiento descarga automáticamente el dataset
   [`diplom-qz7q6/defects-2q87r v8`](https://universe.roboflow.com/diplom-qz7q6/defects-2q87r/dataset/8)
   con la estructura YOLO correcta:
   ```
   train/images/ + train/labels/
   valid/images/ + valid/labels/
   test/images/  + test/labels/
   ```

### 8.3 Configuración en Azure ML

Hay dos opciones para pasar `ROBOFLOW_API_KEY` al pipeline de Azure ML:

**Opción A – Variable de entorno al ejecutar el pipeline:**

```powershell
# Exportar antes de lanzar el pipeline
$env:ROBOFLOW_API_KEY = "<tu_api_key>"
uv run python deployment/azure/pipeline_azure.py
```

El script `pipeline_azure.py` lee automáticamente la variable del entorno
de la máquina local y la inyecta en el step de ingesta.

**Opción B – Azure Key Vault (recomendado para CI/CD):**

```powershell
# Crear secreto en Key Vault
az keyvault secret set `
  --vault-name <nombre-keyvault> `
  --name "ROBOFLOW-API-KEY" `
  --value "<tu_api_key>"

# Referenciar desde pipeline_azure.py (ver documentación Azure ML SDK v2)
```

### 8.4 Validación automática en pipeline

El script `ingest_data.py` valida la presencia de `ROBOFLOW_API_KEY` al inicio:

- ✅ Si existe → descarga desde Roboflow (dataset completo con labels YOLO)
- ⚠️ Si no existe → fallback a Hugging Face (puede obtener 0 imágenes, ver sección 9)

Ejecuta el validador de dataset antes del pipeline:

```powershell
# Verificar estructura del dataset descargado
uv run python deployment/azure/validate_dataset.py --dataset_dir <ruta_dataset>

# Verificar sincronización Dockerfile ↔ imports Python
uv run python deployment/azure/validate_dockerfile.py
```

### Variables específicas del Batch Endpoint (Azure ML)

Estas variables se configuran directamente en el Batch Deployment y no
requieren estar en el `.env` local:

| Variable | Valor por defecto | Descripción |
|----------|-------------------|-------------|
| `ROBOFLOW_API_KEY` | *(requerido)* | API Key de Roboflow para descarga del dataset |
| `PCB_CONF_THRESHOLD` | `0.25` | Umbral de confianza para detecciones |
| `PCB_IOU_THRESHOLD` | `0.45` | Umbral IoU para NMS |
| `PCB_INFERENCE_TIMEOUT` | `30` | Timeout máximo por imagen (segundos) |
| `AZUREML_MODEL_DIR` | *(inyectado por Azure ML)* | Directorio donde Azure ML monta el modelo `best.pt` |

`inference_pipeline.py` las configura automáticamente al crear el deployment.

---

## 9. Troubleshooting

### 9.1 Error: `Dataset completo: 0 imágenes | splits={}`

**Causa:** `ROBOFLOW_API_KEY` no definida → el pipeline usa fallback Hugging Face
que descarga zips incompletos.

**Solución:**
```powershell
# 1. Obtener API Key en https://app.roboflow.com/settings/account
# 2. Exportar la variable
$env:ROBOFLOW_API_KEY = "<tu_api_key>"
# 3. Volver a ejecutar el pipeline
uv run python deployment/azure/pipeline_azure.py
```

---

### 9.2 Error: `WARNING no labels found in segment set, cannot compute metrics`

**Causa:** El modelo se entrenó con imágenes pero sin labels YOLO (mAP = 0).

**Causa raíz:** El dataset de entrenamiento no tiene archivos `.txt` de labels.

**Solución:**
1. Verificar que `ROBOFLOW_API_KEY` esté definida (ver 9.1)
2. Validar el dataset descargado:
   ```powershell
   uv run python deployment/azure/validate_dataset.py --dataset_dir <ruta>
   ```
3. Asegurarse de que el formato sea `yolov8` (incluye labels en formato YOLO detección/segmentación)

---

### 9.3 Error: `images not found, missing path '/mnt/.../val_split/images'`

**Causa:** El `dataset.yaml` guardado durante el entrenamiento tiene rutas absolutas
del contenedor de entrenamiento que no existen en el contenedor de evaluación.

**Solución (ya implementada en v2.0):** `evaluate_model.py` detecta automáticamente
si los paths del `dataset.yaml` son inválidos y genera uno nuevo apuntando al test set
actual. Si ves este error, actualiza el código a la versión más reciente.

---

### 9.4 Error: `ModuleNotFoundError: No module named 'azureml.core'`

**Causa:** El Dockerfile no incluye `azureml-core`.

**Solución (ya implementada en v2.0):** El Dockerfile actualizado incluye:
```dockerfile
RUN pip install --no-cache-dir \
    azureml-core>=1.53.0 \
    azureml-sdk>=1.53.0 \
    ...
```
Verifica que el entorno Azure ML registrado (`pcb-yolo-env`) use la versión
correcta del Dockerfile. Si el entorno ya está registrado con la versión anterior,
incrementa `ENVIRONMENT_VERSION` en `pipeline_azure.py` y re-ejecuta.

---

### 9.5 Verificar sincronización Dockerfile ↔ dependencias

```powershell
# Ejecutar antes de cada deploy
uv run python deployment/azure/validate_dockerfile.py
```

Salida esperada:
```
✅ Todos los imports Python están cubiertos en el Dockerfile.
```

Si aparecen ❌, agrega los paquetes faltantes al Dockerfile y a `conda.yml`.

---

### 9.6 Logs útiles para diagnóstico

| Mensaje | Archivo | Significado |
|---------|---------|-------------|
| `Dataset completo: N imágenes` | `ingest_data.py` | N debe ser > 0 |
| `Split 'train': N imágenes, M labels válidos` | `ingest_data.py` | N ≈ M esperado |
| `Resize=640x640 \| Train=N \| Test=M` | `preprocess_split.py` | N+M = total imágenes |
| `dataset.yaml generado en: ...` | `train_yolo.py` | Confirma rutas correctas |
| `mAP@0.5=X` | `evaluate_model.py` | X > 0 si hay detecciones |
