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
   - 5.5 [Ejecutar el pipeline modular](#55-ejecutar-el-pipeline-modular)
   - 5.6 [Nueva estructura de directorios (Azure)](#56-nueva-estructura-de-directorios-azure)
6. [Despliegue – Azure Container Apps (Frontend)](#6-despliegue--azure-container-apps-frontend)
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

### 5.5 Ejecutar el pipeline modular

```powershell
# Ejecutar el pipeline completo (YOLOv8n fine-tuning, descarga HF automática)
uv run python deployment/azure/pipeline_azure.py
```

> **Sin dataset local requerido:** el pipeline descarga automáticamente el
> dataset [`keremberke/pcb-defect-segmentation`](https://huggingface.co/datasets/keremberke/pcb-defect-segmentation)
> desde Hugging Face en el primer paso.

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

### 5.6 Estructura de directorios (Azure)

```
deployment/azure/
├── pipeline_azure.py          # Orquestador del pipeline (@dsl.pipeline)
├── deploy_azure.py            # Crea la infraestructura de cómputo
├── conda.yml                  # Entorno Conda registrado en Azure ML
├── components/                # Scripts independientes de cada paso
│   ├── ingest_data.py         # Paso 1: descarga dataset desde Hugging Face
│   ├── preprocess_split.py    # Paso 2: transformación + partición 80/20
│   ├── train_yolo.py          # Paso 3: fine-tuning YOLOv8n con MLflow
│   └── evaluate_model.py      # Paso 4: inferencia, mAP y exportación
└── scripts/                   # Scripts heredados (referencia interna)
```

Cada componente en `components/` acepta argumentos `--input_data` y
`--output_data` que Azure ML inyecta dinámicamente al conectar los pasos.

---

## 6. Despliegue – Azure Container Apps (Frontend)

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

Crea un archivo `.env` en la raíz del proyecto (ya incluido en `.gitignore`):

```env
# Backend FastAPI
API_HOST=localhost
API_PORT=8000
API_TIMEOUT=30
LOG_LEVEL=INFO

# Modelo YOLOv8 (Hugging Face)
HF_MODEL_ID=keremberke/yolov8n-pcb-defect-segmentation
```

Para el uso en Docker o Azure Container Apps, inyecta estas variables
directamente como secrets del servicio, sin commitear el archivo `.env`.
