#!/usr/bin/env bash
# deploy_aci.sh — Despliega el backend FastAPI en Azure Container Instances
#
# Uso:
#   ./deployment/azure/scripts/deploy_aci.sh
#
# Variables de entorno requeridas (o en .env):
#   RESOURCE_GROUP, LOCATION, ACR_NAME, PCB_API_KEY, AZURE_ML_TOKEN,
#   STORAGE_ACCOUNT, STORAGE_KEY, AZURE_STORAGE_CONNECTION_STRING
#
# El script:
#   1. Crea/verifica el Resource Group y Azure Container Registry (ACR).
#   2. Construye y empuja la imagen Docker.
#   3. Crea (o actualiza) el Container Group en ACI con las variables de entorno.

set -euo pipefail

# ---------------------------------------------------------------------------
# Valores por defecto (anular con variables de entorno o .env)
# ---------------------------------------------------------------------------
: "${RESOURCE_GROUP:=pcb-ml-rg}"
: "${LOCATION:=centralus}"
: "${ACR_NAME:=pcbdefectacr}"
: "${CONTAINER_GROUP:=pcb-api-cg}"
: "${CONTAINER_NAME:=pcb-api}"
: "${IMAGE_TAG:=latest}"
: "${DNS_LABEL:=pcb-defect-api}"
: "${CPU:=1}"
: "${MEMORY_GB:=2}"
: "${PCB_API_KEY:=changeme-secret-key}"
: "${BATCH_ENDPOINT_URL:=https://pcb-batch-inference.centralus.inference.ml.azure.com/jobs}"
: "${STORAGE_ACCOUNT:=pcbmlworstorage428505ef7}"
: "${PREDICTIONS_CONTAINER:=azureml-blobstore-fa3e2152-1a09-4e81-acb4-3f701118ca5e}"
: "${INPUT_CONTAINER:=pcb-inference-input}"
: "${POLL_INTERVAL_S:=10}"
: "${MAX_POLL_ATTEMPTS:=36}"
: "${CORS_ORIGINS:=https://*.streamlit.app,http://localhost:8501}"

# Variables sensibles (deben estar definidas)
: "${AZURE_ML_TOKEN:?Variable AZURE_ML_TOKEN no definida}"
: "${STORAGE_KEY:=}"
: "${AZURE_STORAGE_CONNECTION_STRING:=}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
IMAGE_NAME="${ACR_NAME}.azurecr.io/${CONTAINER_NAME}:${IMAGE_TAG}"

echo "=========================================="
echo " PCB Defect Detection — ACI Deployment"
echo "=========================================="
echo "Resource Group : ${RESOURCE_GROUP}"
echo "Location       : ${LOCATION}"
echo "ACR            : ${ACR_NAME}"
echo "Image          : ${IMAGE_NAME}"
echo ""

# 1. Crear Resource Group
echo "[1/5] Creando Resource Group (si no existe)…"
az group create \
  --name "${RESOURCE_GROUP}" \
  --location "${LOCATION}" \
  --output none

# 2. Crear ACR
echo "[2/5] Creando Azure Container Registry (si no existe)…"
az acr create \
  --resource-group "${RESOURCE_GROUP}" \
  --name "${ACR_NAME}" \
  --sku Basic \
  --admin-enabled true \
  --output none || true

# 3. Build & push
echo "[3/5] Construyendo y empujando imagen Docker…"
az acr login --name "${ACR_NAME}"
docker build -t "${IMAGE_NAME}" "${REPO_ROOT}/api"
docker push "${IMAGE_NAME}"

# Obtener credenciales del ACR
ACR_USERNAME=$(az acr credential show \
  --resource-group "${RESOURCE_GROUP}" \
  --name "${ACR_NAME}" \
  --query "username" -o tsv)
ACR_PASSWORD=$(az acr credential show \
  --resource-group "${RESOURCE_GROUP}" \
  --name "${ACR_NAME}" \
  --query "passwords[0].value" -o tsv)

# 4. Eliminar container group anterior (si existe)
echo "[4/5] Eliminando Container Group anterior (si existe)…"
az container delete \
  --resource-group "${RESOURCE_GROUP}" \
  --name "${CONTAINER_GROUP}" \
  --yes \
  --output none || true

# 5. Crear Container Group
echo "[5/5] Creando Container Group en ACI…"
az container create \
  --resource-group "${RESOURCE_GROUP}" \
  --name "${CONTAINER_GROUP}" \
  --image "${IMAGE_NAME}" \
  --registry-login-server "${ACR_NAME}.azurecr.io" \
  --registry-username "${ACR_USERNAME}" \
  --registry-password "${ACR_PASSWORD}" \
  --cpu "${CPU}" \
  --memory "${MEMORY_GB}" \
  --dns-name-label "${DNS_LABEL}" \
  --ports 8000 \
  --restart-policy Always \
  --environment-variables \
    PCB_API_KEY="${PCB_API_KEY}" \
    BATCH_ENDPOINT_URL="${BATCH_ENDPOINT_URL}" \
    STORAGE_ACCOUNT="${STORAGE_ACCOUNT}" \
    PREDICTIONS_CONTAINER="${PREDICTIONS_CONTAINER}" \
    INPUT_CONTAINER="${INPUT_CONTAINER}" \
    POLL_INTERVAL_S="${POLL_INTERVAL_S}" \
    MAX_POLL_ATTEMPTS="${MAX_POLL_ATTEMPTS}" \
    CORS_ORIGINS="${CORS_ORIGINS}" \
  --secure-environment-variables \
    AZURE_ML_TOKEN="${AZURE_ML_TOKEN}" \
    STORAGE_KEY="${STORAGE_KEY}" \
    AZURE_STORAGE_CONNECTION_STRING="${AZURE_STORAGE_CONNECTION_STRING}" \
  --output none

FQDN=$(az container show \
  --resource-group "${RESOURCE_GROUP}" \
  --name "${CONTAINER_GROUP}" \
  --query "ipAddress.fqdn" -o tsv)

echo ""
echo "=========================================="
echo "✅ Deployment completado!"
echo "   API URL: http://${FQDN}:8000"
echo "   Health : http://${FQDN}:8000/health"
echo "   Docs   : http://${FQDN}:8000/docs"
echo "=========================================="
