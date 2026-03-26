#!/usr/bin/env bash
# =============================================================================
# deploy_api.sh – Despliega el backend FastAPI a Azure Container Instances (ACI)
# =============================================================================
# Uso:
#   bash deployment/azure/deploy_api.sh
#
# Prerrequisitos:
#   - Azure CLI instalado (az)
#   - Sesión activa (az login) o variables de entorno de service principal
#   - Variables de entorno de .env cargadas (source .env)
#   - Docker instalado y activo
# =============================================================================

set -euo pipefail

# ── Configuración (con defaults) ─────────────────────────────────────────────

: "${AZURE_SUBSCRIPTION_ID:?Variable AZURE_SUBSCRIPTION_ID no definida}"
: "${AZURE_RESOURCE_GROUP:=pcb-ml-rg}"
: "${AZURE_LOCATION:=centralus}"

# Azure Container Registry
: "${ACR_NAME:=pcbmlacr}"
: "${IMAGE_NAME:=pcb-backend}"
: "${IMAGE_TAG:=latest}"

# Azure Container Instances
: "${ACI_NAME:=pcb-backend-aci}"
: "${ACI_DNS_LABEL:=pcb-backend-api}"
: "${ACI_CPU:=1}"
: "${ACI_MEMORY:=1.5}"
: "${API_PORT:=8080}"

# ── Colores ───────────────────────────────────────────────────────────────────

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

info()  { echo -e "${GREEN}[INFO]${NC}  $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC}  $*"; }
error() { echo -e "${RED}[ERROR]${NC} $*" >&2; }

# ── Step 1: Seleccionar suscripción ──────────────────────────────────────────

info "Seleccionando suscripción: $AZURE_SUBSCRIPTION_ID"
az account set --subscription "$AZURE_SUBSCRIPTION_ID"

# ── Step 2: Crear Resource Group si no existe ────────────────────────────────

info "Verificando Resource Group: $AZURE_RESOURCE_GROUP"
az group create \
  --name "$AZURE_RESOURCE_GROUP" \
  --location "$AZURE_LOCATION" \
  --output none

# ── Step 3: Crear ACR si no existe ───────────────────────────────────────────

info "Verificando Azure Container Registry: $ACR_NAME"
if ! az acr show --name "$ACR_NAME" --resource-group "$AZURE_RESOURCE_GROUP" --output none 2>/dev/null; then
  info "Creando Azure Container Registry: $ACR_NAME"
  az acr create \
    --name "$ACR_NAME" \
    --resource-group "$AZURE_RESOURCE_GROUP" \
    --sku Basic \
    --admin-enabled true \
    --output none
fi

ACR_LOGIN_SERVER=$(az acr show \
  --name "$ACR_NAME" \
  --query loginServer \
  --output tsv)

ACR_USERNAME=$(az acr credential show \
  --name "$ACR_NAME" \
  --query username \
  --output tsv)

ACR_PASSWORD=$(az acr credential show \
  --name "$ACR_NAME" \
  --query "passwords[0].value" \
  --output tsv)

# ── Step 4: Build y Push de la imagen ────────────────────────────────────────

FULL_IMAGE="${ACR_LOGIN_SERVER}/${IMAGE_NAME}:${IMAGE_TAG}"
info "Construyendo imagen: $FULL_IMAGE"

docker build \
  --tag "$FULL_IMAGE" \
  --file deployment/api/Dockerfile \
  deployment/api/

info "Haciendo push al ACR..."
az acr login --name "$ACR_NAME"
docker push "$FULL_IMAGE"

# ── Step 5: Desplegar a ACI ───────────────────────────────────────────────────

info "Desplegando $ACI_NAME en Azure Container Instances..."

az container create \
  --resource-group "$AZURE_RESOURCE_GROUP" \
  --name "$ACI_NAME" \
  --image "$FULL_IMAGE" \
  --registry-login-server "$ACR_LOGIN_SERVER" \
  --registry-username "$ACR_USERNAME" \
  --registry-password "$ACR_PASSWORD" \
  --dns-name-label "$ACI_DNS_LABEL" \
  --location "$AZURE_LOCATION" \
  --cpu "$ACI_CPU" \
  --memory "$ACI_MEMORY" \
  --ports "$API_PORT" \
  --environment-variables \
      API_HOST=0.0.0.0 \
      API_PORT="$API_PORT" \
      LOG_LEVEL="${LOG_LEVEL:-INFO}" \
      AZURE_STORAGE_ACCOUNT="${AZURE_STORAGE_ACCOUNT:-}" \
      AZURE_INPUT_CONTAINER="${AZURE_INPUT_CONTAINER:-}" \
      AZURE_OUTPUT_CONTAINER="${AZURE_OUTPUT_CONTAINER:-pcb-results}" \
      AZURE_ML_BATCH_ENDPOINT_URL="${AZURE_ML_BATCH_ENDPOINT_URL:-}" \
      CORS_EXTRA_ORIGINS="${CORS_EXTRA_ORIGINS:-}" \
  --secure-environment-variables \
      BACKEND_API_KEY="${BACKEND_API_KEY:-}" \
      AZURE_STORAGE_KEY="${AZURE_STORAGE_KEY:-}" \
      AZURE_ML_API_KEY="${AZURE_ML_API_KEY:-}" \
  --output table

# ── Step 6: Mostrar la URL pública ────────────────────────────────────────────

FQDN=$(az container show \
  --resource-group "$AZURE_RESOURCE_GROUP" \
  --name "$ACI_NAME" \
  --query ipAddress.fqdn \
  --output tsv 2>/dev/null || echo "pendiente")

info "✅ Deploy completado"
echo ""
echo "  Backend URL : http://${FQDN}:${API_PORT}"
echo "  Health check: http://${FQDN}:${API_PORT}/api/v1/health"
echo "  API Docs    : http://${FQDN}:${API_PORT}/docs"
echo ""
warn "Actualiza AZURE_BACKEND_URL en tu .env del frontend con la URL anterior."
