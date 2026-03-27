#!/usr/bin/env bash
# =============================================================================
# deploy-aci.sh – Despliega backend y frontend a Azure Container Instances
#                 con Managed Identity para autenticación automática
# =============================================================================
# Uso (desde la raíz del repositorio):
#   source deployment/azure/.env
#   bash deployment/azure/deploy-aci.sh
#
# Prerrequisitos:
#   - Azure CLI instalado y autenticado (az login)
#   - Docker instalado y activo
#   - Variables de entorno cargadas desde deployment/azure/.env
#
# Flujo de autenticación:
#   Local dev  →  ~/.azure montado  →  Azure CLI obtiene tokens dinámicamente
#   Azure ACI  →  Managed Identity  →  IMDS proporciona tokens automáticamente
# =============================================================================

set -euo pipefail

# ── Configuración (con defaults) ─────────────────────────────────────────────

: "${AZURE_SUBSCRIPTION_ID:?Variable AZURE_SUBSCRIPTION_ID no definida}"
: "${AZURE_RESOURCE_GROUP:=pcb-ml-rg}"
: "${AZURE_LOCATION:=centralus}"

# Azure Container Registry
: "${ACR_NAME:=pcbmlacr}"
: "${IMAGE_BACKEND:=pcb-backend}"
: "${IMAGE_FRONTEND:=pcb-frontend}"
: "${IMAGE_TAG:=latest}"

# Azure Container Instances
: "${ACI_BACKEND_NAME:=pcb-backend-aci}"
: "${ACI_FRONTEND_NAME:=pcb-frontend-aci}"
: "${ACI_BACKEND_DNS:=pcb-backend-api}"
: "${ACI_FRONTEND_DNS:=pcb-frontend-app}"
: "${ACI_CPU:=1}"
: "${ACI_MEMORY:=1.5}"
: "${API_PORT:=8080}"
: "${FRONTEND_PORT:=8501}"

# ── Colores ───────────────────────────────────────────────────────────────────

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

info()    { echo -e "${GREEN}[INFO]${NC}  $*"; }
warn()    { echo -e "${YELLOW}[WARN]${NC}  $*"; }
error()   { echo -e "${RED}[ERROR]${NC} $*" >&2; }
section() { echo -e "\n${GREEN}═══ $* ═══${NC}"; }

# ── Step 1: Seleccionar suscripción ──────────────────────────────────────────

section "Step 1: Seleccionar suscripción"
info "Suscripción: $AZURE_SUBSCRIPTION_ID"
az account set --subscription "$AZURE_SUBSCRIPTION_ID"

# ── Step 2: Crear Resource Group si no existe ────────────────────────────────

section "Step 2: Resource Group"
info "Verificando: $AZURE_RESOURCE_GROUP"
az group create \
  --name "$AZURE_RESOURCE_GROUP" \
  --location "$AZURE_LOCATION" \
  --output none

# ── Step 3: Crear ACR si no existe ───────────────────────────────────────────

section "Step 3: Azure Container Registry"
info "Verificando ACR: $ACR_NAME"
if ! az acr show --name "$ACR_NAME" --resource-group "$AZURE_RESOURCE_GROUP" --output none 2>/dev/null; then
  info "Creando ACR: $ACR_NAME"
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

# ── Step 4: Build y Push de imágenes ─────────────────────────────────────────

section "Step 4: Build y Push de imágenes"

FULL_BACKEND="${ACR_LOGIN_SERVER}/${IMAGE_BACKEND}:${IMAGE_TAG}"
FULL_FRONTEND="${ACR_LOGIN_SERVER}/${IMAGE_FRONTEND}:${IMAGE_TAG}"

info "Construyendo backend: $FULL_BACKEND"
docker build \
  --tag "$FULL_BACKEND" \
  --file deployment/api/Dockerfile \
  deployment/api/

info "Construyendo frontend: $FULL_FRONTEND"
docker build \
  --tag "$FULL_FRONTEND" \
  --file Dockerfile \
  .

info "Push al ACR..."
az acr login --name "$ACR_NAME"
docker push "$FULL_BACKEND"
docker push "$FULL_FRONTEND"

# ── Step 5: Desplegar Backend con Managed Identity ───────────────────────────

section "Step 5: Backend ACI con Managed Identity"
info "Desplegando $ACI_BACKEND_NAME..."

az container create \
  --resource-group "$AZURE_RESOURCE_GROUP" \
  --name "$ACI_BACKEND_NAME" \
  --image "$FULL_BACKEND" \
  --registry-login-server "$ACR_LOGIN_SERVER" \
  --registry-username "$ACR_USERNAME" \
  --registry-password "$ACR_PASSWORD" \
  --dns-name-label "$ACI_BACKEND_DNS" \
  --location "$AZURE_LOCATION" \
  --cpu "$ACI_CPU" \
  --memory "$ACI_MEMORY" \
  --ports "$API_PORT" \
  --assign-identity \
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
  --output table

info "✅ Backend desplegado"

# ── Step 6: Asignar roles RBAC a la Managed Identity del backend ──────────────

section "Step 6: Asignar roles RBAC"

BACKEND_PRINCIPAL_ID=$(az container show \
  --resource-group "$AZURE_RESOURCE_GROUP" \
  --name "$ACI_BACKEND_NAME" \
  --query identity.principalId \
  --output tsv 2>/dev/null || echo "")

if [ -z "$BACKEND_PRINCIPAL_ID" ]; then
  warn "No se pudo obtener el principalId de la Managed Identity del backend."
  warn "Asigna los roles manualmente desde Azure Portal."
else
  SCOPE="/subscriptions/${AZURE_SUBSCRIPTION_ID}/resourceGroups/${AZURE_RESOURCE_GROUP}"

  info "Asignando rol 'Contributor' al scope del resource group..."
  az role assignment create \
    --assignee "$BACKEND_PRINCIPAL_ID" \
    --role "Contributor" \
    --scope "$SCOPE" \
    --output none 2>/dev/null || warn "El rol 'Contributor' ya está asignado o no tienes permisos"

  info "Asignando rol 'Storage Blob Data Contributor'..."
  az role assignment create \
    --assignee "$BACKEND_PRINCIPAL_ID" \
    --role "Storage Blob Data Contributor" \
    --scope "$SCOPE" \
    --output none 2>/dev/null || warn "El rol ya está asignado o no tienes permisos"

  info "✅ Roles RBAC asignados a la Managed Identity"
fi

# ── Step 7: Desplegar Frontend ────────────────────────────────────────────────

section "Step 7: Frontend ACI"

BACKEND_FQDN=$(az container show \
  --resource-group "$AZURE_RESOURCE_GROUP" \
  --name "$ACI_BACKEND_NAME" \
  --query ipAddress.fqdn \
  --output tsv 2>/dev/null || echo "")

BACKEND_URL="http://${BACKEND_FQDN}:${API_PORT}"
info "Frontend conectará al backend: $BACKEND_URL"

az container create \
  --resource-group "$AZURE_RESOURCE_GROUP" \
  --name "$ACI_FRONTEND_NAME" \
  --image "$FULL_FRONTEND" \
  --registry-login-server "$ACR_LOGIN_SERVER" \
  --registry-username "$ACR_USERNAME" \
  --registry-password "$ACR_PASSWORD" \
  --dns-name-label "$ACI_FRONTEND_DNS" \
  --location "$AZURE_LOCATION" \
  --cpu "$ACI_CPU" \
  --memory "$ACI_MEMORY" \
  --ports "$FRONTEND_PORT" \
  --environment-variables \
      AZURE_BACKEND_URL="$BACKEND_URL" \
      API_HOST="${BACKEND_FQDN}" \
      API_PORT="$API_PORT" \
      API_TIMEOUT=60 \
  --output table

info "✅ Frontend desplegado"

# ── Step 8: Mostrar URLs públicas ─────────────────────────────────────────────

section "Step 8: URLs públicas"

BACKEND_FQDN=$(az container show \
  --resource-group "$AZURE_RESOURCE_GROUP" \
  --name "$ACI_BACKEND_NAME" \
  --query ipAddress.fqdn \
  --output tsv 2>/dev/null || echo "pendiente")

FRONTEND_FQDN=$(az container show \
  --resource-group "$AZURE_RESOURCE_GROUP" \
  --name "$ACI_FRONTEND_NAME" \
  --query ipAddress.fqdn \
  --output tsv 2>/dev/null || echo "pendiente")

echo ""
echo "  ╔══════════════════════════════════════════════════════════════╗"
echo "  ║  Deploy completado con Managed Identity                      ║"
echo "  ╠══════════════════════════════════════════════════════════════╣"
echo "  ║  Backend                                                      ║"
echo "  ║    API URL   : http://${BACKEND_FQDN}:${API_PORT}"
echo "  ║    Health    : http://${BACKEND_FQDN}:${API_PORT}/api/v1/health"
echo "  ║    API Docs  : http://${BACKEND_FQDN}:${API_PORT}/docs"
echo "  ╠══════════════════════════════════════════════════════════════╣"
echo "  ║  Frontend                                                     ║"
echo "  ║    App URL   : http://${FRONTEND_FQDN}:${FRONTEND_PORT}"
echo "  ╚══════════════════════════════════════════════════════════════╝"
echo ""

warn "Autenticación: el backend usa Managed Identity (IMDS)."
warn "No se necesitan credenciales ni tokens manuales en ACI."
