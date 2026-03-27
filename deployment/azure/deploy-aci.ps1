$ErrorActionPreference = "Stop"

Write-Host "=============================================================================" -ForegroundColor Cyan
Write-Host " deploy-aci.ps1 - Despliegue de Backend y Frontend a Azure Container Instances" -ForegroundColor Cyan
Write-Host "=============================================================================" -ForegroundColor Cyan

# ── 1. Cargar variables desde .env ──────────────────────────────────────────
$envFilePath = "deployment\azure\.env"
if (Test-Path $envFilePath) {
    Write-Host "Cargando variables desde $envFilePath..." -ForegroundColor Yellow
    Get-Content $envFilePath | Where-Object { $_ -match '=' -and $_ -notmatch '^#' } | ForEach-Object {
        $name, $value = $_.Split('=', 2)
        Set-Item -Path "Env:$name" -Value $value.Trim()
    }
} else {
    Write-Host "Advertencia: No se encontró $envFilePath. Asegúrate de tener las variables creadas." -ForegroundColor Red
}

# ── Configuración (con defaults) ─────────────────────────────────────────────
$AZURE_SUBSCRIPTION_ID = if ($env:AZURE_SUBSCRIPTION_ID) { $env:AZURE_SUBSCRIPTION_ID } else { throw "Falta AZURE_SUBSCRIPTION_ID en el .env" }
$AZURE_RESOURCE_GROUP = if ($env:AZURE_RESOURCE_GROUP) { $env:AZURE_RESOURCE_GROUP } else { "pcb-ml-rg" }
$ACR_NAME = if ($env:ACR_NAME) { $env:ACR_NAME } else { "pcbmlacr" }
$IMAGE_BACKEND = if ($env:IMAGE_BACKEND) { $env:IMAGE_BACKEND } else { "pcb-backend" }
$IMAGE_FRONTEND = if ($env:IMAGE_FRONTEND) { $env:IMAGE_FRONTEND } else { "pcb-frontend" }
$IMAGE_TAG = if ($env:IMAGE_TAG) { $env:IMAGE_TAG } else { "latest" }
$ACI_BACKEND_NAME = if ($env:ACI_BACKEND_NAME) { $env:ACI_BACKEND_NAME } else { "pcb-backend-aci" }
$ACI_FRONTEND_NAME = if ($env:ACI_FRONTEND_NAME) { $env:ACI_FRONTEND_NAME } else { "pcb-frontend-aci" }
$BACKEND_API_KEY = if ($env:BACKEND_API_KEY) { $env:BACKEND_API_KEY } else { "pcb-api-key-super-secreto" }
$API_PORT = if ($env:API_PORT) { $env:API_PORT } else { "8080" }

# ── 2. Login y Preparación ──────────────────────────────────────────────────
Write-Host "`n── Step 1: Configurando suscripción ──" -ForegroundColor Green
az account set --subscription $AZURE_SUBSCRIPTION_ID

Write-Host "`n── Step 2: Login en ACR ($ACR_NAME) ──" -ForegroundColor Green
az acr login --name $ACR_NAME
$ACR_LOGIN_SERVER = "$ACR_NAME.azurecr.io"
$ACR_PASSWORD = az acr credential show --name $ACR_NAME --query "passwords[0].value" -o tsv

# ── 3. Construir y Subir Imágenes ───────────────────────────────────────────
Write-Host "`n── Step 3: Construir y Subir Backend ──" -ForegroundColor Green
docker build -t "${IMAGE_BACKEND}:${IMAGE_TAG}" -f deployment/api/Dockerfile .
docker tag "${IMAGE_BACKEND}:${IMAGE_TAG}" "${ACR_LOGIN_SERVER}/${IMAGE_BACKEND}:${IMAGE_TAG}"
docker push "${ACR_LOGIN_SERVER}/${IMAGE_BACKEND}:${IMAGE_TAG}"

Write-Host "`n── Step 4: Construir y Subir Frontend ──" -ForegroundColor Green
docker build -t "${IMAGE_FRONTEND}:${IMAGE_TAG}" -f Dockerfile .
docker tag "${IMAGE_FRONTEND}:${IMAGE_TAG}" "${ACR_LOGIN_SERVER}/${IMAGE_FRONTEND}:${IMAGE_TAG}"
docker push "${ACR_LOGIN_SERVER}/${IMAGE_FRONTEND}:${IMAGE_TAG}"

# ── 4. Despliegue de Instancias ──────────────────────────────────────────────
Write-Host "`n── Step 5: Desplegando Backend ACI ──" -ForegroundColor Green
az container create `
  --resource-group $AZURE_RESOURCE_GROUP `
  --name $ACI_BACKEND_NAME `
  --image "${ACR_LOGIN_SERVER}/${IMAGE_BACKEND}:${IMAGE_TAG}" `
  --registry-login-server $ACR_LOGIN_SERVER `
  --registry-username $ACR_NAME `
  --registry-password $ACR_PASSWORD `
  --dns-name-label $ACI_BACKEND_NAME `
  --ports $API_PORT `
  --assign-identity `
  --os-type Linux `
  --environment-variables "BACKEND_API_KEY=$BACKEND_API_KEY" "API_PORT=$API_PORT" "LOG_LEVEL=INFO" "AZURE_CLIENT_ID=system" `
  --cpu 1 --memory 2

Write-Host "`n── Step 6: Obteniendo URL del Backend ──" -ForegroundColor Green
$BACKEND_FQDN = az container show --resource-group $AZURE_RESOURCE_GROUP --name $ACI_BACKEND_NAME --query ipAddress.fqdn --output tsv
$API_URL = "http://${BACKEND_FQDN}:${API_PORT}"

Write-Host "`n── Step 7: Desplegando Frontend ACI ──" -ForegroundColor Green
az container create `
  --resource-group $AZURE_RESOURCE_GROUP `
  --name $ACI_FRONTEND_NAME `
  --image "${ACR_LOGIN_SERVER}/${IMAGE_FRONTEND}:${IMAGE_TAG}" `
  --registry-login-server $ACR_LOGIN_SERVER `
  --registry-username $ACR_NAME `
  --registry-password $ACR_PASSWORD `
  --dns-name-label $ACI_FRONTEND_NAME `
  --ports 8501 `
  --os-type Linux `
  --environment-variables "API_URL=$API_URL" "BACKEND_API_KEY=$BACKEND_API_KEY" `
  --cpu 1 --memory 2

$FRONTEND_FQDN = az container show --resource-group $AZURE_RESOURCE_GROUP --name $ACI_FRONTEND_NAME --query ipAddress.fqdn --output tsv

# ── 5. Resumen Final ─────────────────────────────────────────────────────────
Write-Host "`n  ==========================================================" -ForegroundColor Cyan
Write-Host "  🚀 Deploy completado con éxito en Azure ACI" -ForegroundColor Cyan
Write-Host "  ==========================================================" -ForegroundColor Cyan
Write-Host "  Backend:"
Write-Host "    API URL   : $API_URL"
Write-Host "    Health    : $API_URL/api/v1/health"
Write-Host "    API Docs  : $API_URL/docs"
Write-Host "  ----------------------------------------------------------"
Write-Host "  Frontend (Streamlit):"
Write-Host "    Web URL   : http://${FRONTEND_FQDN}:8501"
Write-Host "  ==========================================================" -ForegroundColor Cyan