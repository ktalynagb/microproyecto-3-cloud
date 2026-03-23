# Microproyecto 3 - Computación en la Nube

## Despliegue en Azure Container Apps

Ejecuta este bloque en la terminal para definir los parámetros del proyecto. Asegúrate de modificar la región si es necesario.

```bash
LOCATION="canadacentral"
PROJECT="pcb-defect-inspection"
RG_APP="${PROJECT}-app"
APP_NAME="${PROJECT}-app"
ENV_APP="${APP_NAME}-env"
ACR_NAME="uaopcbdefect"
```

Ahora crearemos la infraestructura base para el Frontend. Asegúrate de ejecutar estos comandos desde el directorio donde se encuentra el Dockerfile. Primero, creamos el grupo de recursos, el registro de contenedores y extraemos las credenciales de acceso:

```bash
az group create --name $RG_APP --location $LOCATION
az acr create --resource-group $RG_APP --name $ACR_NAME --sku Basic --admin-enabled true
ACR_SERVER=$(az acr show --name $ACR_NAME --resource-group $RG_APP --query loginServer -o tsv)
ACR_USER=$(az acr credential show --name $ACR_NAME --resource-group $RG_APP --query username -o tsv)
ACR_PASS=$(az acr credential show --name $ACR_NAME --resource-group $RG_APP --query "passwords[0].value" -o tsv)
```

A continuación, construimos la imagen de Docker y desplegamos el entorno y la aplicación del Frontend indicando el puerto 8501, que es el que utiliza Streamlit por defecto:

```bash
az acr login -n $ACR_USER
docker build -t $APP_NAME:latest -f Dockerfile .
docker tag $APP_NAME:latest $ACR_SERVER/$APP_NAME:latest
docker push $ACR_SERVER/$APP_NAME:latest
az acr repository list --name $ACR_NAME --output table
az containerapp env create --name $ENV_APP --resource-group $RG_APP --location $LOCATION
az containerapp create --name $APP_NAME --resource-group $RG_APP --environment $ENV_APP --image $ACR_SERVER/$APP_NAME:latest --target-port 8501 --ingress external --registry-server $ACR_SERVER --registry-username $ACR_USER --registry-password $ACR_PASS
```


## Despliegue en AWS ECS Fargate

Ejecuta este bloque en la terminal para definir los parámetros. Asegúrate de modificar la región si es necesario.

```bash
REGION="us-east-2"
PROJECT="pcb-defect-inspection"
APP_NAME="${PROJECT}-app"
ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
ECR_URI="${ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com/${APP_NAME}"
```

Asegúrate de estar en el directorio donde se ubica el Dockerfile. Ejecuta estos comandos para crear el repositorio, autenticar Docker y subir la imagen.

```bash
aws ecr create-repository --repository-name $APP_NAME --region $REGION
aws ecr get-login-password --region $REGION | docker login --username AWS --password-stdin "${ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com"
docker build -t $APP_NAME -f Dockerfile .
docker tag "${APP_NAME}:latest" "${ECR_URI}:latest"
docker push "${ECR_URI}:latest"
```

ECS necesita permisos para descargar la imagen de ECR. Crearemos el rol y luego configuraremos el Grupo de Seguridad para permitir el tráfico web hacia Streamlit.

```bash
aws iam create-role --role-name ecsTaskExecutionRole --assume-role-policy-document file://.aws/ecs-tasks-trust-policy.json
aws iam attach-role-policy --role-name ecsTaskExecutionRole --policy-arn arn:aws:iam::aws:policy/service-role/AmazonECSTaskExecutionRolePolicy
VPC_ID=$(aws ec2 describe-vpcs --filters Name=isDefault,Values=true --query "Vpcs[0].VpcId" --output text)
SUBNET_ID=$(aws ec2 describe-subnets --filters Name=vpc-id,Values=$VPC_ID --query "Subnets[0].SubnetId" --output text)
SG_ID=$(aws ec2 create-security-group --group-name "${APP_NAME}-sg" --description "Frontend SG" --vpc-id $VPC_ID --query "GroupId" --output text)
aws ec2 authorize-security-group-ingress --group-id $SG_ID --protocol tcp --port 8501 --cidr 0.0.0.0/0
```

Ejecutaremos estos comandos para crear los componentes del balanceador de carga utilizando las variables de red existentes.

```bash
SUBNET_ID_2=$(aws ec2 describe-subnets --filters Name=vpc-id,Values=$VPC_ID --query "Subnets[1].SubnetId" --output text)
ALB_ARN=$(aws elbv2 create-load-balancer --name "${APP_NAME}-alb" --subnets $SUBNET_ID $SUBNET_ID_2 --security-groups $SG_ID --query "LoadBalancers[0].LoadBalancerArn" --output text)
TG_ARN=$(aws elbv2 create-target-group --name "${APP_NAME}-tg" --protocol HTTP --port 8501 --vpc-id $VPC_ID --target-type ip --query "TargetGroups[0].TargetGroupArn" --output text)
aws elbv2 create-listener --load-balancer-arn $ALB_ARN --protocol HTTP --port 80 --default-actions Type=forward,TargetGroupArn=$TG_ARN
```

Ahora definiremos las características de la máquina virtual que ejecutará el contenedor. Para generar la definición de la tarea ejecutamos el siguiente comando desde una terminal o copiando el archivo `.aws/ecs-task-definition-template.json` como `.aws/ecs-task-definition.json` y configurando manualmente los valores:

```bash
echo '{"family":"'${APP_NAME}'","networkMode":"awsvpc","containerDefinitions":[{"name":"'${APP_NAME}'","image":"'${ECR_URI}':latest","portMappings":[{"containerPort":8501,"hostPort":8501,"protocol":"tcp"}]}],"requiresCompatibilities":["FARGATE"],"cpu":"1024","memory":"2048","executionRoleArn":"arn:aws:iam::'${ACCOUNT_ID}':role/ecsTaskExecutionRole"}' > .aws/ecs-task-definition.json
```

Una vez definida la tarea crearemos el servicio para mantener la aplicación en línea.

```bash
aws ecs register-task-definition --cli-input-json file://.aws/ecs-task-definition.json
aws ecs create-cluster --cluster-name $APP_NAME
aws ecs create-service --cluster $APP_NAME --service-name $APP_NAME --task-definition $APP_NAME --desired-count 1 --launch-type FARGATE --network-configuration "awsvpcConfiguration={subnets=[$SUBNET_ID],securityGroups=[$SG_ID],assignPublicIp=ENABLED}" --load-balancers "targetGroupArn=$TG_ARN,containerName=$APP_NAME,containerPort=8501"
```

Finalmente, utilizaremos este comando para capturar y mostrar en la terminal la URL estable que AWS ha generado para el Frontend.

```bash
ALB_URL=$(aws elbv2 describe-load-balancers --load-balancer-arns $ALB_ARN --query "LoadBalancers[0].DNSName" --output text)
echo $ALB_URL
```