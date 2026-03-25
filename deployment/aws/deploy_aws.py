import boto3
import sagemaker
from sagemaker import Estimator
from dotenv import load_dotenv

# 1. cargar credenciales de AWS desde el .env 
load_dotenv()

def setup_aws_infra():
    try:
        # 2. configuración de sesión
        session = sagemaker.Session()
        role = "AmazonSageMaker-ExecutionRole-PCB" # El rol debe existir en IAM
        
        print("Configurando instancia de entrenamiento en AWS SageMaker...")

        # 3. definición del Estimador (Alternativa a Azure ML Job)
        # Usamos ml.m5.xlarge (4 vCPU, 16GB RAM) 
        pcb_estimator = Estimator(
            image_uri=None, # Usará la imagen base de PyTorch de AWS
            role=role,
            instance_count=1,
            instance_type='ml.m5.xlarge',
            volume_size=30,
            max_run=3600, 
            sagemaker_session=session
        )

        print("Infraestructura de AWS definida")
        
    except Exception as e:
        print(f"Error conectando a AWS: {e}. Revisa tus credenciales en el .env")

if __name__ == "__main__":
    setup_aws_infra()