"""Pipeline de batch inference para detección de defectos en PCB.

Módulos:
    inference_engine  - ETAPA 2: Inferencia YOLOv8n (BatchInference, BatchResult)
    batch_receiver    - ETAPA 1: Recepción y resize (BatchReceiver, Batch)
    post_processor    - ETAPA 3: Anotaciones y visualización (PostProcessor)
    blob_exporter     - ETAPA 4: Exportación a Blob Storage (BlobExporter)
    delivery          - ETAPA 5: URLs SAS de descarga (DeliveryService)
    config            - Configuración centralizada (AppConfig, config)
    utils             - Utilidades de imagen
    logger            - Logging estructurado JSON
    score             - Script de scoring para Azure ML Batch Endpoint
"""
