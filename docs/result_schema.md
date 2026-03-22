# Result Schema — AI vs Real

Este documento define el esquema estándar de resultados por imagen para garantizar consistencia entre:
- GUI (Streamlit)
- Respuesta gRPC
- Exportación a CSV y reporte PDF

## Columnas obligatorias

| Columna | Tipo | Descripción |
|---|---|---|
| timestamp | string (ISO 8601) | Momento de generación del resultado (UTC recomendado). |
| filename | string | Nombre del archivo de imagen. |
| status | enum: ok/error | Estado del resultado por imagen. |
| predicted_label | enum: AI/REAL o null | Etiqueta predicha (solo si status=ok). |
| prob_ai | float o null | Probabilidad de clase AI (solo si status=ok). |
| prob_real | float o null | Probabilidad de clase REAL (solo si status=ok). |
| preprocess_time_ms | float o null | Tiempo de preprocesamiento en milisegundos. |
| inference_time_ms | float o null | Tiempo de inferencia en milisegundos. |
| error_message | string o null | Mensaje de error (solo si status=error). |

## Reglas

- status=ok:
  - predicted_label, prob_ai, prob_real deben existir
  - prob_ai, prob_real ∈ [0, 1]
  - prob_ai + prob_real ≈ 1 
- status=error:
  - error_message debe existir
  - predicted_label / prob_* pueden ser null

## Ejemplos

### OK
timestamp=2026-03-01T23:45:12Z  
filename=img_01.png  
status=ok  
predicted_label=AI  
prob_ai=0.73  
prob_real=0.27  
preprocess_time_ms=12.4  
inference_time_ms=98.7  
error_message=

### ERROR
timestamp=2026-03-01T23:45:12Z  
filename=img_02.png  
status=error  
predicted_label=  
prob_ai=  
prob_real=  
preprocess_time_ms=  
inference_time_ms=  
error_message=Server unavailable
