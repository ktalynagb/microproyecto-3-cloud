Perfecto, hagamos el ticket **B1: [Arquitectura] Diseñar arquitectura lógica del prototipo (Streamlit + gRPC + inferencia)**. Abajo te dejo el **contenido listo para pegar en `docs/arquitectura.md`** (incluye diagrama + responsabilidades + flujo end-to-end + decisiones clave).

> Nota rápida: algunos archivos que habías subido en la conversación ya **expiraron**. Si quieres que esta arquitectura quede 100% alineada con un diagrama/plantilla específica de esos archivos, vuelve a subirlos y lo ajusto sobre ese material.

---

## Arquitectura lógica del prototipo (Streamlit + gRPC + inferencia)

### Propósito

El prototipo implementa una aplicación web en **Streamlit** que permite cargar imágenes (individual o lote) y obtener una clasificación **IA vs Real** usando un modelo preentrenado. Por requerimiento del curso, la comunicación entre la interfaz y el motor de inferencia se realiza mediante **gRPC**, separando claramente los módulos de presentación (GUI) y procesamiento (servicio de inferencia). El resultado se presenta por imagen con **confianza** y **tiempos**, y se consolida en un **CSV** descargable.

---

### Componentes y responsabilidades

**1) Interfaz (Streamlit / GUI)**

* Entrada de usuario: carga de imágenes (JPG/JPEG/PNG) en lote.
* Presentación: vista previa, estado por imagen (pendiente / procesando / éxito / error).
* Orquestación de análisis: botón “Analizar” que llama al cliente gRPC.
* Salida: tabla/tarjetas con predicción, probabilidades y tiempos; descarga CSV.
* Disclaimer visible: “herramienta de apoyo, no certificación forense”.

**2) Cliente gRPC (Adapter en la GUI)**

* Empaqueta cada imagen (bytes + metadata) según el contrato `.proto`.
* Envía requests al servicio de inferencia y recibe responses.
* Maneja errores de conectividad y timeouts (server caído, timeout, respuesta inválida).
* Devuelve a la GUI una estructura uniforme (éxito/error) para consolidación y CSV.

**3) Servicio de inferencia (Servidor gRPC)**

* Expone métodos gRPC para clasificación.
* Maneja decodificación de bytes → imagen (PIL), validación y errores por imagen.
* Ejecuta preprocesamiento con `AutoImageProcessor`.
* Ejecuta inferencia con `SiglipForImageClassification` en CPU (modo `eval`).
* Calcula probabilidades (softmax), predicción, y mide tiempos.
* Responde con estructura estándar (label, probs, tiempos, estado/error).

**4) Motor de inferencia (submódulo interno del servidor)**

* `ModelLoader`: carga única del modelo y processor al iniciar el servicio.
* `InferenceEngine`: funciones `preprocess()`, `predict()`, `timing()` y manejo de excepciones.

**5) Consolidación de resultados (Data/CSV)**

* Estructura estándar (schema) para resultados por imagen.
* Construye DataFrame del lote y genera CSV descargable desde la GUI.

---

### Diagrama lógico (alto nivel)

```mermaid
flowchart LR
  U[Usuario] -->|Carga imágenes| S[Streamlit GUI]
  S -->|Analizar| GC[gRPC Client Adapter]
  GC -->|Request: bytes + metadata| GS[gRPC Inference Server]
  GS --> IE[InferenceEngine]
  IE --> ML[ModelLoader + Processor]
  IE --> GS
  GS -->|Response: label + probs + tiempos + estado| GC
  GC --> S
  S -->|Resultados + CSV| U
```

---

### Flujo end-to-end (secuencia)

1. El usuario carga 1 o varias imágenes en Streamlit.
2. La GUI valida formatos y muestra previews/estado “Pendiente”.
3. Al presionar **Analizar**, la GUI invoca el **cliente gRPC** (por imagen o por lote).
4. El **servidor gRPC** recibe la solicitud:

   * decodifica bytes,
   * preprocesa,
   * ejecuta inferencia,
   * calcula probabilidades y tiempos,
   * retorna respuesta (éxito o error controlado).
5. La GUI recibe respuestas, actualiza estados por imagen y muestra tabla de resultados.
6. Se consolida un DataFrame y se habilita la **descarga CSV**.
7. Se mantiene visible el **disclaimer** (no forense, apoyo a verificación).

---

### Contrato de comunicación (nivel conceptual)

Para mantener el MVP simple y consistente, el contrato se define para devolver siempre una respuesta estructurada por imagen con:

* `predicted_label` (por ejemplo: `ai` / `human` o las etiquetas del modelo),
* `probabilities` (por clase),
* `preprocess_time_ms` y `inference_time_ms`,
* `status` (OK / ERROR),
* `error_message` (si aplica).

> Implementación recomendada: **RPC por imagen** (`ClassifyImage`) y la GUI itera el lote. Esto reduce complejidad inicial. Si el rendimiento lo exige, se puede añadir luego `ClassifyBatch` (nice-to-have).

---

### Manejo de errores (estrategia implementada)

#### Cliente gRPC (`app/clientGrpc.py`)

El cliente distingue y mapea los códigos de estado gRPC a mensajes amigables:

| Código gRPC          | Causa probable                          | Mensaje al usuario                                         |
|----------------------|-----------------------------------------|------------------------------------------------------------|
| `DEADLINE_EXCEEDED`  | Timeout de la llamada RPC               | Informa que se superó el timeout configurado               |
| `UNAVAILABLE`        | Servidor caído o no iniciado            | Informa que el servicio no está disponible                 |
| `INVALID_ARGUMENT`   | Payload de imagen corrupto o inválido   | Informa que el archivo no es JPG/PNG válido                |
| `INTERNAL`           | Error interno inesperado en el servidor | Informa del error interno e indica revisar logs            |
| `CANCELLED`          | Llamada cancelada por el cliente        | Informa que la llamada fue cancelada                       |
| `RESOURCE_EXHAUSTED` | Servidor sin recursos                   | Informa que el servidor no tiene recursos suficientes      |
| Otros códigos        | Cualquier otro error gRPC               | Mensaje genérico con el nombre del código                  |

**Métodos disponibles:**

* `classify_image(...)` — lanza `GRPCClientError` con mensaje amigable en caso de fallo.
  Útil cuando el llamador quiere manejar la excepción explícitamente.
* `classify_image_safe(...)` — **nunca lanza**; retorna un dict con `status="error"` y
  `error_message` con el mensaje amigable.  Recomendado para bucles de lote en la GUI,
  ya que un fallo en una imagen no aborta el procesamiento del resto.

**Conexión inicial:**

* Si el canal no está listo dentro del `timeout`, se lanza `GRPCClientError` con mensaje
  que indica el timeout y el servidor.
* `grpc.FutureTimeoutError` se captura explícitamente y se distingue de otros errores de
  conexión.

#### Servidor gRPC (`service/inference_server.py`)

* Verifica `context.is_active()` al inicio de cada llamada; si la solicitud fue cancelada
  por el cliente, responde con `CANCELLED` sin desperdiciar recursos.
* Errores de imagen inválida (`INVALID_IMAGE`) generan código `INVALID_ARGUMENT`.
* Cualquier excepción inesperada dentro del pipeline genera código `INTERNAL`.
* El servidor **nunca colapsa** por una solicitud fallida; continúa atendiendo el resto.

#### Errores parciales en el lote

* El lote continúa aunque falle una imagen (gracias a `classify_image_safe`).
* El CSV registra tanto resultados exitosos (`status=ok`) como errores (`status=error`),
  incluyendo el mensaje de error por imagen.

---

### Estructura del repositorio organización en módulos

* `app/`

  * `app.py` (Streamlit)
  * `clientGrpc.py` (adapter gRPC con manejo de errores y timeouts)
  * `batch_upload.py` (gestión del lote de imágenes)
  * `result_table.py` (consolidación y export CSV)
  * `streamlit_app.py` (ejemplo de integración completa)
* `service/`

  * `inference_server.py` (gRPC server con manejo de errores y status codes)
  * `inference/inference_engine.py` (preprocess + predict + timing)
  * `inference/model_loader.py` (carga modelo/processor)
* `proto/`

  * `inference.proto`
  * `generated/` (stubs)
* `tests/`

  * `test_client_grpc.py` (cliente: conexión, clasificación, errores, classify_image_safe)
  * `test_grpc_server.py` (servidor: inicio, respuestas, errores, status codes)
  * `test_grpc_stubs.py` (stubs generados)
  * `test_inference_engine.py` (motor de inferencia)
  * `test_preprocessing.py` (preprocesamiento)
* `docs/`

  * `arquitectura.md` (este documento)


### Decisiones de diseño (por qué así)

* **Separación GUI vs inferencia**: mejora claridad, permite cumplir requisito gRPC y simplifica pruebas.
* **gRPC mínimo viable**: suficiente para comunicación obligatoria sin complejidad innecesaria.
* **Inferencia en CPU**: coherente con alcance acotado; medición de tiempos permite evidenciar limitaciones.
* **Respuesta estandarizada**: facilita tabla en GUI, CSV y pruebas automatizadas.
* **Errores amigables**: `_grpc_error_message()` centraliza el mapeo código→texto para facilitar la GUI.
* **`classify_image_safe`**: patrón "nunca lanza" para bucles de lote; evita que un fallo en una imagen
  aborte todo el procesamiento.