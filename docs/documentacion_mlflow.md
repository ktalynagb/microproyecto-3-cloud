# MLflow en el pipeline de ImageAivsReal

## ¿Qué es MLflow y para qué lo usamos?

[MLflow](https://mlflow.org/) es una plataforma de código abierto para gestionar
el ciclo de vida del aprendizaje automático. En este proyecto lo usamos
exclusivamente como **herramienta de observabilidad**: registramos métricas y
etiquetas cada vez que el servicio de inferencia carga el modelo, lo que permite:

- Comprobar fácilmente que el modelo se cargó correctamente.
- Ver el historial de cargas (desde qué fuente, en qué dispositivo, etc.).
- Detectar fallos de arranque del servicio antes de que lleguen peticiones reales.

---

## ¿Dónde se guardan los datos?

Los datos de MLflow se almacenan en un archivo SQLite local:

```
mlflow.db   (en la raíz del proyecto)
```

Tanto el health-check como la UI de MLflow apuntan a este mismo fichero,
de modo que todo lo que registra el health-check es inmediatamente visible
en la interfaz web.

La URI de seguimiento se puede cambiar definiendo la variable de entorno
`MLFLOW_TRACKING_URI` (en el fichero `.env` o en el entorno del sistema).
Si no se define, se usa `sqlite:///mlflow.db` como valor por defecto.

---

## Componentes del pipeline relacionados con MLflow

```
┌─────────────────────────────────────────────────────┐
│                  Pipeline de arranque                │
│                                                     │
│  make link_model  ──►  mlflow_health_check.py       │
│  make healthcheck ──►  mlflow_health_check.py       │
│                             │                       │
│                             ▼                       │
│                    model_loader.py                  │
│                    init_inference_artifacts()        │
│                       (descarga / carga modelo)     │
│                             │                       │
│                             ▼                       │
│                    report_loaded_to_mlflow()         │
│                       (escribe tags en mlflow.db)   │
└─────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────┐
│                  Visualización                      │
│                                                     │
│  make mlflow  ──►  mlflow ui  ──►  mlflow.db        │
│                      (http://127.0.0.1:5000)        │
└─────────────────────────────────────────────────────┘
```

### Archivos involucrados

| Archivo | Rol |
|---------|-----|
| `service/inference/mlflow_health_check.py` | Punto de entrada del health-check; configura el experimento y la URI de MLflow, llama a `model_loader` |
| `service/inference/model_loader.py` | Carga el modelo desde HuggingFace y registra los tags de resultado en MLflow |
| `mlflow.db` | Base de datos SQLite donde se guardan todos los experimentos y ejecuciones |

---

## Comandos Make relacionados con MLflow

### `make link_model`
Establece automáticamente el modelo predeterminado (`Ateeqq/ai-vs-human-image-detector`)
y ejecuta el health-check completo.

```bash
make link_model
```

Internamente ejecuta:
```bash
HF_MODEL_ID=Ateeqq/ai-vs-human-image-detector uv run -m service.inference.mlflow_health_check
```

**Cuándo usarlo:** primera vez que se configura el entorno, o al cambiar de modelo.

---

### `make healthcheck`
Ejecuta el health-check usando el valor de `HF_MODEL_ID` definido en `.env`
(o en el entorno del sistema).

```bash
make healthcheck
```

Internamente ejecuta:
```bash
uv run -m service.inference.mlflow_health_check
```

**Requisito:** `HF_MODEL_ID` debe estar definido en `.env` o en el entorno.
El fichero `.env` ya incluye el valor por defecto:
```
HF_MODEL_ID=Ateeqq/ai-vs-human-image-detector
```

**Cuándo usarlo:** para verificar que el modelo carga correctamente y el
registro en MLflow funciona, sin cambiar el modelo configurado.

---

### `make mlflow`
Inicia el servidor web de MLflow UI apuntando al mismo fichero `mlflow.db`.

```bash
make mlflow
```

Internamente ejecuta:
```bash
uv run mlflow ui --host 127.0.0.1 --port 5000 --backend-store-uri sqlite:///mlflow.db
```

Acceder en el navegador a: **http://127.0.0.1:5000**

**Cuándo usarlo:** para visualizar los experimentos y las ejecuciones registradas.

---

## Cómo comprobar que MLflow funciona correctamente

### Paso 1 — Ejecutar el health-check

```bash
make link_model
```

Si todo va bien verás en la consola:
```
✅ OK: reportado a MLflow
```

Si no tienes `HF_MODEL_ID` en `.env`, usa `make link_model` (que lo fija
automáticamente) en lugar de `make healthcheck`.

### Paso 2 — Arrancar la UI de MLflow

En otra terminal (o después de que el health-check termine):

```bash
make mlflow
```

### Paso 3 — Verificar en el navegador

1. Abre **http://127.0.0.1:5000**
2. Selecciona el experimento **ImageAivsReal-Service-Health** en el panel izquierdo.
3. Deberías ver al menos una ejecución llamada **startup-model-load**.
4. Al hacer clic en esa ejecución, en la pestaña **Tags** verás:

| Tag | Valor esperado |
|-----|----------------|
| `service.inference_loaded` | `true` |
| `service.model_source` | `hf` |
| `service.model_id_or_uri` | `Ateeqq/ai-vs-human-image-detector` |
| `service.device` | `cpu` |

---

## Solución de problemas habituales

### Error: `hf_model_id está vacío`

```
ModelLoadError: hf_model_id está vacío. Define HF_MODEL_ID...
```

**Causa:** `HF_MODEL_ID` no está definida en `.env` ni en el entorno.

**Solución:** usa `make link_model` (fija el modelo automáticamente) o añade
la variable a `.env`:
```
HF_MODEL_ID=Ateeqq/ai-vs-human-image-detector
```

---

### La UI de MLflow no muestra ejecuciones

**Causa más probable:** el health-check y la UI están apuntando a ubicaciones
distintas (`./mlruns` vs. `sqlite:///mlflow.db`).

**Solución:** asegúrate de que usas las versiones actualizadas de los comandos.
El `make mlflow` ya incluye `--backend-store-uri sqlite:///mlflow.db` y el
health-check lee `MLFLOW_TRACKING_URI` (por defecto `sqlite:///mlflow.db`),
por lo que ambos apuntan al mismo fichero.

Si el problema persiste, borra el directorio `mlruns/` (si existe) y vuelve
a ejecutar:
```bash
make link_model
make mlflow
```

---

### Error de red al descargar el modelo

```
OSError: We couldn't connect to 'https://huggingface.co'...
```

**Causa:** sin conexión a internet o el nombre del modelo no existe en HuggingFace.

**Solución:**
- Verifica la conexión a internet.
- Comprueba que `HF_MODEL_ID` es correcto: `Ateeqq/ai-vs-human-image-detector`.
- Si el modelo es privado, define `HF_TOKEN` en `.env`.
