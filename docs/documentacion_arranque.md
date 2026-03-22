# Instalación, arranque y ejecución local de la aplicación

## Requisitos previos

Antes de ejecutar el sistema se requiere contar con:

1.  Python 3.10 o superior\
2.  Conexión a internet (para la primera descarga del modelo)\
3.  Git (opcional, para clonar el repositorio)

El sistema utiliza librerías de aprendizaje automático y procesamiento
de imágenes como PyTorch, Hugging Face Transformers y la
interfaz web está desarrollada con Streamlit.

Se recomienda utilizar uv para la gestión de dependencias, ya que
permite instalar y ejecutar el proyecto de manera más rápida y
reproducible.


## 1. Clonar el repositorio

Primero se debe obtener el código del proyecto desde el repositorio.

``` bash
git clone https://github.com/Deivs117/ImageAivsReal.git
cd ImageAivsReal
```

## 2. Crear y activar el entorno virtual (opcional)

Se recomienda crear un entorno virtual para aislar las dependencias del
proyecto.

``` bash
python -m venv venv
```

### Activar el entorno virtual

Windows

``` bash
venv\Scripts\activate
```

Linux / Mac

``` bash
source venv/bin/activate
```


## 3. Instalación de dependencias

### Método recomendado (usando uv)

Instalar las dependencias del proyecto:

``` bash
make install
```

Este comando ejecuta internamente:

``` bash
uv sync
```

lo que instala automáticamente todas las dependencias definidas en el
proyecto.

También es posible ejecutar comandos del proyecto mediante:

``` bash
uv run <comando>
```

por ejemplo:

``` bash
uv run python script.py
```

### Método alternativo (pip)

Si no se utiliza uv, las dependencias pueden instalarse manualmente
con:

``` bash
pip install -r requirements.txt
```


## 4. Generar stubs de gRPC

Se utiliza gRPC para la comunicación entre la interfaz y el servidor
de inferencia.

Para generar los stubs a partir del archivo `.proto`:

``` bash
make grpc
```

o

``` bash
make proto-gen
```

Esto generará los archivos necesarios en:

    proto/generated/


## 5. Ejecutar el servidor de inferencia

El servidor de inferencia debe iniciarse antes de ejecutar la aplicación
web.\
Este servidor es el encargado de cargar el modelo y procesar las
solicitudes enviadas desde la interfaz.

Desde la raíz del proyecto ejecutar:

``` bash
make inference
```

Este comando inicia el servidor de inferencia que procesa las
solicitudes del cliente.

También es posible iniciar directamente el servidor gRPC con:

``` bash
make grpc-server
```


## 6. Ejecutar la aplicación Streamlit

En una segunda terminal, ejecutar la interfaz gráfica.

### Con make

``` bash
make gui
```

### Manualmente

``` bash
venv\Scripts\activate
streamlit run app/streamlit_app.py
```

La aplicación se abrirá automáticamente en el navegador, normalmente en:

    http://localhost:8501

Desde esta interfaz el usuario puede:

1.  cargar imágenes
2.  ejecutar el análisis
3.  visualizar los resultados generados por el modelo


## 7. Primera descarga del modelo

Durante la primera ejecución del sistema, el modelo será descargado
automáticamente desde Hugging Face.

Este proceso puede tardar algunos minutos dependiendo de la velocidad de
conexión. Una vez descargado, el modelo quedará almacenado en caché
local y no será necesario descargarlo nuevamente en ejecuciones
posteriores.


## 8. Otros comandos útiles del proyecto

El proyecto incluye un Makefile con varios comandos útiles para
desarrollo y pruebas.

``` bash
make help
```

Muestra todos los comandos disponibles.

### Algunos comandos importantes

    make test                → ejecutar todos los tests
    make test-inference      → ejecutar tests del motor de inferencia
    make test-preprocessing  → ejecutar tests de preprocesamiento
    make test-coverage       → generar reporte de cobertura
    make mlflow              → iniciar servidor de MLflow
    make clean               → limpiar archivos temporales
