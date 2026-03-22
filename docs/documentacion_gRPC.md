## Arquitectura del sistema e integración gRPC

# 1. Arquitectura general del sistema

El sistema está compuesto por varios módulos que trabajan de forma desacoplada para permitir el procesamiento de imágenes y la clasificación entre imágenes generadas por inteligencia artificial y fotografías reales.

Los principales componentes del sistema son:

GUI (Interfaz de Usuario): Permite al usuario cargar imágenes, ejecutar el análisis y visualizar los resultados.

Cliente gRPC: Encargado de enviar las imágenes al servidor de inferencia y recibir las predicciones.

Servidor gRPC: Recibe las solicitudes del cliente, ejecuta el modelo de clasificación y retorna los resultados.

Motor de inferencia: Implementa el pipeline de preprocesamiento y ejecución del modelo de clasificación.

Modelo de clasificación: Modelo basado en Vision Transformer que determina si una imagen es real o generada por IA.

Generación de reportes: Permite exportar los resultados en formato CSV y PDF.

# 2. Diagrama de arquitectura

+----------------+
|     Usuario    |
+--------+-------+
         |
         v
+----------------+
|      GUI       |
+--------+-------+
         |
         v
+----------------+
|  Cliente gRPC  |
+--------+-------+
         |
         v
+----------------+
|  Servidor gRPC |
+--------+-------+
         |
         v
+----------------+
| InferenceEngine|
+--------+-------+
         |
         v
+----------------+
| Modelo AI/Real |
+--------+-------+
         |
         v
+----------------+
| CSV / PDF Gen  |
+----------------+
# 3. Responsabilidades por módulo

GUI

La interfaz gráfica constituye el punto de interacción entre el usuario y el sistema. A través de ella el usuario puede seleccionar una o varias imágenes desde su dispositivo, iniciar el proceso de análisis y visualizar los resultados obtenidos por el modelo. Además, la interfaz gestiona el estado del procesamiento de cada imagen (por ejemplo pending, éxito o error) y permite la exportación de los resultados en formatos de reporte como CSV y PDF.

Cliente gRPC

El cliente gRPC funciona como intermediario entre la interfaz gráfica y el servidor de inferencia. Su función principal es enviar las imágenes seleccionadas por el usuario al servidor para su procesamiento y recibir posteriormente las predicciones generadas. También se encarga de gestionar la comunicación con el servidor y manejar posibles errores de conexión o respuesta, devolviendo finalmente los resultados a la interfaz gráfica para su visualización.

Servidor gRPC

El servidor gRPC es el componente responsable de recibir las solicitudes de inferencia provenientes del cliente. Una vez recibidas las imágenes, el servidor coordina la ejecución del pipeline de procesamiento llamando al motor de inferencia correspondiente. Tras completar el análisis, el servidor construye y envía la respuesta con los resultados de clasificación al cliente gRPC, gestionando además posibles errores durante el procesamiento.

Motor de inferencia

El motor de inferencia implementa el pipeline central de procesamiento de imágenes. Este módulo se encarga de validar los archivos recibidos, realizar el preprocesamiento necesario para adaptar las imágenes al formato requerido por el modelo y ejecutar la inferencia. Además, calcula las probabilidades asociadas a cada clase y registra los tiempos de procesamiento para fines de análisis de desempeño.

Modelo de clasificación

El modelo de clasificación corresponde al componente encargado de realizar la predicción final sobre cada imagen procesada. En este sistema se utiliza un modelo basado en Vision Transformer, entrenado para distinguir entre imágenes reales y generadas por inteligencia artificial. A partir de la imagen preprocesada, el modelo produce probabilidades para cada clase y permite determinar la etiqueta final asociada a la imagen analizada.

# 4. Flujo de datos de una inferencia

El flujo de procesamiento de una imagen sigue los siguientes pasos:
1. El usuario selecciona una o varias imágenes desde la interfaz gráfica.
2. La GUI envía las imágenes al cliente gRPC.
3. El cliente gRPC envía la solicitud al servidor gRPC.
4. El servidor gRPC recibe la imagen y la pasa al motor de inferencia.
5. El motor de inferencia realiza el preprocesamiento de la imagen.
6. El modelo ejecuta la inferencia y genera las probabilidades de clasificación.
7. Se construye una respuesta con la etiqueta predicha, probabilidades y tiempos de procesamiento.
8. El servidor gRPC envía la respuesta al cliente.
9. El cliente gRPC entrega los resultados a la GUI.
10. La GUI muestra los resultados al usuario y permite exportarlos en CSV o PDF.

# 5. Consideraciones de escalabilidad

La arquitectura basada en gRPC permite separar claramente la interfaz de usuario del motor de inferencia, lo que facilita la evolución y escalabilidad del sistema. Gracias a esta separación, es posible desplegar el servidor de inferencia en un servidor dedicado sin modificar la interfaz gráfica, habilitar el procesamiento paralelo de múltiples solicitudes y permitir la integración del sistema con otros clientes o servicios en el futuro.