# Resumen de tiempos de inferencia CPU

- Carpeta evaluada: `C:\Users\ruben\ImageAivsReal\data\benchmark_cpu`
- Requests totales: **10**
- Requests OK: **10**
- Requests con error: **0**

## Promedios globales

- Preprocesamiento: **28 ms** (min 8 / max 45)
- Inferencia: **625.1 ms** (min 576 / max 655)
- Total reportado por servidor: **653.5 ms** (min 585 / max 696)
- Round trip cliente→servidor→cliente: **657.657 ms** (min 588.615 / max 699.647)

## Detalle por imagen

| Imagen | Tamaño (bytes) | Resolución | Avg preprocess (ms) | Avg inference (ms) | Avg total servidor (ms) | Avg round trip (ms) |
|---|---:|---|---:|---:|---:|---:|
| IA1.jpg | 317226 | 1067x1690 | 42 | 647 | 689 | 694.293 |
| IA2.jpg | 156484 | 1800x1200 | 40 | 655 | 696 | 699.647 |
| IA3.jpg | 51777 | 750x450 | 9 | 630 | 640 | 645.788 |
| IA4.jpg | 76450 | 750x362 | 8 | 576 | 585 | 588.615 |
| IA5.jpg | 38720 | 512x512 | 10 | 606 | 616 | 619.433 |
| Polar.jpg | 61146 | 1280x720 | 26 | 591 | 617 | 620.727 |
| Reall1.jpg | 63030 | 720x1280 | 30 | 635 | 666 | 669.957 |
| Reall2.jpg | 118620 | 720x1280 | 35 | 621 | 656 | 659.303 |
| Reall3.jpg | 103316 | 720x1280 | 35 | 647 | 682 | 686.696 |
| Reall4.jpg | 231534 | 720x1280 | 45 | 643 | 688 | 692.106 |


Se ejecutó una medición básica de desempeño en CPU enviando un conjunto pequeño de imágenes al servidor gRPC del sistema. Para cada request se registraron el tamaño del archivo, la resolución de la imagen, el tiempo de preprocesamiento, el tiempo de inferencia y el tiempo total reportado por el servidor. Adicionalmente, se midió el tiempo end-to-end desde el cliente hasta recibir la respuesta. Con estos datos se calcularon promedios, mínimos y máximos como referencia de desempeño del prototipo.