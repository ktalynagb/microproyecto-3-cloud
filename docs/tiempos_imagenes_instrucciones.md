# Instrucciones para el issue #29

## Qué hace
`script_tiempos_inf.py` mide tiempos de inferencia CPU enviando imágenes al servidor gRPC y genera:

- `mediciones_tiempos_cpu.csv`
- `resumen_tiempos_cpu.json`
- `resumen_tiempos_cpu.md`

## Dónde guardar las imágenes
Carpeta recomendada:

```text
data/benchmark_cpu/
```

Ejemplo:

```text
data/benchmark_cpu/
├── img_01.jpg
├── img_02.png
└── img_03.jpeg
```

## Cómo ejecutarlo

1. Genera stubs si aún no existen:
```bash
make grpc
```

2. Levanta el servidor gRPC:
```bash
make grpc-server
```

3. En otra terminal ejecuta el script:
```bash
uv run python script_tiempos_inf.py --input-dir data/benchmark_cpu --output-dir docs/evidencias/issue_29
```

## Cómo verificar que funciona

- Debe imprimir una línea `[OK]` por cada imagen/repetición.
- Debe crear tres archivos de salida en `docs/evidencias/issue_29/`.
- El CSV debe contener columnas de tamaño, resolución y tiempos.
- El Markdown debe traer un resumen listo para pegar al informe.

## Para que NO entre en pytest
- Déjalo en la raíz del repo o en una carpeta que no sea `tests/`.
- No lo nombres `test_*.py`.
- Así `make test` no lo ejecuta.
