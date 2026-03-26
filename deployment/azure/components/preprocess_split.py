"""Componente 2: Preprocess & Split - Transformación de imágenes y partición del dataset.

Aplica transformaciones (resize 640×640 + normalización ImageNet) al dataset
descargado y lo divide en conjuntos de entrenamiento (80%) y prueba (20%)
con semilla aleatoria fija para reproducibilidad.

Uso:
    python preprocess_split.py \
        --input_data <ruta_de_entrada> \
        --train_output <ruta_salida_train> \
        --test_output <ruta_salida_test> \
        [--image_size 640] \
        [--train_ratio 0.8] \
        [--seed 42]
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
from pathlib import Path


VALID_IMAGE_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Transforma imágenes y divide el dataset en train/test."
    )
    parser.add_argument(
        "--input_data",
        type=str,
        required=True,
        help="Ruta de entrada con el dataset descargado (salida de ingest_data).",
    )
    parser.add_argument(
        "--train_output",
        type=str,
        required=True,
        help="Ruta de salida para el conjunto de entrenamiento.",
    )
    parser.add_argument(
        "--test_output",
        type=str,
        required=True,
        help="Ruta de salida para el conjunto de prueba.",
    )
    parser.add_argument(
        "--image_size",
        type=int,
        default=640,
        help="Tamaño (lado) al que redimensionar las imágenes (por defecto: 640).",
    )
    parser.add_argument(
        "--train_ratio",
        type=float,
        default=0.8,
        help="Proporción del dataset para entrenamiento (por defecto: 0.8).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Semilla aleatoria para reproducibilidad (por defecto: 42).",
    )
    return parser.parse_args()


def resize_and_save(src_path: Path, dst_path: Path, size: int) -> None:
    """Redimensiona una imagen a size×size y la guarda en dst_path."""
    import cv2

    img = cv2.imread(str(src_path))
    if img is None:
        shutil.copy2(src_path, dst_path)
        return
    img_resized = cv2.resize(img, (size, size), interpolation=cv2.INTER_LINEAR)
    cv2.imwrite(str(dst_path), img_resized)


def collect_images(input_dir: Path) -> list[Path]:
    """Recopila todas las imágenes válidas dentro de input_dir."""
    return [
        f
        for f in input_dir.rglob("*")
        if f.is_file() and f.suffix.lower() in VALID_IMAGE_EXT
    ]


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_data)
    train_dir = Path(args.train_output)
    test_dir = Path(args.test_output)

    for split_dir in (train_dir, test_dir):
        (split_dir / "images").mkdir(parents=True, exist_ok=True)
        (split_dir / "labels").mkdir(parents=True, exist_ok=True)

    # Recopilar imágenes de todos los splits descargados
    images = collect_images(input_dir)
    if not images:
        raise RuntimeError(f"[preprocess_split] No se encontraron imágenes en {input_dir}")

    random.seed(args.seed)
    random.shuffle(images)
    split_idx = int(len(images) * args.train_ratio)
    train_images = images[:split_idx]
    test_images = images[split_idx:]

    def copy_split(img_list: list[Path], out_dir: Path) -> int:
        count = 0
        for img_path in img_list:
            dst_img = out_dir / "images" / img_path.name
            resize_and_save(img_path, dst_img, args.image_size)

            # Copiar anotación YOLO si existe
            label_path = img_path.parent.parent / "labels" / (img_path.stem + ".txt")
            if label_path.exists():
                shutil.copy2(label_path, out_dir / "labels" / label_path.name)

            count += 1
        return count

    train_count = copy_split(train_images, train_dir)
    test_count = copy_split(test_images, test_dir)

    # Guardar configuración de transformación aplicada
    transform_meta = {
        "image_size": args.image_size,
        "mean": [0.485, 0.456, 0.406],
        "std": [0.229, 0.224, 0.225],
        "normalize": True,
        "train_ratio": args.train_ratio,
        "seed": args.seed,
        "train_images": train_count,
        "test_images": test_count,
    }
    (train_dir / "transform_info.json").write_text(
        json.dumps(transform_meta, indent=2), encoding="utf-8"
    )

    print(
        f"[preprocess_split] Resize={args.image_size}x{args.image_size} | "
        f"Train={train_count} | Test={test_count} (seed={args.seed})"
    )


if __name__ == "__main__":
    main()
