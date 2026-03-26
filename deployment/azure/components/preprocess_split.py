"""Componente 2: Preprocess & Split - Transformación de imágenes y partición del dataset.

Aplica transformaciones (resize 640×640) al dataset descargado y lo divide en
conjuntos de entrenamiento y prueba, preservando los splits originales de
Hugging Face cuando están disponibles (train/valid/test).

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


def find_label_for_image(img_path: Path) -> Path | None:
    """Busca el archivo de label YOLO correspondiente a una imagen.

    Prueba varias rutas relativas habituales en datasets YOLO.
    """
    stem = img_path.stem
    # Candidatos en orden de preferencia
    candidates = [
        # Estructura estándar YOLO: images/ → labels/
        img_path.parent.parent / "labels" / f"{stem}.txt",
        # Misma carpeta que la imagen
        img_path.parent / f"{stem}.txt",
        # Raíz del split
        img_path.parent.parent.parent / "labels" / f"{stem}.txt",
    ]
    for path in candidates:
        if path.exists():
            return path
    return None


def collect_images(input_dir: Path) -> list[Path]:
    """Recopila todas las imágenes válidas dentro de input_dir."""
    return [
        f
        for f in input_dir.rglob("*")
        if f.is_file() and f.suffix.lower() in VALID_IMAGE_EXT
    ]


def _detect_hf_splits(input_dir: Path) -> tuple[list[Path], list[Path]] | None:
    """Detecta splits originales train/valid de Hugging Face.

    Retorna (train_images, val_images) si la estructura se reconoce, o None.
    """
    train_img_dir = input_dir / "train" / "images"
    valid_img_dir = input_dir / "valid" / "images"

    # Buscar dentro de posibles subdirectorios raíz
    if not train_img_dir.exists():
        subdirs = [d for d in input_dir.iterdir() if d.is_dir() and d.name != ".cache"]
        for sub in subdirs:
            if (sub / "train" / "images").exists():
                train_img_dir = sub / "train" / "images"
                valid_img_dir = sub / "valid" / "images"
                break

    if not train_img_dir.exists():
        return None

    train_imgs = collect_images(train_img_dir)
    val_imgs = collect_images(valid_img_dir) if valid_img_dir.exists() else []
    if not train_imgs:
        return None

    print(
        f"[preprocess_split] Splits HF detectados: train={len(train_imgs)}, valid={len(val_imgs)}"
    )
    return train_imgs, val_imgs


def copy_split(img_list: list[Path], out_dir: Path, image_size: int) -> int:
    """Copia y redimensiona imágenes y sus labels al directorio de salida."""
    seen_names: set[str] = set()
    count = 0
    for img_path in img_list:
        # Evitar colisiones de nombres entre splits
        name = img_path.name
        if name in seen_names:
            name = f"{img_path.parent.parent.name}_{name}"
        seen_names.add(name)
        stem = Path(name).stem

        dst_img = out_dir / "images" / name
        resize_and_save(img_path, dst_img, image_size)

        label_path = find_label_for_image(img_path)
        if label_path is not None:
            shutil.copy2(label_path, out_dir / "labels" / f"{stem}.txt")

        count += 1
    return count


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_data)
    train_dir = Path(args.train_output)
    test_dir = Path(args.test_output)

    for split_dir in (train_dir, test_dir):
        (split_dir / "images").mkdir(parents=True, exist_ok=True)
        (split_dir / "labels").mkdir(parents=True, exist_ok=True)

    # Intentar usar los splits originales de Hugging Face
    hf_splits = _detect_hf_splits(input_dir)
    if hf_splits is not None:
        train_images, val_images = hf_splits
        # Usar valid como test si existe, de lo contrario cortar del train
        if val_images:
            test_images = val_images
        else:
            random.seed(args.seed)
            random.shuffle(train_images)
            split_idx = int(len(train_images) * args.train_ratio)
            test_images = train_images[split_idx:]
            train_images = train_images[:split_idx]
    else:
        # Fallback: re-split todas las imágenes
        all_images = collect_images(input_dir)
        if not all_images:
            raise RuntimeError(
                f"[preprocess_split] No se encontraron imágenes en {input_dir}"
            )
        random.seed(args.seed)
        random.shuffle(all_images)
        split_idx = int(len(all_images) * args.train_ratio)
        train_images = all_images[:split_idx]
        test_images = all_images[split_idx:]

    train_count = copy_split(train_images, train_dir, args.image_size)
    test_count = copy_split(test_images, test_dir, args.image_size)

    # Guardar configuración de transformación aplicada
    transform_meta = {
        "image_size": args.image_size,
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
