"""Validador de integridad del dataset descargado.

Verifica que el dataset tenga la estructura YOLO correcta con imágenes y labels
correspondientes en los splits requeridos.

Uso:
    python deployment/azure/validate_dataset.py --dataset_dir <ruta_dataset>
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

VALID_IMAGE_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}
REQUIRED_SPLITS = ["train", "valid"]  # test es opcional


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Valida la integridad de un dataset YOLO (imágenes + labels)."
    )
    parser.add_argument(
        "--dataset_dir",
        type=str,
        required=True,
        help="Ruta al directorio raíz del dataset (con subdirectorios train/, valid/, test/).",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        default=False,
        help="Modo estricto: falla si imágenes ≠ labels en cualquier split.",
    )
    return parser.parse_args()


def _find_dataset_root(base_dir: Path) -> Path:
    """Detecta el directorio raíz del dataset (puede estar en un subdirectorio)."""
    if (base_dir / "train" / "images").exists():
        return base_dir
    for sub in sorted(base_dir.iterdir()):
        if sub.is_dir() and sub.name != ".cache" and (sub / "train" / "images").exists():
            print(f"[validate_dataset] Dataset encontrado en subdirectorio: {sub}")
            return sub
    return base_dir


def _validate_yolo_label(label_path: Path) -> bool:
    """Verifica que un archivo .txt tiene formato YOLO válido."""
    try:
        lines = label_path.read_text(encoding="utf-8").strip().splitlines()
        if not lines:
            return True  # Imagen de fondo (sin defectos) es válida
        for line in lines:
            parts = line.strip().split()
            if len(parts) < 5:
                return False
            int(parts[0])  # class_id must be an integer
            for val in parts[1:]:
                f = float(val)
                if f < 0.0 or f > 1.0:
                    return False  # coordenadas deben estar en [0, 1]
        return True
    except Exception:
        return False


def validate_split(split_dir: Path, split_name: str, strict: bool) -> dict:
    """Valida un split individual del dataset."""
    result = {
        "split": split_name,
        "exists": False,
        "images": 0,
        "labels": 0,
        "valid_labels": 0,
        "missing_labels": [],
        "invalid_labels": [],
        "errors": [],
    }

    img_dir = split_dir / "images"
    lbl_dir = split_dir / "labels"

    if not split_dir.exists():
        result["errors"].append(f"Directorio no encontrado: {split_dir}")
        return result

    if not img_dir.exists():
        result["errors"].append(f"Directorio de imágenes no encontrado: {img_dir}")
        return result

    result["exists"] = True

    # Contar imágenes
    images = [
        f for f in img_dir.iterdir()
        if f.is_file() and f.suffix.lower() in VALID_IMAGE_EXT
    ]
    result["images"] = len(images)

    if len(images) == 0:
        result["errors"].append(f"❌ No se encontraron imágenes en {img_dir}")
        return result

    # Verificar labels
    if not lbl_dir.exists():
        result["errors"].append(f"❌ Directorio de labels no encontrado: {lbl_dir}")
        return result

    labels = list(lbl_dir.glob("*.txt"))
    result["labels"] = len(labels)

    # Verificar correspondencia imagen ↔ label
    label_stems = {lbl.stem for lbl in labels}
    for img in images:
        if img.stem not in label_stems:
            result["missing_labels"].append(img.name)

    # Validar formato YOLO en labels existentes
    valid_count = 0
    for lbl in labels:
        if _validate_yolo_label(lbl):
            valid_count += 1
        else:
            result["invalid_labels"].append(lbl.name)
    result["valid_labels"] = valid_count

    return result


def validate_dataset(dataset_dir: Path, strict: bool = False) -> bool:
    """Valida toda la estructura del dataset.

    Retorna True si el dataset es válido para entrenamiento.
    """
    data_root = _find_dataset_root(dataset_dir)
    print(f"\n{'='*60}")
    print("VALIDACIÓN: Integridad del Dataset YOLO")
    print(f"{'='*60}")
    print(f"Directorio raíz: {data_root}\n")

    all_valid = True
    total_images = 0
    total_labels = 0

    for split_name in ("train", "valid", "test"):
        split_dir = data_root / split_name
        result = validate_split(split_dir, split_name, strict)

        is_required = split_name in REQUIRED_SPLITS
        prefix = "✅" if result["exists"] and not result["errors"] else ("❌" if is_required else "⚠️")

        print(f"{prefix} Split '{split_name}':")

        if not result["exists"]:
            if is_required:
                print(f"   ❌ No encontrado (REQUERIDO)")
                all_valid = False
            else:
                print(f"   (no encontrado, opcional)")
            continue

        print(f"   Imágenes: {result['images']}")
        print(f"   Labels:   {result['labels']} ({result['valid_labels']} válidos YOLO)")

        if result["errors"]:
            for err in result["errors"]:
                print(f"   {err}")
            if is_required:
                all_valid = False

        if result["missing_labels"]:
            n = len(result["missing_labels"])
            print(f"   ⚠️ {n} imágenes sin label correspondiente")
            if strict and is_required:
                for f in result["missing_labels"][:5]:
                    print(f"     - {f}")
                all_valid = False

        if result["invalid_labels"]:
            n = len(result["invalid_labels"])
            print(f"   ⚠️ {n} labels con formato YOLO inválido")
            if strict:
                for f in result["invalid_labels"][:5]:
                    print(f"     - {f}")
                all_valid = False

        total_images += result["images"]
        total_labels += result["labels"]

    print(f"\nTotal: {total_images} imágenes, {total_labels} labels")

    # Verificar data.yaml
    data_yaml = data_root / "data.yaml"
    if data_yaml.exists():
        print(f"✅ data.yaml encontrado: {data_yaml}")
    else:
        print("⚠️ data.yaml no encontrado (se generará durante el entrenamiento)")

    if all_valid and total_images > 0:
        print("\n✅ Dataset válido para entrenamiento.")
    else:
        print("\n❌ Dataset inválido o incompleto.")

    return all_valid and total_images > 0


def main() -> None:
    args = parse_args()
    dataset_dir = Path(args.dataset_dir)

    if not dataset_dir.exists():
        print(f"❌ Directorio no encontrado: {dataset_dir}")
        sys.exit(1)

    success = validate_dataset(dataset_dir, strict=args.strict)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
