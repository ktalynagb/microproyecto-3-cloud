"""Componente 1: Ingest Data - Descarga el dataset PCB desde Roboflow.

Usa Roboflow como fuente primaria (requiere ROBOFLOW_API_KEY) y descarga el
dataset `diplom-qz7q6/defects-2q87r` versión 8 en formato YOLOv8, que incluye
imágenes y labels listos para entrenamiento.

Clases del dataset (4 clases, índices 0-3):
  0: Dry_joint, 1: Incorrect_installation, 2: PCB_damage, 3: Short_circuit

Fallback: si ROBOFLOW_API_KEY no está definida, intenta descargar desde
Hugging Face (keremberke/pcb-defect-segmentation).
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

# Clases correctas del dataset Roboflow (4 clases, índices 0-3)
CLASSES = {
    "0": "Dry_joint",
    "1": "Incorrect_installation",
    "2": "PCB_damage",
    "3": "Short_circuit",
}
NC = 4

# Configuración del dataset Roboflow
ROBOFLOW_WORKSPACE = "diplom-qz7q6"
ROBOFLOW_PROJECT = "defects-2q87r"
ROBOFLOW_VERSION = 8
VALID_IMAGE_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}


def check_dependencies() -> None:
    """Valida que las librerías necesarias estén instaladas antes de ejecutar."""
    required = {
        "roboflow": "Descargar dataset desde Roboflow",
        "huggingface_hub": "Fallback: descargar desde Hugging Face",
    }
    missing = []
    for lib, reason in required.items():
        try:
            __import__(lib)
        except ImportError:
            missing.append(f"  - {lib}: {reason}")
    if missing:
        # No lanzar error si roboflow falta pero HF puede funcionar como fallback
        # Solo informar
        print(
            "[ingest_data] ⚠️ Algunas dependencias opcionales no están instaladas:\n"
            + "\n".join(missing)
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Descarga el dataset PCB desde Roboflow (o Hugging Face como fallback)."
    )
    parser.add_argument(
        "--output_data",
        type=str,
        required=True,
        help="Ruta de salida donde se guardarán las imágenes y anotaciones.",
    )
    return parser.parse_args()


def _validate_yolo_label(label_path: Path) -> bool:
    """Verifica que un archivo .txt tiene formato YOLO válido (mínimo 5 valores por línea)."""
    try:
        lines = label_path.read_text(encoding="utf-8").strip().splitlines()
        for line in lines:
            parts = line.strip().split()
            if len(parts) < 5:
                return False
            int(parts[0])  # class_id must be an integer
        return True
    except Exception:
        return False


def _find_dataset_root(output_dir: Path) -> Path:
    """Encuentra el directorio raíz del dataset (puede ser un subdirectorio de output_dir)."""
    if (output_dir / "train" / "images").exists():
        return output_dir
    for sub in sorted(output_dir.iterdir()):
        if sub.is_dir() and sub.name != ".cache" and (sub / "train" / "images").exists():
            return sub
    return output_dir


def _count_and_validate_splits(data_root: Path) -> dict[str, int]:
    """Cuenta imágenes y valida que existan labels YOLO en cada split."""
    splits_written: dict[str, int] = {}
    for split in ("train", "valid", "test"):
        img_dir = data_root / split / "images"
        lbl_dir = data_root / split / "labels"
        if not img_dir.exists():
            continue

        images = [
            f for f in img_dir.iterdir()
            if f.is_file() and f.suffix.lower() in VALID_IMAGE_EXT
        ]
        labels = list(lbl_dir.glob("*.txt")) if lbl_dir.exists() else []

        # Validar correspondencia imagen-label
        valid_labels = [lbl for lbl in labels if _validate_yolo_label(lbl)]
        splits_written[split] = len(images)
        print(
            f"[ingest_data] Split '{split}': {len(images)} imágenes, "
            f"{len(valid_labels)} labels YOLO válidos en {img_dir}"
        )
        if len(images) != len(labels):
            print(
                f"[ingest_data] ⚠️ Discrepancia en split '{split}': "
                f"{len(images)} imágenes ≠ {len(labels)} labels"
            )
        elif len(valid_labels) < len(labels):
            print(
                f"[ingest_data] ⚠️ {len(labels) - len(valid_labels)} labels inválidos "
                f"en split '{split}'"
            )

    return splits_written


def _ingest_via_roboflow(output_dir: Path, api_key: str) -> dict[str, int]:
    """Descarga el dataset PCB desde Roboflow en formato YOLOv8.

    Descarga workspace/project/version definido por las constantes del módulo.
    Retorna dict con {split_name: num_imágenes}.
    """
    from roboflow import Roboflow

    print(
        f"[ingest_data] Conectando a Roboflow: "
        f"{ROBOFLOW_WORKSPACE}/{ROBOFLOW_PROJECT} v{ROBOFLOW_VERSION}"
    )
    rf = Roboflow(api_key=api_key)
    project = rf.workspace(ROBOFLOW_WORKSPACE).project(ROBOFLOW_PROJECT)
    version = project.version(ROBOFLOW_VERSION)

    print(f"[ingest_data] Descargando dataset en formato YOLOv8...")
    dataset = version.download("yolov8", location=str(output_dir), overwrite=True)
    actual_dir = Path(dataset.location)
    print(f"[ingest_data] Dataset descargado en: {actual_dir}")

    # Si Roboflow creó un subdirectorio, mover los contenidos al output_dir raíz
    if actual_dir.resolve() != output_dir.resolve() and actual_dir.parent.resolve() == output_dir.resolve():
        print(f"[ingest_data] Moviendo datos de {actual_dir} → {output_dir}")
        for item in actual_dir.iterdir():
            dest = output_dir / item.name
            if not dest.exists():
                shutil.move(str(item), str(dest))
        try:
            actual_dir.rmdir()
        except OSError:
            pass  # ignorar si no está vacío

    data_root = _find_dataset_root(output_dir)
    return _count_and_validate_splits(data_root)



def _polygon_to_yolo_seg(segmentation: list, width: int, height: int) -> str | None:
    """Convierte polígono COCO [x1,y1,x2,y2,...] a coordenadas normalizadas YOLO.

    Devuelve una cadena con los puntos normalizados o None si hay menos de 3 puntos.
    """
    if not segmentation or len(segmentation) < 6:
        return None
    coords: list[str] = []
    for i in range(0, len(segmentation) - 1, 2):
        nx = segmentation[i] / width
        ny = segmentation[i + 1] / height
        coords.append(f"{nx:.6f} {ny:.6f}")
    return " ".join(coords)


def _bbox_to_yolo_det(bbox: list, width: int, height: int) -> str:
    """Convierte bbox COCO [x, y, w, h] a formato YOLO detección normalizado."""
    x_center = (bbox[0] + bbox[2] / 2) / width
    y_center = (bbox[1] + bbox[3] / 2) / height
    norm_w = bbox[2] / width
    norm_h = bbox[3] / height
    return f"{x_center:.6f} {y_center:.6f} {norm_w:.6f} {norm_h:.6f}"


def _write_labels(
    objects: dict,
    label_path: Path,
    width: int,
    height: int,
) -> None:
    """Escribe el archivo de labels YOLO para un ejemplo del dataset.

    Prioriza formato segmentación (polígonos) sobre detección (bbox).
    """
    bboxes = objects.get("bbox", [])
    categories = objects.get("category", [])
    segmentations = objects.get("segmentation", [])

    lines: list[str] = []
    for i, (cat, bbox) in enumerate(zip(categories, bboxes)):
        seg = segmentations[i] if i < len(segmentations) else None

        if seg and isinstance(seg, list) and len(seg) >= 6:
            # YOLO segmentación: class x1 y1 x2 y2 ...
            poly = _polygon_to_yolo_seg(seg, width, height)
            if poly:
                lines.append(f"{cat} {poly}")
                continue

        # Fallback a YOLO detección: class x_c y_c w h
        lines.append(f"{cat} {_bbox_to_yolo_det(bbox, width, height)}")

    if lines:
        label_path.write_text("\n".join(lines), encoding="utf-8")


def _ingest_via_load_dataset(output_dir: Path) -> dict[str, int]:
    """Descarga usando HuggingFace datasets, preservando los splits originales."""
    from datasets import load_dataset

    print("[ingest_data] Usando load_dataset con trust_remote_code=True...")
    dataset = load_dataset(
        "keremberke/pcb-defect-segmentation",
        name="full",
        trust_remote_code=True,
    )

    splits_written: dict[str, int] = {}
    for split_name, split_data in dataset.items():
        split_dir = output_dir / split_name
        images_dir = split_dir / "images"
        labels_dir = split_dir / "labels"
        images_dir.mkdir(parents=True, exist_ok=True)
        labels_dir.mkdir(parents=True, exist_ok=True)

        count = 0
        for idx, example in enumerate(split_data):
            image = example.get("image")
            if image is None:
                continue

            img_filename = f"{split_name}_{idx:05d}.jpg"
            image.save(str(images_dir / img_filename))

            objects = example.get("objects", {})
            if objects:
                label_path = labels_dir / f"{split_name}_{idx:05d}.txt"
                _write_labels(objects, label_path, image.width, image.height)

            count += 1

        splits_written[split_name] = count
        print(f"[ingest_data] Split '{split_name}': {count} imágenes → {split_dir}")

    return splits_written


def _ingest_via_zip(output_dir: Path, zip_files: list[str]) -> bool:
    """Descarga y extrae el archivo zip del dataset.

    Retorna True si se descargó y extrajo correctamente.
    """
    import zipfile

    from huggingface_hub import hf_hub_download

    repo_id = "keremberke/pcb-defect-segmentation"
    zip_filename = zip_files[0]
    print(f"[ingest_data] Descargando {zip_filename}...")
    zip_path = hf_hub_download(
        repo_id=repo_id,
        filename=zip_filename,
        repo_type="dataset",
        cache_dir=str(output_dir / ".cache"),
    )

    print(f"[ingest_data] Extrayendo {zip_path}...")
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(str(output_dir))
    print(f"[ingest_data] Dataset extraído en {output_dir}")
    return True


def main() -> None:
    check_dependencies()
    args = parse_args()
    output_dir = Path(args.output_data)
    output_dir.mkdir(parents=True, exist_ok=True)

    api_key = os.environ.get("ROBOFLOW_API_KEY", "").strip()

    splits_written: dict[str, int] = {}

    if api_key:
        print(f"[ingest_data] ROBOFLOW_API_KEY detectada. Descargando desde Roboflow...")
        splits_written = _ingest_via_roboflow(output_dir, api_key)
        source = f"roboflow:{ROBOFLOW_WORKSPACE}/{ROBOFLOW_PROJECT}/v{ROBOFLOW_VERSION}"
    else:
        print(
            "[ingest_data] ⚠️ ROBOFLOW_API_KEY no definida. "
            "Usando fallback Hugging Face (keremberke/pcb-defect-segmentation)."
        )
        print(
            "[ingest_data] Para mejor calidad de datos, define ROBOFLOW_API_KEY "
            "y usa el dataset de Roboflow (ver README.md sección 8)."
        )
        print("[ingest_data] Descargando dataset: keremberke/pcb-defect-segmentation")

        from huggingface_hub import list_repo_files

        repo_id = "keremberke/pcb-defect-segmentation"
        print(f"[ingest_data] Listando archivos del repositorio {repo_id}...")
        files = list(list_repo_files(repo_id=repo_id, repo_type="dataset"))
        print(f"[ingest_data] Archivos encontrados: {files}")

        zip_files = [f for f in files if f.endswith(".zip")]

        if zip_files:
            # Descargar todos los zips de splits (train, valid, test)
            for zf in zip_files:
                try:
                    _ingest_via_zip(output_dir, [zf])
                except Exception as exc:
                    print(f"[ingest_data] ⚠️ Error descargando {zf}: {exc}")
            # Contar imágenes por split en la estructura extraída
            for split in ("train", "valid", "test"):
                img_dir = output_dir / split / "images"
                if not img_dir.exists():
                    candidates = list(output_dir.rglob(f"{split}/images"))
                    if candidates:
                        img_dir = candidates[0]
                if img_dir.exists():
                    count = len(list(img_dir.glob("*")))
                    splits_written[split] = count
                    print(f"[ingest_data] Split '{split}': {count} imágenes en {img_dir}")
        else:
            splits_written = _ingest_via_load_dataset(output_dir)

        source = repo_id

    # Guardar metadatos con clases correctas (4 clases)
    metadata = {
        "source": source,
        "splits": splits_written,
        "nc": NC,
        "classes": CLASSES,
    }
    (output_dir / "dataset_info.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    total = sum(splits_written.values())
    print(f"[ingest_data] Dataset completo: {total} imágenes | splits={splits_written}")


if __name__ == "__main__":
    main()