"""Componente 1: Ingest Data - Descarga el dataset PCB desde Hugging Face.

El dataset keremberke/pcb-defect-segmentation tiene 4 clases:
  0: dry_joint, 1: incorrect_installation, 2: pcb_damage, 3: short_circuit

Genera labels en formato YOLO segmentación cuando el dataset incluye polígonos,
o en formato YOLO detección (bbox) como fallback.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

# Clases correctas del dataset (4 clases, índices 0-3)
CLASSES = {
    "0": "dry_joint",
    "1": "incorrect_installation",
    "2": "pcb_damage",
    "3": "short_circuit",
}
NC = 4


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Descarga el dataset PCB desde Hugging Face.")
    parser.add_argument(
        "--output_data",
        type=str,
        required=True,
        help="Ruta de salida donde se guardarán las imágenes y anotaciones.",
    )
    return parser.parse_args()


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
    args = parse_args()
    output_dir = Path(args.output_data)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("[ingest_data] Descargando dataset: keremberke/pcb-defect-segmentation")

    from huggingface_hub import list_repo_files

    repo_id = "keremberke/pcb-defect-segmentation"
    print(f"[ingest_data] Listando archivos del repositorio {repo_id}...")
    files = list(list_repo_files(repo_id=repo_id, repo_type="dataset"))
    print(f"[ingest_data] Archivos encontrados: {files}")

    zip_files = [f for f in files if f.endswith(".zip")]

    splits_written: dict[str, int] = {}
    if zip_files:
        _ingest_via_zip(output_dir, zip_files)
        # Contar imágenes por split en la estructura extraída
        for split in ("train", "valid", "test"):
            img_dir = output_dir / split / "images"
            if not img_dir.exists():
                # Algunos zips extraen con un subdirectorio raíz
                candidates = list(output_dir.rglob(f"{split}/images"))
                if candidates:
                    img_dir = candidates[0]
            if img_dir.exists():
                count = len(list(img_dir.glob("*")))
                splits_written[split] = count
                print(f"[ingest_data] Split '{split}': {count} imágenes en {img_dir}")
    else:
        splits_written = _ingest_via_load_dataset(output_dir)

    # Guardar metadatos con clases correctas (4 clases)
    metadata = {
        "source": repo_id,
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