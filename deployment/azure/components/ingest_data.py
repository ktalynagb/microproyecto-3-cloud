"""Componente 1: Ingest Data - Descarga el dataset PCB desde Hugging Face.

Descarga el dataset keremberke/pcb-defect-segmentation usando la librería
datasets de Hugging Face y lo guarda en la ruta de salida montada por
Azure ML para que el siguiente paso del pipeline pueda consumirlo.

Uso:
    python ingest_data.py --output_data <ruta_de_salida>
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Descarga el dataset PCB desde Hugging Face.")
    parser.add_argument(
        "--output_data",
        type=str,
        required=True,
        help="Ruta de salida donde se guardarán las imágenes y anotaciones.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_data)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("[ingest_data] Descargando dataset desde Hugging Face: keremberke/pcb-defect-segmentation")

    from datasets import load_dataset

    dataset = load_dataset("keremberke/pcb-defect-segmentation", name="full")

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

            # Guardar anotaciones en formato YOLO si están disponibles
            objects = example.get("objects", {})
            if objects:
                label_filename = f"{split_name}_{idx:05d}.txt"
                width = image.width
                height = image.height
                lines: list[str] = []
                bboxes = objects.get("bbox", [])
                categories = objects.get("category", [])
                for bbox, cat in zip(bboxes, categories):
                    # Convertir de formato COCO [x, y, w, h] a YOLO normalizado
                    x_center = (bbox[0] + bbox[2] / 2) / width
                    y_center = (bbox[1] + bbox[3] / 2) / height
                    norm_w = bbox[2] / width
                    norm_h = bbox[3] / height
                    lines.append(f"{cat} {x_center:.6f} {y_center:.6f} {norm_w:.6f} {norm_h:.6f}")
                (labels_dir / label_filename).write_text("\n".join(lines), encoding="utf-8")

            count += 1

        splits_written[split_name] = count
        print(f"[ingest_data] Split '{split_name}': {count} imágenes guardadas en {split_dir}")

    # Guardar metadatos del dataset
    metadata = {
        "source": "keremberke/pcb-defect-segmentation",
        "splits": splits_written,
        "classes": {
            "0": "Dry_joint",
            "1": "Incorrect_installation",
            "2": "PCB_damage",
            "3": "Short_circuit",
            "4": "Mousebites",
            "5": "Opens",
        },
    }
    (output_dir / "dataset_info.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    total = sum(splits_written.values())
    print(f"[ingest_data] Dataset completo: {total} imágenes en {output_dir}")


if __name__ == "__main__":
    main()
