"""Componente 1: Ingest Data - Descarga el dataset PCB desde Hugging Face."""

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

    # ✅ USAR hf_hub_download EN LUGAR DE load_dataset
    from huggingface_hub import hf_hub_download, list_repo_files
    import zipfile
    import os

    repo_id = "keremberke/pcb-defect-segmentation"
    
    # Descargar el archivo principal del dataset
    print(f"[ingest_data] Listando archivos del repositorio {repo_id}...")
    files = list_repo_files(repo_id=repo_id, repo_type="dataset")
    print(f"[ingest_data] Archivos encontrados: {files}")

    # Buscar archivos comprimidos o de imágenes
    zip_files = [f for f in files if f.endswith('.zip')]
    
    if zip_files:
        # Si hay un zip, descargarlo
        zip_filename = zip_files[0]
        print(f"[ingest_data] Descargando {zip_filename}...")
        zip_path = hf_hub_download(
            repo_id=repo_id,
            filename=zip_filename,
            repo_type="dataset",
            cache_dir=str(output_dir / ".cache"),
        )
        
        print(f"[ingest_data] Extrayendo {zip_path}...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(str(output_dir))
        print(f"[ingest_data] Dataset extraído en {output_dir}")
    else:
        # Alternativa: usar load_dataset CON trust_remote_code
        print("[ingest_data] Usando load_dataset con trust_remote_code=True...")
        from datasets import load_dataset
        
        dataset = load_dataset(
            "keremberke/pcb-defect-segmentation",
            name="full",
            trust_remote_code=True,  # ← CRÍTICO
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
                    label_filename = f"{split_name}_{idx:05d}.txt"
                    width = image.width
                    height = image.height
                    lines: list[str] = []
                    bboxes = objects.get("bbox", [])
                    categories = objects.get("category", [])
                    for bbox, cat in zip(bboxes, categories):
                        x_center = (bbox[0] + bbox[2] / 2) / width
                        y_center = (bbox[1] + bbox[3] / 2) / height
                        norm_w = bbox[2] / width
                        norm_h = bbox[3] / height
                        lines.append(f"{cat} {x_center:.6f} {y_center:.6f} {norm_w:.6f} {norm_h:.6f}")
                    (labels_dir / label_filename).write_text("\n".join(lines), encoding="utf-8")

                count += 1

            splits_written[split_name] = count
            print(f"[ingest_data] Split '{split_name}': {count} imágenes guardadas en {split_dir}")

        # Guardar metadatos
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