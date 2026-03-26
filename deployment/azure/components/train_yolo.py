"""Componente 3: Train YOLOv8n - Fine-tuning del modelo sobre el dataset PCB.

Descarga el modelo base YOLOv8n desde Hugging Face, genera el archivo de
configuración YAML del dataset y ejecuta el fine-tuning registrando métricas
con MLflow.

El dataset keremberke/pcb-defect-segmentation tiene 4 clases:
  0: dry_joint, 1: incorrect_installation, 2: pcb_damage, 3: short_circuit

Se detecta automáticamente si los labels están en formato segmentación (polígonos)
o detección (bbox) para configurar la tarea correctamente.

Uso:
    python train_yolo.py \
        --input_data <ruta_train> \
        --output_data <ruta_salida_modelo> \
        [--hf_model_id keremberke/yolov8n-pcb-defect-segmentation] \
        [--epochs 50] \
        [--imgsz 640] \
        [--batch 16] \
        [--lr0 0.01] \
        [--task auto|detect|segment]
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
from pathlib import Path

# Clases correctas del dataset (4 clases, índices 0-3)
CLASSES = ["dry_joint", "incorrect_installation", "pcb_damage", "short_circuit"]
NC = 4


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fine-tuning de YOLOv8n sobre el dataset de defectos PCB."
    )
    parser.add_argument(
        "--input_data",
        type=str,
        required=True,
        help="Ruta de entrada con el conjunto de entrenamiento (salida de preprocess_split).",
    )
    parser.add_argument(
        "--output_data",
        type=str,
        required=True,
        help="Ruta de salida donde se guardará el modelo entrenado (best.pt).",
    )
    parser.add_argument(
        "--hf_model_id",
        type=str,
        default="keremberke/yolov8n-pcb-defect-segmentation",
        help="ID del modelo en Hugging Face Hub.",
    )
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--lr0", type=float, default=0.01)
    parser.add_argument(
        "--task",
        type=str,
        default="auto",
        choices=["auto", "detect", "segment"],
        help=(
            "Tarea YOLO: 'segment' para segmentación (polígonos), "
            "'detect' para detección (bbox), "
            "'auto' para detección automática desde los labels (por defecto)."
        ),
    )
    parser.add_argument(
        "--val_ratio",
        type=float,
        default=0.1,
        help="Fracción de los datos de entrenamiento usada como validación (por defecto: 0.1).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Semilla aleatoria para la partición train/val interna.",
    )
    return parser.parse_args()


def _detect_task_from_labels(labels_dir: Path) -> str:
    """Determina la tarea YOLO inspeccionando los archivos de label.

    - Si algún label tiene más de 5 columnas → segmentación (polígono).
    - Si todos tienen 5 columnas → detección (bbox).
    - Si no hay labels → detección (por defecto seguro).
    """
    label_files = list(labels_dir.glob("*.txt"))
    if not label_files:
        print("[train_yolo] No se encontraron labels; se usará task='detect' por defecto.")
        return "detect"

    for lf in label_files[:20]:  # Inspeccionar hasta 20 archivos
        try:
            lines = lf.read_text(encoding="utf-8").strip().splitlines()
            for line in lines:
                values = line.strip().split()
                if len(values) > 5:
                    print(
                        f"[train_yolo] Labels en formato segmentación detectados "
                        f"({len(values)} valores en {lf.name}). task='segment'."
                    )
                    return "segment"
        except OSError:
            continue

    print("[train_yolo] Labels en formato detección (5 valores). task='detect'.")
    return "detect"


def _create_val_split(
    train_dir: Path,
    val_dir: Path,
    val_ratio: float,
    seed: int,
) -> None:
    """Crea un subconjunto de validación copiando archivos desde train_dir."""
    val_dir.mkdir(parents=True, exist_ok=True)
    (val_dir / "images").mkdir(parents=True, exist_ok=True)
    (val_dir / "labels").mkdir(parents=True, exist_ok=True)

    images = list((train_dir / "images").glob("*"))
    if not images:
        return

    random.seed(seed)
    random.shuffle(images)
    n_val = max(1, int(len(images) * val_ratio))
    val_images = images[:n_val]

    for img_path in val_images:
        shutil.copy2(img_path, val_dir / "images" / img_path.name)
        label_path = train_dir / "labels" / f"{img_path.stem}.txt"
        if label_path.exists():
            shutil.copy2(label_path, val_dir / "labels" / label_path.name)

    print(f"[train_yolo] Split validación interno: {n_val}/{len(images)} imágenes.")


def _build_dataset_yaml(
    output_dir: Path,
    train_dir: Path,
    val_dir: Path,
    task: str,
) -> Path:
    """Genera el dataset.yaml para YOLO con rutas absolutas y clases correctas."""
    # Nombre de la clave depende del task (por compatibilidad con todas las versiones)
    names_block = "\n".join(f"  {i}: {name}" for i, name in enumerate(CLASSES))
    content = (
        f"path: {train_dir}\n"
        f"train: images\n"
        f"val: {val_dir / 'images'}\n"
        f"nc: {NC}\n"
        f"names:\n{names_block}\n"
    )
    if task == "segment":
        content += "task: segment\n"

    dataset_yaml = output_dir / "dataset.yaml"
    dataset_yaml.write_text(content, encoding="utf-8")
    print(f"[train_yolo] dataset.yaml generado:\n{content}")
    return dataset_yaml


def main() -> None:
    args = parse_args()
    train_dir = Path(args.input_data)
    output_dir = Path(args.output_data)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Detectar tarea ─────────────────────────────────────────────────────
    labels_dir = train_dir / "labels"
    if args.task == "auto":
        task = _detect_task_from_labels(labels_dir)
    else:
        task = args.task
    print(f"[train_yolo] Tarea YOLO: {task}")

    # ── Descargar modelo base desde Hugging Face ───────────────────────────
    print(f"[train_yolo] Descargando modelo base: {args.hf_model_id}")
    from huggingface_hub import hf_hub_download

    model_path = hf_hub_download(repo_id=args.hf_model_id, filename="best.pt")
    base_model_dst = output_dir / "base_model.pt"
    shutil.copy2(model_path, base_model_dst)
    print(f"[train_yolo] Modelo base guardado en: {base_model_dst}")

    # ── Crear split de validación interno ────────────────────────────────
    val_dir = output_dir / "val_split"
    _create_val_split(train_dir, val_dir, args.val_ratio, args.seed)

    # ── Generar dataset.yaml ───────────────────────────────────────────────
    dataset_yaml = _build_dataset_yaml(output_dir, train_dir, val_dir, task)

    # ── Entrenamiento ──────────────────────────────────────────────────────
    import mlflow
    from ultralytics import YOLO, settings

    settings.update({"mlflow": False})

    mlflow.start_run()
    mlflow.log_params(
        {
            "hf_model_id": args.hf_model_id,
            "epochs": args.epochs,
            "imgsz": args.imgsz,
            "batch": args.batch,
            "lr0": args.lr0,
            "task": task,
            "nc": NC,
        }
    )

    model = YOLO(str(base_model_dst))
    results = model.train(
        data=str(dataset_yaml),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        lr0=args.lr0,
        task=task,
        project=str(output_dir),
        name="yolov8n_pcb_finetune",
        exist_ok=True,
    )
    print("[train_yolo] Entrenamiento completado.")

    # ── Registrar métricas con MLflow ─────────────────────────────────────
    if hasattr(results, "results_dict"):
        for k, v in results.results_dict.items():
            try:
                mlflow.log_metric(k, float(v))
                print(f"[train_yolo] Métrica: {k}={v}")
            except (TypeError, ValueError):
                pass

    # ── Copiar best.pt al directorio raíz ─────────────────────────────────
    best_pt = output_dir / "yolov8n_pcb_finetune" / "weights" / "best.pt"
    if best_pt.exists():
        shutil.copy2(best_pt, output_dir / "best.pt")
        print(f"[train_yolo] best.pt → {output_dir / 'best.pt'}")
    else:
        print(f"[train_yolo] Advertencia: best.pt no encontrado en {best_pt}")

    # ── Guardar resumen ────────────────────────────────────────────────────
    summary = {
        "hf_model_id": args.hf_model_id,
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "lr0": args.lr0,
        "task": task,
        "nc": NC,
        "classes": CLASSES,
        "model_path": str(output_dir / "best.pt"),
        "dataset_yaml": str(dataset_yaml),
    }
    (output_dir / "training_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    mlflow.end_run()
    print(f"[train_yolo] Modelo y resumen exportados a: {output_dir}")


if __name__ == "__main__":
    main()