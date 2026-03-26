"""Componente 3: Train YOLOv8n - Fine-tuning del modelo sobre el dataset PCB.

Descarga el modelo base YOLOv8n desde Hugging Face, genera el archivo de
configuración YAML del dataset y ejecuta el fine-tuning registrando métricas
con MLflow.

Uso:
    python train_yolo.py \
        --input_data <ruta_train> \
        --output_data <ruta_salida_modelo> \
        [--hf_model_id keremberke/yolov8n-pcb-defect-segmentation] \
        [--epochs 50] \
        [--imgsz 640] \
        [--batch 16] \
        [--lr0 0.01]
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


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
        help="ID del modelo en Hugging Face Hub (por defecto: keremberke/yolov8n-pcb-defect-segmentation).",
    )
    parser.add_argument("--epochs", type=int, default=50, help="Número de épocas de entrenamiento.")
    parser.add_argument("--imgsz", type=int, default=640, help="Tamaño de imagen de entrada.")
    parser.add_argument("--batch", type=int, default=16, help="Tamaño del batch.")
    parser.add_argument("--lr0", type=float, default=0.01, help="Tasa de aprendizaje inicial.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_dir = Path(args.input_data)
    output_dir = Path(args.output_data)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Descargar modelo base desde Hugging Face
    print(f"[train_yolo] Descargando modelo base desde Hugging Face: {args.hf_model_id}")
    from huggingface_hub import hf_hub_download

    model_path = hf_hub_download(repo_id=args.hf_model_id, filename="best.pt")
    base_model_dst = output_dir / "base_model.pt"
    shutil.copy2(model_path, base_model_dst)
    print(f"[train_yolo] Modelo base guardado en: {base_model_dst}")

    # Generar dataset.yaml para YOLO
    dataset_yaml = output_dir / "dataset.yaml"
    dataset_yaml.write_text(
        f"path: {train_dir}\n"
        "train: images\n"
        "val: images\n"
        "nc: 6\n"
        "names:\n"
        "  0: Dry_joint\n"
        "  1: Incorrect_installation\n"
        "  2: PCB_damage\n"
        "  3: Short_circuit\n"
        "  4: Mousebites\n"
        "  5: Opens\n",
        encoding="utf-8",
    )
    print(f"[train_yolo] dataset.yaml generado en: {dataset_yaml}")

    # Registrar hiperparámetros con MLflow
    import mlflow
    from ultralytics import YOLO

    mlflow.start_run()
    mlflow.log_params(
        {
            "hf_model_id": args.hf_model_id,
            "epochs": args.epochs,
            "imgsz": args.imgsz,
            "batch": args.batch,
            "lr0": args.lr0,
            "task": "segment",
        }
    )

    # ✅ DESACTIVAR MLflow callback de Ultralytics para evitar conflicto con Azure ML
    import os
    os.environ["YOLO_VERBOSE"] = "False"

    model = YOLO(str(base_model_dst))
    
    # ✅ Entrenar SIN parámetros inválidos
    results = model.train(
        data=str(dataset_yaml),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        lr0=args.lr0,
        task="segment",
        project=str(output_dir),
        name="yolov8n_pcb_finetune",
        exist_ok=True,
    )

    print(f"[train_yolo] Entrenamiento completado")

    # Registrar métricas finales con MLflow MANUALMENTE
    if hasattr(results, "results_dict"):
        for k, v in results.results_dict.items():
            try:
                mlflow.log_metric(k, float(v))
                print(f"[train_yolo] Métrica registrada: {k}={v}")
            except (TypeError, ValueError):
                pass

    # Copiar best.pt al directorio raíz de salida
    best_pt = output_dir / "yolov8n_pcb_finetune" / "weights" / "best.pt"
    if best_pt.exists():
        shutil.copy2(best_pt, output_dir / "best.pt")
        print(f"[train_yolo] Entrenamiento completo. best.pt guardado en: {output_dir / 'best.pt'}")
    else:
        print(f"[train_yolo] Advertencia: best.pt no encontrado en {best_pt}")

    # Guardar resumen de entrenamiento
    summary = {
        "hf_model_id": args.hf_model_id,
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "lr0": args.lr0,
        "model_path": str(output_dir / "best.pt"),
    }
    (output_dir / "training_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    mlflow.end_run()
    print(f"[train_yolo] Modelo y resumen exportados a: {output_dir}")


if __name__ == "__main__":
    main()