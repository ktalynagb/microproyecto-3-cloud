"""Paso 7: Train PyTorch Model - Fine-tuning YOLOv8n con MLflow logging."""
import argparse
import json
from pathlib import Path

import mlflow

parser = argparse.ArgumentParser()
parser.add_argument("--train_data", required=True)
parser.add_argument("--model_config", required=True)
parser.add_argument("--output_path", required=True)
args = parser.parse_args()

cfg_file = Path(args.model_config) / "hyperparams.json"
with open(cfg_file, encoding="utf-8") as fh:
    hp = json.load(fh)

dst = Path(args.output_path)
dst.mkdir(parents=True, exist_ok=True)

mlflow.start_run()
mlflow.log_params({k: v for k, v in hp.items() if k != "model"})

from ultralytics import YOLO
model = YOLO(hp["model"])
results = model.train(
    data=hp["data"],
    epochs=hp["epochs"],
    imgsz=hp["imgsz"],
    batch=hp["batch"],
    lr0=hp["lr0"],
    task=hp.get("task", "segment"),
    project=str(dst),
    name="yolov8n_pcb_finetune",
    exist_ok=True,
)

# Registrar métricas finales
if hasattr(results, "results_dict"):
    for k, v in results.results_dict.items():
        try:
            mlflow.log_metric(k, float(v))
        except (TypeError, ValueError):
            pass

# Copiar best.pt al output
import shutil
best_pt = Path(str(dst)) / "yolov8n_pcb_finetune" / "weights" / "best.pt"
if best_pt.exists():
    shutil.copy2(best_pt, dst / "best.pt")
    print(f"[step07] Training complete. best.pt saved to {dst}")
else:
    print(f"[step07] Warning: best.pt not found at expected path {best_pt}")

mlflow.end_run()
