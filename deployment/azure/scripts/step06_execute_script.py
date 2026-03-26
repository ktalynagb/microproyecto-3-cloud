"""Paso 6: Execute Python Script - Descarga YOLOv8n y genera config de fine-tuning."""
import argparse
import json
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--train_data", required=True)
parser.add_argument("--output_path", required=True)
parser.add_argument("--hf_model_id", default="keremberke/yolov8n-pcb-defect-segmentation")
parser.add_argument("--epochs", type=int, default=50)
parser.add_argument("--imgsz", type=int, default=640)
parser.add_argument("--batch", type=int, default=16)
parser.add_argument("--lr0", type=float, default=0.01)
args = parser.parse_args()

dst = Path(args.output_path)
dst.mkdir(parents=True, exist_ok=True)

# Descargar modelo base desde Hugging Face
from huggingface_hub import hf_hub_download
model_path = hf_hub_download(repo_id=args.hf_model_id, filename="best.pt")
import shutil
shutil.copy2(model_path, dst / "base_model.pt")
print(f"[step06] Base model downloaded from HF: {args.hf_model_id}")

# Generar archivo de configuración YAML para fine-tuning
train_path = Path(args.train_data)
dataset_yaml = dst / "dataset.yaml"
dataset_yaml.write_text(
    f"path: {train_path}\n"
    "train: .\n"
    "val: .\n"
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

# Guardar hiperparámetros
hyperparams = {
    "model": str(dst / "base_model.pt"),
    "data": str(dataset_yaml),
    "epochs": args.epochs,
    "imgsz": args.imgsz,
    "batch": args.batch,
    "lr0": args.lr0,
    "task": "segment",
    "hf_model_id": args.hf_model_id,
}
(dst / "hyperparams.json").write_text(json.dumps(hyperparams, indent=2), encoding="utf-8")
print(f"[step06] Fine-tuning config saved: epochs={args.epochs}, imgsz={args.imgsz}x{args.imgsz}")
