"""Paso 10: Export Data - Exporta modelo y resultados a Blob Storage."""
import argparse
import shutil
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--model_path", required=True)
parser.add_argument("--metrics_path", required=True)
parser.add_argument("--predictions_path", required=True)
parser.add_argument("--output_path", required=True)
args = parser.parse_args()

dst = Path(args.output_path)
dst.mkdir(parents=True, exist_ok=True)

# Exportar modelo
model_file = Path(args.model_path) / "best.pt"
if model_file.exists():
    shutil.copy2(model_file, dst / "best.pt")
    print(f"[step10] Model exported: {dst / 'best.pt'}")

# Exportar métricas
metrics_file = Path(args.metrics_path) / "metrics.json"
if metrics_file.exists():
    shutil.copy2(metrics_file, dst / "metrics.json")
    print(f"[step10] Metrics exported: {dst / 'metrics.json'}")

# Exportar predicciones e imágenes anotadas
preds_src = Path(args.predictions_path)
for f in preds_src.rglob("*"):
    if f.is_file():
        rel = f.relative_to(preds_src)
        out = dst / "predictions" / rel
        out.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(f, out)

print(f"[step10] All results exported to Blob Storage path: {dst}")
print("[step10] Pipeline complete. Ephemeral storage: no central database used.")
