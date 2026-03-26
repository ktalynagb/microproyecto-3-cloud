"""Paso 8: Score Image Model - Genera predicciones sobre el set de prueba."""
import argparse
import json
from pathlib import Path

import cv2

parser = argparse.ArgumentParser()
parser.add_argument("--test_data", required=True)
parser.add_argument("--model_path", required=True)
parser.add_argument("--output_path", required=True)
parser.add_argument("--conf_threshold", type=float, default=0.25)
args = parser.parse_args()

VALID_EXT = {".jpg", ".jpeg", ".png", ".bmp"}
src = Path(args.test_data)
model_file = Path(args.model_path) / "best.pt"
dst = Path(args.output_path)
dst.mkdir(parents=True, exist_ok=True)

from ultralytics import YOLO
model = YOLO(str(model_file))

predictions = []
images = [f for f in src.rglob("*") if f.is_file() and f.suffix.lower() in VALID_EXT]
for img_path in images:
    results = model(str(img_path), conf=args.conf_threshold)
    annotated = results[0].plot()
    out_img = dst / img_path.name
    cv2.imwrite(str(out_img), annotated)

    boxes = []
    has_defects = False
    if results[0].boxes is not None and len(results[0].boxes) > 0:
        has_defects = True
        for box in results[0].boxes:
            boxes.append({
                "class": model.names[int(box.cls.item())],
                "confidence": round(float(box.conf.item()), 4),
                "bbox": box.xyxy.tolist(),
            })

    predictions.append({
        "filename": img_path.name,
        "has_defects": has_defects,
        "detections": boxes,
        "message": "PCB sin defectos" if not has_defects else "",
    })

(dst / "predictions.json").write_text(
    json.dumps(predictions, indent=2, ensure_ascii=False), encoding="utf-8"
)
print(f"[step08] Scored {len(images)} images. Predictions saved to {dst}/predictions.json")
