"""Paso 4: Apply Image Transformation - Resize + normalización sobre el dataset."""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--input_path", required=True)
parser.add_argument("--transform_config", required=True)
parser.add_argument("--output_path", required=True)
args = parser.parse_args()

src = Path(args.input_path)
cfg_file = Path(args.transform_config) / "transform_config.json"
dst = Path(args.output_path)
dst.mkdir(parents=True, exist_ok=True)

with open(cfg_file, encoding="utf-8") as fh:
    cfg = json.load(fh)

size = cfg["image_size"]
VALID_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}
count = 0
for f in src.rglob("*"):
    if f.is_file() and f.suffix.lower() in VALID_EXT:
        img = cv2.imread(str(f))
        if img is None:
            continue
        img_resized = cv2.resize(img, (size, size), interpolation=cv2.INTER_LINEAR)
        out_file = dst / f.name
        cv2.imwrite(str(out_file), img_resized)
        count += 1

# Copy non-image files (annotations, labels) as-is
for f in src.rglob("*"):
    if f.is_file() and f.suffix.lower() not in VALID_EXT:
        rel = f.relative_to(src)
        (dst / rel).parent.mkdir(parents=True, exist_ok=True)
        import shutil
        shutil.copy2(f, dst / rel)

print(f"[step04] Applied transform to {count} images ({size}x{size}) in {dst}")
