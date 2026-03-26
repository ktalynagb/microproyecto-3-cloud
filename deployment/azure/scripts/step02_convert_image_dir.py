"""Paso 2: Convert to Image Directory - Organiza imágenes en estructura estándar."""
import argparse
import shutil
from pathlib import Path

VALID_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}

parser = argparse.ArgumentParser()
parser.add_argument("--input_path", required=True)
parser.add_argument("--output_path", required=True)
args = parser.parse_args()

src = Path(args.input_path)
dst = Path(args.output_path)
dst.mkdir(parents=True, exist_ok=True)

count = 0
for f in src.rglob("*"):
    if f.is_file() and f.suffix.lower() in VALID_EXT:
        dest_file = dst / f.name
        shutil.copy2(f, dest_file)
        count += 1

print(f"[step02] Image directory created: {count} images in {dst}")
