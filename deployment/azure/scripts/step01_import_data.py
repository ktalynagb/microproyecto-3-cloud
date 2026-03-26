"""Paso 1: Import Data - Copia imágenes desde Blob Storage al output."""
import argparse
import shutil
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--input_path", required=True)
parser.add_argument("--output_path", required=True)
args = parser.parse_args()

src = Path(args.input_path)
dst = Path(args.output_path)
dst.mkdir(parents=True, exist_ok=True)

for f in src.rglob("*"):
    if f.is_file():
        rel = f.relative_to(src)
        (dst / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(f, dst / rel)

print(f"[step01] Imported {len(list(dst.rglob('*')))} files to {dst}")
