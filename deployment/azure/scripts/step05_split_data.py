"""Paso 5: Split Image Directory - Partición 80/20 con semilla fija."""
import argparse
import random
import shutil
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--input_path", required=True)
parser.add_argument("--train_output", required=True)
parser.add_argument("--test_output", required=True)
parser.add_argument("--train_ratio", type=float, default=0.8)
parser.add_argument("--seed", type=int, default=42)
args = parser.parse_args()

VALID_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}
src = Path(args.input_path)
train_dst = Path(args.train_output)
test_dst = Path(args.test_output)
train_dst.mkdir(parents=True, exist_ok=True)
test_dst.mkdir(parents=True, exist_ok=True)

images = [f for f in src.rglob("*") if f.is_file() and f.suffix.lower() in VALID_EXT]
random.seed(args.seed)
random.shuffle(images)
split_idx = int(len(images) * args.train_ratio)
train_imgs = images[:split_idx]
test_imgs = images[split_idx:]

for img in train_imgs:
    shutil.copy2(img, train_dst / img.name)
for img in test_imgs:
    shutil.copy2(img, test_dst / img.name)

# Copy labels/annotations alongside images
for label_dir in [d for d in src.rglob("*") if d.is_dir() and d.name == "labels"]:
    for txt_file in label_dir.rglob("*.txt"):
        stem = txt_file.stem
        if any(i.stem == stem for i in train_imgs):
            shutil.copy2(txt_file, train_dst / txt_file.name)
        elif any(i.stem == stem for i in test_imgs):
            shutil.copy2(txt_file, test_dst / txt_file.name)

print(f"[step05] Split: {len(train_imgs)} train / {len(test_imgs)} test (seed={args.seed})")
