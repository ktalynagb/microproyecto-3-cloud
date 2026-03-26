"""Paso 3: Init Image Transformation - Genera configuración de resize y normalización."""
import argparse
import json
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--input_path", required=True)
parser.add_argument("--output_path", required=True)
parser.add_argument("--image_size", type=int, default=640)
parser.add_argument("--mean", default="0.485,0.456,0.406")
parser.add_argument("--std", default="0.229,0.224,0.225")
args = parser.parse_args()

dst = Path(args.output_path)
dst.mkdir(parents=True, exist_ok=True)

config = {
    "image_size": args.image_size,
    "mean": [float(v) for v in args.mean.split(",")],
    "std": [float(v) for v in args.std.split(",")],
    "normalize": True,
}
(dst / "transform_config.json").write_text(
    json.dumps(config, indent=2), encoding="utf-8"
)
print(f"[step03] Transform config saved: resize={args.image_size}x{args.image_size}, ImageNet norm")
