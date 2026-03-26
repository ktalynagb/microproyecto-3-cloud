"""Componente 4: Evaluate Model - Inferencia, cálculo de métricas y exportación.

Carga el modelo entrenado, genera predicciones sobre el conjunto de prueba,
calcula métricas de evaluación (mAP, precisión, recall) y exporta el modelo
junto con todos los resultados al directorio de salida.

Uso:
    python evaluate_model.py \
        --input_data <ruta_test> \
        --model_data <ruta_modelo_entrenado> \
        --output_data <ruta_salida_resultados> \
        [--conf_threshold 0.25]
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


VALID_IMAGE_EXT = {".jpg", ".jpeg", ".png", ".bmp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluación del modelo YOLOv8n: inferencia, métricas y exportación."
    )
    parser.add_argument(
        "--input_data",
        type=str,
        required=True,
        help="Ruta de entrada con el conjunto de prueba (salida de preprocess_split).",
    )
    parser.add_argument(
        "--model_data",
        type=str,
        required=True,
        help="Ruta de entrada con el modelo entrenado (salida de train_yolo).",
    )
    parser.add_argument(
        "--output_data",
        type=str,
        required=True,
        help="Ruta de salida donde se exportarán métricas, predicciones y el modelo.",
    )
    parser.add_argument(
        "--conf_threshold",
        type=float,
        default=0.25,
        help="Umbral de confianza para las detecciones (por defecto: 0.25).",
    )
    return parser.parse_args()


def _find_dataset_yaml(model_dir: Path, test_dir: Path, output_dir: Path) -> Path | None:
    """Localiza o genera un dataset.yaml para la evaluación oficial con YOLO.

    Busca en model_dir; si no existe o sus rutas son inválidas (p.ej. paths
    absolutos del contenedor de entrenamiento), genera uno nuevo apuntando
    al test set actual.
    """
    yaml_path = model_dir / "dataset.yaml"
    if yaml_path.exists():
        # Validar que los paths del YAML sean accesibles desde este contenedor
        try:
            content = yaml_path.read_text(encoding="utf-8")
            # Buscar la línea 'val: ...' y verificar que la ruta exista
            import re
            val_match = re.search(r"^val:\s*(.+)$", content, re.MULTILINE)
            if val_match:
                val_path_str = val_match.group(1).strip()
                val_path = Path(val_path_str)
                if val_path.is_absolute() and not val_path.exists():
                    print(
                        f"[evaluate_model] dataset.yaml encontrado pero path inválido "
                        f"(val={val_path_str}). Generando nuevo YAML para evaluación."
                    )
                    yaml_path = None  # Forzar regeneración
        except Exception:
            yaml_path = None  # En caso de error, regenerar

    if yaml_path is None or not yaml_path.exists():
        yaml_path = None  # Asegurar que sea None para regenerar

    if yaml_path is None:
        # Leer resumen de entrenamiento para obtener clases y nc
        summary_path = model_dir / "training_summary.json"
        if summary_path.exists():
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            nc = summary.get("nc", 4)
            classes = summary.get("classes", ["Dry_joint", "Incorrect_installation", "PCB_damage", "Short_circuit"])
        else:
            nc = 4
            classes = ["Dry_joint", "Incorrect_installation", "PCB_damage", "Short_circuit"]

        names_block = "\n".join(f"  {i}: {name}" for i, name in enumerate(classes))
        images_dir = test_dir / "images" if (test_dir / "images").exists() else test_dir
        content = (
            f"path: {test_dir}\n"
            f"train: images\n"
            f"val: {images_dir}\n"
            f"nc: {nc}\n"
            f"names:\n{names_block}\n"
        )
        yaml_path = output_dir / "eval_dataset.yaml"
        yaml_path.write_text(content, encoding="utf-8")
        print(f"[evaluate_model] dataset.yaml generado para evaluación: {yaml_path}")

    return yaml_path


def main() -> None:
    args = parse_args()
    test_dir = Path(args.input_data)
    model_dir = Path(args.model_data)
    output_dir = Path(args.output_data)

    annotated_dir = output_dir / "annotated_images"
    output_dir.mkdir(parents=True, exist_ok=True)
    annotated_dir.mkdir(parents=True, exist_ok=True)

    model_file = model_dir / "best.pt"
    if not model_file.exists():
        raise FileNotFoundError(f"[evaluate_model] Modelo no encontrado: {model_file}")

    import cv2
    import mlflow
    from ultralytics import YOLO

    model = YOLO(str(model_file))

    # Inferencia sobre el conjunto de prueba
    images_dir = test_dir / "images" if (test_dir / "images").exists() else test_dir
    images = [
        f for f in images_dir.rglob("*")
        if f.is_file() and f.suffix.lower() in VALID_IMAGE_EXT
    ]

    predictions: list[dict] = []
    for img_path in images:
        results = model(str(img_path), conf=args.conf_threshold)
        annotated = results[0].plot()
        cv2.imwrite(str(annotated_dir / img_path.name), annotated)

        boxes = []
        has_defects = False
        if results[0].boxes is not None and len(results[0].boxes) > 0:
            has_defects = True
            for box in results[0].boxes:
                boxes.append(
                    {
                        "class": model.names[int(box.cls.item())],
                        "confidence": round(float(box.conf.item()), 4),
                        "bbox": box.xyxy.tolist(),
                    }
                )

        predictions.append(
            {
                "filename": img_path.name,
                "has_defects": has_defects,
                "detections": boxes,
            }
        )

    (output_dir / "predictions.json").write_text(
        json.dumps(predictions, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"[evaluate_model] Inferencia: {len(images)} imágenes procesadas.")

    # Calcular métricas de resumen
    total = len(predictions)
    with_defects = sum(1 for p in predictions if p["has_defects"])
    all_detections = [d for p in predictions for d in p["detections"]]
    avg_conf = (
        sum(d["confidence"] for d in all_detections) / len(all_detections)
        if all_detections
        else 0.0
    )

    # Ejecutar validación oficial YOLO para mAP
    map50 = 0.0
    map50_95 = 0.0
    try:
        yaml_path = _find_dataset_yaml(model_dir, test_dir, output_dir)
        if yaml_path:
            val_results = model.val(data=str(yaml_path), conf=args.conf_threshold)
            map50 = float(val_results.box.map50)
            map50_95 = float(val_results.box.map)
            print(f"[evaluate_model] mAP@0.5={map50:.4f} | mAP@0.5:0.95={map50_95:.4f}")
    except Exception as exc:
        print(f"[evaluate_model] No se pudo calcular mAP oficial: {exc}")

    metrics = {
        "total_images_evaluated": total,
        "images_with_defects": with_defects,
        "images_without_defects": total - with_defects,
        "total_detections": len(all_detections),
        "avg_detection_confidence": round(avg_conf, 4),
        "map50": round(map50, 4),
        "map50_95": round(map50_95, 4),
    }

    # Registrar métricas con MLflow
    mlflow.start_run()
    for k, v in metrics.items():
        mlflow.log_metric(k, v)

    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"[evaluate_model] Métricas: {metrics}")

    # Exportar modelo al directorio de resultados
    if model_file.exists():
        shutil.copy2(model_file, output_dir / "best.pt")
        print(f"[evaluate_model] Modelo exportado: {output_dir / 'best.pt'}")

    mlflow.end_run()
    print(f"[evaluate_model] Resultados completos en: {output_dir}")


if __name__ == "__main__":
    main()
