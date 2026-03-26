"""Tests unitarios para el pipeline de batch inference y componentes de entrenamiento."""

from __future__ import annotations

import json
import sys
import textwrap
from io import BytesIO
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# Añadir el directorio de componentes al sys.path
_COMPONENTS_DIR = Path(__file__).resolve().parent.parent / "components"
sys.path.insert(0, str(_COMPONENTS_DIR))


# ── Tests: ingest_data ────────────────────────────────────────────────────

class TestIngestData:
    def test_polygon_to_yolo_seg_valid(self):
        from ingest_data import _polygon_to_yolo_seg

        seg = [0, 0, 100, 0, 100, 100, 0, 100]
        result = _polygon_to_yolo_seg(seg, width=200, height=200)
        assert result is not None
        parts = result.split()
        # 4 puntos × 2 = 8 valores
        assert len(parts) == 8

    def test_polygon_to_yolo_seg_too_short(self):
        from ingest_data import _polygon_to_yolo_seg

        # Menos de 3 puntos (< 6 valores) → None
        result = _polygon_to_yolo_seg([0, 0, 10, 10], width=100, height=100)
        assert result is None

    def test_polygon_to_yolo_seg_normalization(self):
        from ingest_data import _polygon_to_yolo_seg

        seg = [100, 200, 300, 400, 500, 600]
        result = _polygon_to_yolo_seg(seg, width=1000, height=1000)
        vals = [float(v) for v in result.split()]
        assert all(0.0 <= v <= 1.0 for v in vals)

    def test_bbox_to_yolo_det(self):
        from ingest_data import _bbox_to_yolo_det

        # bbox COCO: [x, y, w, h] = [10, 20, 40, 60] en imagen 100×100
        result = _bbox_to_yolo_det([10, 20, 40, 60], width=100, height=100)
        parts = result.split()
        assert len(parts) == 4
        x_c, y_c, w, h = (float(p) for p in parts)
        # x_center = (10 + 40/2) / 100 = 0.30
        assert abs(x_c - 0.30) < 1e-5
        # y_center = (20 + 60/2) / 100 = 0.50
        assert abs(y_c - 0.50) < 1e-5
        assert abs(w - 0.40) < 1e-5
        assert abs(h - 0.60) < 1e-5

    def test_write_labels_with_segmentation(self, tmp_path):
        from ingest_data import _write_labels

        objects = {
            "bbox": [[10, 20, 40, 60]],
            "category": [0],
            "segmentation": [[10, 20, 50, 20, 50, 80, 10, 80]],
        }
        label_path = tmp_path / "test.txt"
        _write_labels(objects, label_path, width=100, height=100)
        content = label_path.read_text(encoding="utf-8")
        line = content.strip().splitlines()[0]
        parts = line.split()
        # class_id + 8 polygon coords = 9 values
        assert parts[0] == "0"
        assert len(parts) == 9

    def test_write_labels_fallback_to_bbox(self, tmp_path):
        from ingest_data import _write_labels

        objects = {
            "bbox": [[10, 20, 40, 60]],
            "category": [2],
            "segmentation": [],  # Vacío → fallback a bbox
        }
        label_path = tmp_path / "test.txt"
        _write_labels(objects, label_path, width=100, height=100)
        content = label_path.read_text(encoding="utf-8")
        line = content.strip().splitlines()[0]
        parts = line.split()
        # class_id + 4 bbox coords = 5 values
        assert parts[0] == "2"
        assert len(parts) == 5

    def test_classes_count(self):
        from ingest_data import CLASSES, NC

        assert NC == 4
        assert len(CLASSES) == 4
        assert "dry_joint" in CLASSES.values()
        assert "short_circuit" in CLASSES.values()


# ── Tests: preprocess_split ───────────────────────────────────────────────

class TestPreprocessSplit:
    def test_find_label_for_image_standard_structure(self, tmp_path):
        from preprocess_split import find_label_for_image

        # Crear estructura estándar YOLO: images/ + labels/
        (tmp_path / "images").mkdir()
        (tmp_path / "labels").mkdir()
        img_path = tmp_path / "images" / "test.jpg"
        img_path.touch()
        label_path = tmp_path / "labels" / "test.txt"
        label_path.write_text("0 0.5 0.5 0.3 0.3")

        found = find_label_for_image(img_path)
        assert found == label_path

    def test_find_label_for_image_not_found(self, tmp_path):
        from preprocess_split import find_label_for_image

        (tmp_path / "images").mkdir()
        img_path = tmp_path / "images" / "no_label.jpg"
        img_path.touch()

        found = find_label_for_image(img_path)
        assert found is None

    def test_collect_images_filters_valid_ext(self, tmp_path):
        from preprocess_split import collect_images

        (tmp_path / "a.jpg").touch()
        (tmp_path / "b.PNG").touch()
        (tmp_path / "c.txt").touch()
        (tmp_path / "d.xml").touch()

        imgs = collect_images(tmp_path)
        names = {p.name for p in imgs}
        assert "a.jpg" in names
        assert "b.PNG" in names
        assert "c.txt" not in names
        assert "d.xml" not in names

    def test_detect_hf_splits(self, tmp_path):
        from preprocess_split import _detect_hf_splits

        # Crear estructura HuggingFace
        for split in ("train", "valid"):
            (tmp_path / split / "images").mkdir(parents=True)
            # Crear imagen de prueba (1×1 px negro)
            import cv2
            img = np.zeros((1, 1, 3), dtype=np.uint8)
            cv2.imwrite(str(tmp_path / split / "images" / f"{split}_001.jpg"), img)

        result = _detect_hf_splits(tmp_path)
        assert result is not None
        train_imgs, val_imgs = result
        assert len(train_imgs) == 1
        assert len(val_imgs) == 1


# ── Tests: train_yolo ─────────────────────────────────────────────────────

class TestTrainYolo:
    def test_detect_task_from_labels_segment(self, tmp_path):
        from train_yolo import _detect_task_from_labels

        labels_dir = tmp_path / "labels"
        labels_dir.mkdir()
        # Label de segmentación: 9 valores (1 + 8 coords)
        (labels_dir / "seg.txt").write_text("0 0.1 0.2 0.3 0.2 0.3 0.4 0.1 0.4")

        task = _detect_task_from_labels(labels_dir)
        assert task == "segment"

    def test_detect_task_from_labels_detect(self, tmp_path):
        from train_yolo import _detect_task_from_labels

        labels_dir = tmp_path / "labels"
        labels_dir.mkdir()
        # Label de detección: 5 valores
        (labels_dir / "det.txt").write_text("0 0.5 0.5 0.3 0.3")

        task = _detect_task_from_labels(labels_dir)
        assert task == "detect"

    def test_detect_task_no_labels(self, tmp_path):
        from train_yolo import _detect_task_from_labels

        labels_dir = tmp_path / "empty_labels"
        labels_dir.mkdir()

        task = _detect_task_from_labels(labels_dir)
        assert task == "detect"

    def test_build_dataset_yaml_content(self, tmp_path):
        from train_yolo import _build_dataset_yaml

        train_dir = tmp_path / "train"
        val_dir = tmp_path / "val"
        train_dir.mkdir()
        val_dir.mkdir()
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        yaml_path = _build_dataset_yaml(output_dir, train_dir, val_dir, "detect")
        content = yaml_path.read_text(encoding="utf-8")

        assert "nc: 4" in content
        assert "dry_joint" in content
        assert "incorrect_installation" in content
        assert "pcb_damage" in content
        assert "short_circuit" in content
        assert "Mousebites" not in content
        assert "Opens" not in content

    def test_class_count(self):
        from train_yolo import CLASSES, NC

        assert NC == 4
        assert len(CLASSES) == 4

    def test_create_val_split(self, tmp_path):
        from train_yolo import _create_val_split

        train_dir = tmp_path / "train"
        (train_dir / "images").mkdir(parents=True)
        (train_dir / "labels").mkdir(parents=True)

        import cv2
        for i in range(10):
            img = np.zeros((10, 10, 3), dtype=np.uint8)
            cv2.imwrite(str(train_dir / "images" / f"img_{i:03d}.jpg"), img)
            (train_dir / "labels" / f"img_{i:03d}.txt").write_text("0 0.5 0.5 0.3 0.3")

        val_dir = tmp_path / "val"
        _create_val_split(train_dir, val_dir, val_ratio=0.2, seed=42)

        val_images = list((val_dir / "images").glob("*.jpg"))
        assert len(val_images) == 2  # 20% de 10


# ── Tests: utils ─────────────────────────────────────────────────────────

class TestUtils:
    def test_validate_image_format_valid(self):
        from utils import validate_image_format

        assert validate_image_format("photo.jpg") is True
        assert validate_image_format("photo.JPEG") is True
        assert validate_image_format("photo.png") is True

    def test_validate_image_format_invalid(self):
        from utils import validate_image_format

        assert validate_image_format("document.pdf") is False
        assert validate_image_format("data.csv") is False
        assert validate_image_format("image.bmp") is False

    def test_resize_image(self):
        from utils import resize_image

        img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        resized = resize_image(img, size=320)
        assert resized.shape == (320, 320, 3)

    def test_normalize_image(self):
        from utils import normalize_image

        img = np.ones((10, 10, 3), dtype=np.uint8) * 128
        normalized = normalize_image(img)
        assert normalized.dtype == np.float32
        assert normalized.min() >= 0.0
        assert normalized.max() <= 1.0

    def test_load_image_from_bytes(self):
        import cv2
        from utils import load_image_from_bytes

        img = np.zeros((100, 100, 3), dtype=np.uint8)
        _, buf = cv2.imencode(".jpg", img)
        loaded = load_image_from_bytes(buf.tobytes())
        assert loaded.shape == (100, 100, 3)

    def test_load_image_from_bytes_invalid(self):
        from utils import load_image_from_bytes

        with pytest.raises(ValueError, match="No se pudo decodificar"):
            load_image_from_bytes(b"not an image")

    def test_image_to_bytes(self):
        import cv2
        from utils import image_to_bytes

        img = np.zeros((100, 100, 3), dtype=np.uint8)
        data = image_to_bytes(img, ext=".jpg")
        assert len(data) > 0
        # Verificar que se puede decodificar
        arr = np.frombuffer(data, dtype=np.uint8)
        decoded = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        assert decoded is not None

    def test_draw_bbox(self):
        from utils import draw_bbox

        img = np.zeros((100, 100, 3), dtype=np.uint8)
        result = draw_bbox(img, 10, 10, 50, 50, "test 0.90", (0, 255, 0))
        assert result.shape == img.shape
        # Píxel dentro del bbox debe ser verde (o parte del rectángulo)
        assert result[10, 10, 1] == 255  # canal verde en esquina


# ── Tests: batch_receiver ─────────────────────────────────────────────────

class TestBatchReceiver:
    def _make_image_bytes(self, w: int = 100, h: int = 100) -> bytes:
        import cv2
        img = np.zeros((h, w, 3), dtype=np.uint8)
        _, buf = cv2.imencode(".jpg", img)
        return buf.tobytes()

    def test_receive_valid_batch(self):
        from batch_receiver import BatchReceiver

        receiver = BatchReceiver()
        files = [("img1.jpg", self._make_image_bytes()), ("img2.png", self._make_image_bytes())]
        batch = receiver.receive(files)

        assert batch.size == 2
        assert batch.status == "En procesamiento"
        assert batch.batch_id is not None

    def test_receive_empty_batch(self):
        from batch_receiver import BatchReceiver

        receiver = BatchReceiver()
        with pytest.raises(ValueError, match="vacío"):
            receiver.receive([])

    def test_receive_too_many_images(self):
        from batch_receiver import BatchReceiver

        receiver = BatchReceiver(max_size=3)
        files = [("img.jpg", self._make_image_bytes())] * 4
        with pytest.raises(ValueError, match="máximo es 3"):
            receiver.receive(files)

    def test_receive_invalid_format(self):
        from batch_receiver import BatchReceiver

        receiver = BatchReceiver()
        files = [("doc.pdf", b"fake pdf content")]
        with pytest.raises(ValueError, match="Formato inválido"):
            receiver.receive(files)

    def test_batch_image_is_resized(self):
        from batch_receiver import BatchReceiver

        receiver = BatchReceiver(image_size=64)
        img_bytes = self._make_image_bytes(w=200, h=300)
        batch = receiver.receive([("test.jpg", img_bytes)])

        assert batch.images[0].resized.shape == (64, 64, 3)

    def test_unique_batch_ids(self):
        from batch_receiver import BatchReceiver

        receiver = BatchReceiver()
        files = [("img.jpg", self._make_image_bytes())]
        b1 = receiver.receive(files)
        b2 = receiver.receive(files)
        assert b1.batch_id != b2.batch_id


# ── Tests: post_processor ─────────────────────────────────────────────────

class TestPostProcessor:
    def _make_batch_and_result(self, has_detections: bool = True):
        """Crea objetos Batch y BatchResult de prueba."""
        import cv2
        from batch_inference import BatchResult, Detection, ImageResult
        from batch_receiver import Batch, BatchImage

        img = np.zeros((100, 100, 3), dtype=np.uint8)
        _, buf = cv2.imencode(".jpg", img)
        img_bytes = buf.tobytes()

        bi = BatchImage(
            filename="test.jpg",
            original_bytes=img_bytes,
            resized=img,
            width_orig=100,
            height_orig=100,
        )
        batch = Batch(batch_id="test-batch", images=[bi])

        detections = []
        if has_detections:
            detections = [
                Detection(
                    class_name="dry_joint",
                    class_id=0,
                    confidence=0.85,
                    bbox=[0.1, 0.1, 0.5, 0.5],
                )
            ]

        image_result = ImageResult(
            filename="test.jpg",
            detections=detections,
            inference_time_ms=10.0,
        )
        batch_result = BatchResult(
            batch_id="test-batch",
            image_results=[image_result],
            total_time_ms=10.0,
        )
        return batch, batch_result

    def test_process_with_defects(self):
        from post_processor import PostProcessor

        processor = PostProcessor(conf_threshold=0.25)
        batch, batch_result = self._make_batch_and_result(has_detections=True)
        annotated = processor.process(batch, batch_result)

        assert len(annotated) == 1
        assert annotated[0].has_defects is True
        assert annotated[0].no_defect_notification is None
        assert len(annotated[0].annotated_bytes) > 0

    def test_process_without_defects(self):
        from post_processor import PostProcessor

        processor = PostProcessor(conf_threshold=0.25)
        batch, batch_result = self._make_batch_and_result(has_detections=False)
        annotated = processor.process(batch, batch_result)

        assert annotated[0].has_defects is False
        assert annotated[0].no_defect_notification == "✅ PCB sin defectos"

    def test_process_with_mask(self):
        from batch_inference import BatchResult, Detection, ImageResult
        from batch_receiver import Batch, BatchImage
        from post_processor import PostProcessor
        import cv2

        img = np.zeros((100, 100, 3), dtype=np.uint8)
        _, buf = cv2.imencode(".jpg", img)
        img_bytes = buf.tobytes()
        bi = BatchImage("img.jpg", img_bytes, img, 100, 100)
        batch = Batch(batch_id="b1", images=[bi])

        det = Detection(
            class_name="pcb_damage",
            class_id=2,
            confidence=0.9,
            bbox=[0.1, 0.1, 0.9, 0.9],
            mask_points=[[0.1, 0.1], [0.9, 0.1], [0.9, 0.9], [0.1, 0.9]],
        )
        ir = ImageResult(filename="img.jpg", detections=[det], inference_time_ms=5.0)
        br = BatchResult(batch_id="b1", image_results=[ir], total_time_ms=5.0)

        processor = PostProcessor()
        annotated = processor.process(batch, br)
        assert annotated[0].has_defects is True


# ── Tests: logger ────────────────────────────────────────────────────────

class TestLogger:
    def test_get_logger_returns_logger(self):
        from logger import get_logger

        log = get_logger("test_module")
        assert log is not None
        assert log.name == "test_module"

    def test_structured_format_produces_json(self):
        import logging
        from logger import StructuredFormatter

        formatter = StructuredFormatter()
        record = logging.LogRecord(
            name="test", level=logging.INFO, pathname="", lineno=0,
            msg="Hello %s", args=("world",), exc_info=None,
        )
        output = formatter.format(record)
        data = json.loads(output)
        assert data["message"] == "Hello world"
        assert data["level"] == "INFO"
        assert "timestamp" in data


# ── Tests: evaluate_model ─────────────────────────────────────────────────

class TestEvaluateModel:
    def test_find_dataset_yaml_uses_existing(self, tmp_path):
        from evaluate_model import _find_dataset_yaml

        model_dir = tmp_path / "model"
        model_dir.mkdir()
        yaml_content = "nc: 4\nnames:\n  0: dry_joint\n"
        yaml_path = model_dir / "dataset.yaml"
        yaml_path.write_text(yaml_content)

        result = _find_dataset_yaml(model_dir, tmp_path / "test", tmp_path / "output")
        assert result == yaml_path

    def test_find_dataset_yaml_generates_when_missing(self, tmp_path):
        from evaluate_model import _find_dataset_yaml

        model_dir = tmp_path / "model"
        model_dir.mkdir()
        test_dir = tmp_path / "test"
        test_dir.mkdir()
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        result = _find_dataset_yaml(model_dir, test_dir, output_dir)
        assert result is not None
        assert result.exists()
        content = result.read_text(encoding="utf-8")
        assert "nc: 4" in content

    def test_find_dataset_yaml_uses_training_summary(self, tmp_path):
        from evaluate_model import _find_dataset_yaml

        model_dir = tmp_path / "model"
        model_dir.mkdir()
        summary = {"nc": 4, "classes": ["dry_joint", "incorrect_installation", "pcb_damage", "short_circuit"]}
        (model_dir / "training_summary.json").write_text(json.dumps(summary))

        test_dir = tmp_path / "test"
        test_dir.mkdir()
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        yaml_path = _find_dataset_yaml(model_dir, test_dir, output_dir)
        content = yaml_path.read_text()
        assert "dry_joint" in content
