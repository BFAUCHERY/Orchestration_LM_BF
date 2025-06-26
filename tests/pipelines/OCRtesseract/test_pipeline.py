import pytest
from unittest.mock import MagicMock, patch
from pathlib import Path
from PIL import Image
import numpy as np
import cv2
import os

from orchestration_lm_bf.pipelines.OCRtesseract.nodes import (
    get_detections,
    extract_text
)

from orchestration_lm_bf.pipelines.OCRtesseract.pipeline import create_pipeline

def test_OCRtesseract_pipeline_structure():
    pipeline = create_pipeline()
    assert pipeline is not None
    assert len(pipeline.nodes) > 0


@pytest.fixture
def fake_image_folder(tmp_path):
    # Crée une image factice
    img_path = tmp_path / "test.png"
    img = Image.new("RGB", (100, 100), color=(255, 255, 255))
    img.save(img_path)
    return tmp_path


def test_get_detections(fake_image_folder):
    mock_model = MagicMock()
    mock_result = MagicMock()

    mock_result.boxes.xyxy.cpu().numpy.return_value = np.array([[10, 10, 90, 90]])
    mock_result.boxes.conf.cpu().numpy.return_value = np.array([0.95])
    mock_result.boxes.cls.cpu().numpy.return_value = np.array([1])
    mock_model.__call__.return_value = [mock_result]

    detections = get_detections(mock_model, str(fake_image_folder))

    assert isinstance(detections, list)
    assert len(detections) == 1
    assert detections[0]["score"] == 0.95
    assert detections[0]["class"] == 1
    assert detections[0]["boxes"] == [10.0, 10.0, 90.0, 90.0]


@patch("orchestration_lm_bf.pipelines.OCRtesseract.nodes.cv2.imread")
@patch("orchestration_lm_bf.pipelines.OCRtesseract.nodes.easyocr.Reader.readtext")
@patch("orchestration_lm_bf.pipelines.OCRtesseract.nodes.easyocr.Reader.__init__", return_value=None)
def test_extract_text_success(mock_init, mock_readtext, mock_imread, fake_image_folder):
    # Fake detection
    detection = {
        "image_path": str(fake_image_folder / "test.png"),
        "boxes": [10, 10, 90, 90],
        "score": 0.9,
        "class": 1
    }

    fake_img = np.ones((100, 100, 3), dtype=np.uint8)
    mock_imread.return_value = fake_img

    mock_readtext.return_value = [
        ([[[10, 10], [90, 10], [90, 90], [10, 90]]], "STOP", 0.98)
    ]

    results = extract_text([detection])

    assert isinstance(results, list)
    assert len(results) == 1
    assert results[0]["image_path"] == detection["image_path"]
    assert results[0]["text"][0]["text"] == "STOP"
    assert isinstance(results[0]["text"][0]["confidence"], float)
    assert isinstance(results[0]["text"][0]["bbox"], list)


@patch("orchestration_lm_bf.pipelines.OCRtesseract.nodes.cv2.imread", return_value=None)
def test_extract_text_error_image(mock_imread):
    detection = {
        "image_path": "nonexistent.png",
        "boxes": [0, 0, 100, 100],
        "score": 0.9,
        "class": 1
    }

    results = extract_text([detection])
    assert results == []