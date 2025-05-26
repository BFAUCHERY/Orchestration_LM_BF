import os
import pytest
from unittest.mock import MagicMock, patch
from PIL import Image
import numpy as np

from orchestration_lm_bf.pipelines.OCRtesseract.pipeline import create_pipeline
from orchestration_lm_bf.pipelines.OCRtesseract.nodes import (
    configure_tesseract,
    get_detections,
    prepare_ocr_data,
    evaluate_ocr
)

def test_ocr_pipeline_structure():
    pipeline = create_pipeline()
    assert pipeline is not None
    assert len(pipeline.nodes) > 0

def test_configure_tesseract_sets_env(monkeypatch):
    monkeypatch.setenv("TESSERACT_CMD", "/usr/bin/tesseract")
    config = configure_tesseract()
    assert isinstance(config, dict)
    assert 'lang' in config
    assert 'config' in config

@patch("orchestration_lm_bf.pipelines.OCRtesseract.nodes.Image.open")
def test_get_detections(mock_open):
    mock_open.return_value = Image.new("RGB", (100, 100))

    mock_model = MagicMock()
    mock_result = MagicMock()
    mock_result.boxes.xyxy.cpu().numpy.return_value = np.array([[10, 10, 50, 50]])
    mock_result.boxes.conf.cpu().numpy.return_value = np.array([0.9])
    mock_result.boxes.cls.cpu().numpy.return_value = np.array([1])
    mock_model.return_value = [mock_result]

    detections = get_detections(mock_model, "tests/test_data/")
    assert isinstance(detections, list)
    assert len(detections) == 1
    assert "bbox" in detections[0]

@patch("orchestration_lm_bf.pipelines.OCRtesseract.nodes.Image.open")
def test_prepare_ocr_data(mock_open):
    fake_img = Image.new("RGB", (100, 100))
    mock_open.return_value = fake_img

    detections = [{
        "image_path": "tests/test_data/stop.png",
        "bbox": [10, 10, 100, 100]
    }]
    crops = prepare_ocr_data(detections)
    assert len(crops) == 1
    assert isinstance(crops[0], Image.Image)

def test_evaluate_ocr_real_stop(monkeypatch):
    monkeypatch.setenv("TESSERACT_CMD", "/usr/bin/tesseract")
    img = Image.open("tests/test_data/stop.png")
    crops = [img.crop((10, 10, 100, 100))]
    tess_config = {'lang': 'eng', 'config': '--psm 6'}
    ground_truths = ["STOP"]

    result = evaluate_ocr(crops, tess_config, ground_truths)
    assert isinstance(result, dict)
    assert "texts" in result
    assert "average_cer" in result
    assert result["average_cer"] is not None