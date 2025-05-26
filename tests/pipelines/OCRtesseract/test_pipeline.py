import pytest
from unittest.mock import MagicMock, patch
from PIL import Image
import numpy as np
import os
import platform
import pytesseract

from orchestration_lm_bf.pipelines.OCRtesseract.nodes import (
    configure_tesseract,
    get_detections,
    prepare_ocr_data,
    evaluate_ocr
)


def test_configure_tesseract_env(monkeypatch):
    monkeypatch.setenv("TESSERACT_CMD", "/custom/path/tesseract")
    config = configure_tesseract()
    assert config["lang"] == "eng"
    assert "--psm" in config["config"]
    assert pytesseract.pytesseract.tesseract_cmd == "/custom/path/tesseract"


@patch("platform.system", return_value="Darwin")
def test_configure_tesseract_mac(mock_platform):
    os.environ.pop("TESSERACT_CMD", None)
    config = configure_tesseract()
    assert pytesseract.pytesseract.tesseract_cmd == "/opt/homebrew/bin/tesseract"
    assert isinstance(config, dict)


@patch("platform.system", return_value="Windows")
def test_configure_tesseract_windows(mock_platform):
    os.environ.pop("TESSERACT_CMD", None)
    config = configure_tesseract()
    assert "C:/Program Files/Tesseract-OCR/tesseract.exe" in pytesseract.pytesseract.tesseract_cmd
    assert os.environ.get("TESSDATA_PREFIX") == "C:/Program Files/Tesseract-OCR/tessdata"


@patch("orchestration_lm_bf.pipelines.OCRtesseract.nodes.Image.open")
def test_get_detections(mock_open):
    mock_open.return_value = Image.new("RGB", (100, 100))
    mock_model = MagicMock()
    mock_result = MagicMock()
    mock_result.boxes.xyxy.cpu().numpy.return_value = np.array([[10, 10, 50, 50]])
    mock_result.boxes.conf.cpu().numpy.return_value = np.array([0.95])
    mock_result.boxes.cls.cpu().numpy.return_value = np.array([1])
    mock_model.return_value = [mock_result]

    detections = get_detections(mock_model, "tests/test_data/")
    assert isinstance(detections, list)
    assert len(detections) == 1
    assert detections[0]["class"] == 1


@patch("orchestration_lm_bf.pipelines.OCRtesseract.nodes.Image.open")
def test_prepare_ocr_data(mock_open):
    fake_img = Image.new("RGB", (100, 100))
    mock_open.return_value = fake_img

    detections = [{
        "image_path": "tests/test_data/stop.png",
        "bbox": [10, 10, 90, 90]
    }]
    crops = prepare_ocr_data(detections)
    assert isinstance(crops, list)
    assert isinstance(crops[0], Image.Image)


@patch("pytesseract.image_to_string", return_value="STOP")
def test_evaluate_ocr_success(mock_ocr):
    fake_img = Image.new("RGB", (100, 100))
    crops = [fake_img]
    tess_config = {'lang': 'eng', 'config': '--psm 6'}
    result = evaluate_ocr(crops, tess_config, ground_truths=["STOP"])
    assert isinstance(result, dict)
    assert "texts" in result
    assert result["average_cer"] == 0.0


def test_evaluate_ocr_invalid_crop():
    small_img = Image.new("RGB", (0, 0))  # Taille invalide
    tess_config = {'lang': 'eng', 'config': '--psm 6'}
    result = evaluate_ocr([small_img], tess_config)
    assert result["average_cer"] is None