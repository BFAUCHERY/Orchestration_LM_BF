import pytest
from unittest.mock import MagicMock, patch
from unittest import mock
from pathlib import Path
from PIL import Image
import numpy as np
import cv2
import os
import tempfile
from orchestration_lm_bf.pipelines.OCRtesseract.pipeline import create_pipeline, get_detections, clean_text, extract_text

def test_OCRtesseract_pipeline_structure():
    pipeline = create_pipeline()
    assert pipeline is not None
    assert len(pipeline.nodes) > 0




@pytest.fixture
def temp_image_file():
    # Crée un fichier temporaire image
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        img = np.ones((100, 100, 3), dtype=np.uint8) * 255
        cv2.imwrite(tmp.name, img)
        yield tmp.name
        os.unlink(tmp.name)


def test_clean_text_corrections():
    assert clean_text("CUP") == "STOP"
    assert clean_text("5TOP") == "STOP"
    assert clean_text("SIOP") == "STOP"
    assert clean_text("ST0P") == "STOP"
    assert clean_text("STOPP") == "STOP"

def test_clean_text_digits():
    assert clean_text("50") == "50"
    assert clean_text("SPEED30") == "30"
    assert clean_text("ABC") == "ABC"

def test_get_detections(temp_image_file):
    class MockModel:
        def __call__(self, image, conf=0.2):
            class Box:
                xyxy = mock.Mock(detach=lambda: mock.Mock(numpy=lambda: np.array([[10, 10, 50, 50]])))
                conf = mock.Mock(detach=lambda: mock.Mock(numpy=lambda: np.array([0.9])))
                cls = mock.Mock(detach=lambda: mock.Mock(numpy=lambda: np.array([1])))
            result = mock.Mock()
            result.boxes = Box()
            return [result]

    detections = get_detections(MockModel(), Path(temp_image_file).parent.as_posix())
    assert len(detections) == 1
    det = detections[0]
    assert isinstance(det["boxes"], list)
    assert isinstance(det["score"], float)
    assert isinstance(det["class"], int)

def test_extract_text(monkeypatch, temp_image_file):
    fake_detection = [{
        'image_path': temp_image_file,
        'boxes': [0, 0, 100, 100]
    }]

    monkeypatch.setattr("pytesseract.image_to_string", lambda img, config: "STOP")
    
    results = extract_text(fake_detection)
    assert len(results) == 1
    assert results[0]['text'][0]['text'] == "STOP"
    assert results[0]['text'][0]['confidence'] >= 0.90