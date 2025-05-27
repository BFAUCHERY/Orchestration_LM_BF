"""
This is a boilerplate test file for pipeline 'ocrAPI'
generated using Kedro 0.19.12.
Please add your pipeline tests here.

Kedro recommends using `pytest` framework, more info about it can be found
in the official documentation:
https://docs.pytest.org/en/latest/getting-started.html
"""

import pytest
import numpy as np
from unittest.mock import patch, MagicMock
from orchestration_lm_bf.pipelines.ocrAPI.nodes import prepare_crops_from_roboflow, extract_text_from_crops
from orchestration_lm_bf.pipelines.ocrAPI.pipeline import create_pipeline

def test_ocrAPI_pipeline_structure():
    pipeline = create_pipeline()
    assert pipeline is not None
    assert len(pipeline.nodes) > 0

@patch("cv2.imread")
@patch("cv2.imwrite")
@patch("os.remove")
def test_prepare_crops_from_roboflow(mock_remove, mock_imwrite, mock_imread):
    dummy_img = np.ones((200, 200, 3), dtype=np.uint8) * 255
    mock_imread.return_value = dummy_img

    predictions_dict = {
        "predictions_by_image": {
            "test_image.jpg": {
                "predictions": [
                    {
                        "x": 100,
                        "y": 100,
                        "width": 50,
                        "height": 50,
                        "confidence": 0.9,
                        "class": "text"
                    }
                ]
            }
        }
    }

    base_folder = "tests/resources"
    crops = prepare_crops_from_roboflow(predictions_dict, base_folder)
    assert isinstance(crops, list)
    assert len(crops) == 1
    assert crops[0].shape[0] > 0 and crops[0].shape[1] > 0

@patch("easyocr.Reader.readtext")
@patch("easyocr.Reader.__init__", return_value=None)
def test_extract_text_from_crops(mock_reader_init, mock_readtext):
    mock_readtext.return_value = [
        ([[0, 0], [1, 0], [1, 1], [0, 1]], "Hello", 0.95)
    ]
    dummy_crop = np.ones((100, 100, 3), dtype=np.uint8) * 255
    results = extract_text_from_crops([dummy_crop])
    assert isinstance(results, list)
    assert len(results) == 1
    assert results[0][0]["text"] == "Hello"
    assert "confidence" in results[0][0]


# Additional tests for prepare_crops_from_roboflow error handling
from unittest.mock import patch
import numpy as np

@patch("cv2.imread", return_value=None)
@patch("os.remove")
def test_prepare_crop_image_not_read(mock_remove, mock_imread):
    predictions = {
        "predictions_by_image": {
            "img.jpg": {
                "predictions": [{"x": 100, "y": 100, "width": 50, "height": 50}]
            }
        }
    }
    crops = prepare_crops_from_roboflow(predictions, "base/folder")
    assert crops == []


@patch("cv2.imread")
@patch("os.remove")
def test_prepare_crop_missing_coordinates(mock_remove, mock_imread):
    dummy_img = np.ones((200, 200, 3), dtype=np.uint8) * 255
    mock_imread.return_value = dummy_img

    predictions = {
        "predictions_by_image": {
            "img.jpg": {
                "predictions": [{"x": None, "y": 100, "width": 50, "height": 50}]
            }
        }
    }
    crops = prepare_crops_from_roboflow(predictions, "base/folder")
    assert crops == []


@patch("cv2.imread")
@patch("cv2.imwrite")
@patch("os.remove")
def test_prepare_crop_empty_crop(mock_remove, mock_imwrite, mock_imread):
    dummy_img = np.ones((10, 10, 3), dtype=np.uint8) * 255
    mock_imread.return_value = dummy_img

    predictions = {
        "predictions_by_image": {
            "img.jpg": {
                "predictions": [{"x": 5, "y": 5, "width": 1, "height": 1}]
            }
        }
    }
    # This crop will likely fall outside the image boundary after padding
    crops = prepare_crops_from_roboflow(predictions, "base/folder")
    assert isinstance(crops, list)
    assert len(crops) == 1
    assert isinstance(crops[0], np.ndarray)
    assert crops[0].shape == (10, 10, 3)
