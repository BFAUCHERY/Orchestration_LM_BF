from unittest.mock import MagicMock, patch
import numpy as np
import pytest
from orchestration_lm_bf.pipelines.evaluateYOLO.pipeline import create_pipeline
from orchestration_lm_bf.pipelines.evaluateYOLO.nodes import predict_yolo

def test_evaluateYOLO_pipeline_structure():
    pipeline = create_pipeline()
    assert pipeline is not None
    assert len(pipeline.nodes) > 0

@patch("orchestration_lm_bf.pipelines.evaluateYOLO.nodes.YOLO")
def test_predict_yolo_returns_predictions(mock_yolo_class):
    mock_result = MagicMock()

    mock_result.boxes.xyxy.cpu.return_value.numpy.return_value = np.array([[0, 0, 100, 100]])
    mock_result.boxes.conf.cpu.return_value.numpy.return_value = np.array([0.95])
    mock_result.boxes.cls.cpu.return_value.numpy.return_value = np.array([1])
    mock_result.names = {0: "stop", 1: "yield", 2: "speed_limit"}

    mock_model = MagicMock()
    mock_model.predict.return_value = [mock_result]
    mock_yolo_class.return_value = mock_model

    predictions = predict_yolo("model.pt", "image.png")

    assert len(predictions) == 1
    assert predictions[0]["boxes"] == [[0, 0, 100, 100]]
    assert predictions[0]["scores"] == [0.95]
    assert predictions[0]["classes"] == ["yield"]  # id=1 => "yield"

@patch("orchestration_lm_bf.pipelines.evaluateYOLO.nodes.YOLO")
def test_predict_yolo_returns_predictions_simple(mock_yolo_class):
    mock_result = MagicMock()

    boxes_array = np.array([[0, 0, 100, 100]])
    conf_array = np.array([0.95])
    cls_array = np.array([1])

    mock_result.boxes.xyxy.cpu.return_value.numpy.return_value = boxes_array
    mock_result.boxes.conf.cpu.return_value.numpy.return_value = conf_array
    mock_result.boxes.cls.cpu.return_value.numpy.return_value = cls_array

    mock_result.names = {0: "stop", 1: "yield", 2: "speed_limit"}

    mock_model = MagicMock()
    mock_model.predict.return_value = [mock_result]
    mock_yolo_class.return_value = mock_model

    predictions = predict_yolo("model.pt", "image.png")

    assert len(predictions) == 1
    assert predictions[0]["boxes"] == [[0, 0, 100, 100]]
    assert predictions[0]["scores"] == [0.95]
    assert predictions[0]["classes"] == ["yield"]