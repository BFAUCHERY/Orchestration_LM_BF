import pytest
from unittest.mock import MagicMock, patch
from orchestration_lm_bf.pipelines.evaluateYOLO.nodes import evaluate_yolov8_10

@patch("orchestration_lm_bf.pipelines.evaluateYOLO.nodes.YOLO")
def test_evaluate_yolov8_10_mock(mock_yolo_class):
    # Simule l'objet retourné par YOLO().val()
    mock_model = MagicMock()
    mock_model.val.return_value.results_dict = {
        "metrics/mAP_0.5": 0.75,
        "metrics/precision": 0.85
    }
    mock_yolo_class.return_value = mock_model

    model_path = "fake_model.pt"
    data_yaml_path = "fake_data.yaml"

    metrics = evaluate_yolov8_10(model_path, data_yaml_path)

    assert isinstance(metrics, dict)
    assert metrics["metrics/mAP_0.5"] == 0.75