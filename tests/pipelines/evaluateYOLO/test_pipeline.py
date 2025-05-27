import pytest
from unittest.mock import patch, MagicMock
from orchestration_lm_bf.pipelines.evaluateYOLO.nodes import evaluate_yolov8_10, predict_yolo
from orchestration_lm_bf.pipelines.evaluateYOLO.pipeline import create_pipeline


def test_evaluateYOLO_pipeline_structure():
    pipeline = create_pipeline()
    assert pipeline is not None
    assert len(pipeline.nodes) > 0


def test_evaluate_yolov8_10_returns_metrics_dict():
    with patch("orchestration_lm_bf.pipelines.evaluateYOLO.nodes.YOLO") as mock_yolo_class:
        mock_model = MagicMock()
        mock_model.val.return_value.results_dict = {
            "metrics/mAP_0.5": 0.85,
            "metrics/recall": 0.9
        }
        mock_yolo_class.return_value = mock_model

        metrics = evaluate_yolov8_10("model.pt", "data.yaml")

        assert isinstance(metrics, dict)
        assert "metrics/mAP_0.5" in metrics
        assert metrics["metrics/recall"] == 0.9


def test_predict_yolo_returns_predictions():
    with patch("orchestration_lm_bf.pipelines.evaluateYOLO.nodes.YOLO") as mock_yolo_class:
        mock_result = MagicMock()
        mock_result.boxes.xyxy.cpu().numpy.return_value = [[0, 0, 100, 100]]
        mock_result.boxes.conf.cpu().numpy.return_value = [0.95]
        mock_result.boxes.cls.cpu().numpy.return_value = [1]

        mock_model = MagicMock()
        mock_model.predict.return_value = [mock_result]
        mock_yolo_class.return_value = mock_model

        predictions = predict_yolo("model.pt", "image.png")

        assert isinstance(predictions, list)
        assert len(predictions) == 1
        assert "boxes" in predictions[0]
        assert "scores" in predictions[0]
        assert "classes" in predictions[0]