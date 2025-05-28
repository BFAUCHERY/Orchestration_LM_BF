import pytest
from unittest.mock import MagicMock, patch
from orchestration_lm_bf.pipelines.detectionYOLO.nodes import train_yolo
from orchestration_lm_bf.pipelines.detectionYOLO.pipeline import create_pipeline

def test_detectionYOLO_pipeline_structure():
    pipeline = create_pipeline()
    assert pipeline is not None
    assert len(pipeline.nodes) > 0

@patch("orchestration_lm_bf.pipelines.detectionYOLO.nodes.YOLO")
def test_train_yolo_mock(mock_yolo_class):
    # Simule l'objet modèle YOLO
    mock_model = MagicMock()
    mock_yolo_class.return_value = mock_model

    print("TEST: Mocking YOLO training - this should not run actual training.")
    # Appelle la fonction avec un chemin fictif
    model_path = train_yolo("fake_data.yaml")

    # Vérifie que YOLO a bien été instancié avec le bon fichier
    mock_yolo_class.assert_called_once_with("model/yolov8n.pt")

    # Vérifie que train() a été appelé avec les bons arguments
    mock_model.train.assert_called_once_with(
        data="fake_data.yaml",
        epochs=1,
        imgsz=160,
        batch=4,
        device="cpu",
        patience=0
    )
    print("TEST: Training function was mocked successfully.")

    # Vérifie que le chemin retourné est correct
    assert model_path == "model/model.pt"