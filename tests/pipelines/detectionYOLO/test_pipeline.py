import pytest
from unittest.mock import MagicMock, patch
from orchestration_lm_bf.pipelines.detectionYOLO.nodes import train_yolo

@patch("orchestration_lm_bf.pipelines.detectionYOLO.nodes.YOLO")
def test_train_yolo_mock(mock_yolo_class):
    # Simule l'objet modèle YOLO
    mock_model = MagicMock()
    mock_yolo_class.return_value = mock_model

    # Appelle la fonction avec un chemin fictif
    model_path = train_yolo("fake_data.yaml")

    # Vérifie que train() a été appelé avec les bons arguments
    mock_model.train.assert_called_once_with(
        data="fake_data.yaml",
        epochs=1,
        imgsz=160,
        batch=4,
        device="cpu",
        patience=0
    )

    # Vérifie que le chemin retourné est correct
    assert model_path == "data/model.pt"