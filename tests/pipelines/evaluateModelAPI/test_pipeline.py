import pytest
from unittest.mock import patch, mock_open, MagicMock
from orchestration_lm_bf.pipelines.evaluateModelAPI.nodes import evaluate_model_api_node
from orchestration_lm_bf.pipelines.evaluateModelAPI.pipeline import create_pipeline


def test_evaluateModelAPI_pipeline_structure():
    pipeline = create_pipeline()
    assert pipeline is not None
    assert len(pipeline.nodes) > 0

@pytest.fixture
def mock_response():
    mock = MagicMock()
    mock.json.return_value = {
        "predictions": [
            {
                "class": "stop",
                "confidence": 0.95,
                "x": 100,
                "y": 150,
                "width": 50,
                "height": 50,
                "detection_id": "abc123"
            },
            {
                "class": "go",
                "confidence": 0.89,
                "x": 300,
                "y": 350,
                "width": 60,
                "height": 60,
                "detection_id": "def456"
            }
        ],
        "image": {"width": 640, "height": 480},
        "time": 0.123
    }
    mock.raise_for_status = MagicMock()
    return mock

@patch("shutil.copy")
@patch("builtins.open", new_callable=mock_open, read_data=b"fake image data")
@patch("os.path.exists", return_value=True)
@patch("requests.post")
def test_single_image_success(mock_post, mock_exists, mock_file, mock_copy, mock_response):
    mock_post.return_value = mock_response

    result = evaluate_model_api_node(
        image_source={"single_image_path": "data/test.jpg"},
        api_key="FAKE_API_KEY",
        project_id="test_project",
        model_version="1",
        confidence=0.5
    )

    assert result["summary"]["total_images_processed"] == 1
    assert result["summary"]["successful_predictions"] == 1
    assert result["summary"]["total_detections"] == 2
    assert not result["errors"]

@patch("shutil.copy")
@patch("os.listdir", return_value=["img1.jpg", "img2.jpg"])
@patch("builtins.open", new_callable=mock_open, read_data=b"fake image data")
@patch("os.path.exists", return_value=True)
@patch("requests.post")
def test_folder_images_success(mock_post, mock_exists, mock_file, mock_listdir, mock_copy, mock_response):
    mock_post.return_value = mock_response

    result = evaluate_model_api_node(
        image_source="data/images",
        api_key="FAKE_API_KEY",
        project_id="test_project",
        model_version="1",
        confidence=0.5
    )

    assert result["summary"]["total_images_processed"] == 2
    assert result["summary"]["successful_predictions"] == 2
    assert result["summary"]["total_detections"] == 4
    assert not result["errors"]

@patch("os.path.exists", return_value=False)
def test_single_image_not_found(mock_exists):
    result = evaluate_model_api_node(
        image_source={"single_image_path": "nonexistent.jpg"},
        api_key="FAKE_API_KEY",
        project_id="test_project",
        model_version="1",
        confidence=0.5
    )
    assert "Image non trouvée" in result["errors"][0]
    assert result["summary"] == {}

@patch("os.path.exists", return_value=False)
def test_folder_not_found(mock_exists):
    result = evaluate_model_api_node(
        image_source="nonexistent_folder",
        api_key="FAKE_API_KEY",
        project_id="test_project",
        model_version="1",
        confidence=0.5
    )
    assert "Dossier non trouvé" in result["errors"][0]
    assert result["summary"] == {}

@patch("os.listdir", return_value=[])
@patch("os.path.exists", return_value=True)
def test_folder_empty(mock_exists, mock_listdir):
    result = evaluate_model_api_node(
        image_source="empty_folder",
        api_key="FAKE_API_KEY",
        project_id="test_project",
        model_version="1",
        confidence=0.5
    )
    assert "Aucune image trouvée" in result["errors"][0]
    assert result["summary"] == {}