import pytest
import os
from unittest import mock
from orchestration_lm_bf.pipelines.LoadDataKaggle.nodes import load_and_prepare_gtsrb_data

def test_load_and_prepare_gtsrb_data(monkeypatch, tmp_path):
    # Simule le fichier ZIP existant
    zip_path = tmp_path / "traffic-sign-gtrb.zip"
    zip_path.write_text("fake zip content")

    # Simule l'appel à l'API Kaggle
    class MockKaggleAPI:
        def dataset_download_files(self, *args, **kwargs):
            pass  # rien à faire ici car on crée manuellement le fichier zip

    monkeypatch.setattr(
        "orchestration_lm_bf.pipelines.LoadDataKaggle.nodes.kaggle.api", MockKaggleAPI()
    )

    # Simule le zipfile.ZipFile context manager
    class MockZipFile:
        def __init__(self, file, mode='r'):
            assert os.path.exists(file)
        def extractall(self, path):
            os.makedirs(path, exist_ok=True)
        def __enter__(self):
            return self
        def __exit__(self, exc_type, exc_val, exc_tb):
            pass

    monkeypatch.setattr(
        "orchestration_lm_bf.pipelines.LoadDataKaggle.nodes.zipfile.ZipFile", MockZipFile
    )

    # Crée une image simulée et un faux chemin
    image_path = tmp_path / "traffic-sign-gtrb" / "folder" / "img1.png"
    image_path.parent.mkdir(parents=True, exist_ok=True)
    image_path.write_bytes(b"fake_image_data")

    monkeypatch.setattr(
        "orchestration_lm_bf.pipelines.LoadDataKaggle.nodes.glob.glob",
        lambda *args, **kwargs: [str(image_path)]
    )

    monkeypatch.setattr(
        "orchestration_lm_bf.pipelines.LoadDataKaggle.nodes.os.path.exists",
        lambda path: str(path) == str(zip_path)
    )

    monkeypatch.setattr(
        "orchestration_lm_bf.pipelines.LoadDataKaggle.nodes.os.remove",
        lambda path: None  # ignorer suppression
    )

    monkeypatch.setattr(
        "orchestration_lm_bf.pipelines.LoadDataKaggle.nodes.cv2.imread",
        lambda path: [[123]]  # simule une image lue
    )

    result = load_and_prepare_gtsrb_data()
    assert isinstance(result, dict)
    assert "img1.png" in result
    assert result["img1.png"] == [[123]]