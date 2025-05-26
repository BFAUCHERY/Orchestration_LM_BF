import pytest
import os
import numpy as np
from pathlib import Path
from unittest import mock
from orchestration_lm_bf.pipelines.LoadDataKaggle.nodes import load_and_prepare_gtsrb_data
from orchestration_lm_bf.pipelines.LoadDataKaggle.pipeline import create_pipeline

def test_LoadDataKaggle_pipeline_structure():
    pipeline = create_pipeline()
    assert pipeline is not None
    assert len(pipeline.nodes) > 0

def test_load_and_prepare_gtsrb_data(monkeypatch, tmp_path):
    class MockKaggleAPI:
        def dataset_download_files(self, *args, **kwargs):
            zip_path = tmp_path / "traffic-sign-gtrb.zip"
            with open(zip_path, "wb") as f:
                f.write(b"fake zip content")

    monkeypatch.setattr("orchestration_lm_bf.pipelines.LoadDataKaggle.nodes.kaggle.api", MockKaggleAPI())

    class MockZipFile:
        def __init__(self, file, mode='r'):
            pass
        def extractall(self, path):
            (Path(path) / "train").mkdir(parents=True, exist_ok=True)
        def __enter__(self): return self
        def __exit__(self, exc_type, exc_val, exc_tb): pass

    monkeypatch.setattr("orchestration_lm_bf.pipelines.LoadDataKaggle.nodes.zipfile.ZipFile", MockZipFile)

    fake_img_path = tmp_path / "traffic-sign-gtrb" / "train" / "test.png"
    fake_img_path.parent.mkdir(parents=True, exist_ok=True)
    fake_img_path.write_bytes(b"fake")

    monkeypatch.setattr("orchestration_lm_bf.pipelines.LoadDataKaggle.nodes.glob.glob", lambda *args, **kwargs: [str(fake_img_path)])
    monkeypatch.setattr("orchestration_lm_bf.pipelines.LoadDataKaggle.nodes.cv2.imread", lambda path: [[255, 255, 255]])

    result = load_and_prepare_gtsrb_data()
    assert isinstance(result, dict)
    assert "test.png" in result
    assert result["test.png"] == [[255, 255, 255]]

def test_load_and_prepare_gtsrb_data_handles_corrupt(monkeypatch, tmp_path):
    class MockKaggleAPI:
        def dataset_download_files(self, *args, **kwargs):
            zip_path = tmp_path / "traffic-sign-gtrb.zip"
            with open(zip_path, "wb") as f:
                f.write(b"fake zip")

    monkeypatch.setattr("orchestration_lm_bf.pipelines.LoadDataKaggle.nodes.kaggle.api", MockKaggleAPI())

    class MockZipFile:
        def __init__(self, file, mode='r'):
            pass
        def extractall(self, path): pass
        def __enter__(self): return self
        def __exit__(self, exc_type, exc_val, exc_tb): pass

    monkeypatch.setattr("orchestration_lm_bf.pipelines.LoadDataKaggle.nodes.zipfile.ZipFile", MockZipFile)

    bad_img_path = tmp_path / "traffic-sign-gtrb" / "train" / "bad.png"
    bad_img_path.parent.mkdir(parents=True, exist_ok=True)
    bad_img_path.write_bytes(b"")

    monkeypatch.setattr("orchestration_lm_bf.pipelines.LoadDataKaggle.nodes.glob.glob", lambda *args, **kwargs: [str(bad_img_path)])
    monkeypatch.setattr("orchestration_lm_bf.pipelines.LoadDataKaggle.nodes.cv2.imread", lambda path: None)

    result = load_and_prepare_gtsrb_data()
    assert result == {}

def test_load_and_prepare_gtsrb_data_zip_not_exist(monkeypatch, tmp_path):
    # Simuler absence de zip après téléchargement
    class MockKaggleAPI:
        def dataset_download_files(self, *args, **kwargs):
            pass  # ne crée pas de zip

    monkeypatch.setattr("orchestration_lm_bf.pipelines.LoadDataKaggle.nodes.kaggle.api", MockKaggleAPI())

    monkeypatch.setattr("orchestration_lm_bf.pipelines.LoadDataKaggle.nodes.os.path.exists", lambda path: False)
    monkeypatch.setattr("orchestration_lm_bf.pipelines.LoadDataKaggle.nodes.glob.glob", lambda *args, **kwargs: [])
    
    result = load_and_prepare_gtsrb_data()
    assert isinstance(result, dict)
    assert result == {}