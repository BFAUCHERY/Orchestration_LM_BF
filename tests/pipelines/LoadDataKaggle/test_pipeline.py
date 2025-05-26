import pytest
import os
import numpy as np
from unittest import mock
from orchestration_lm_bf.pipelines.LoadDataKaggle.nodes import load_and_prepare_gtsrb_data
from orchestration_lm_bf.pipelines.LoadDataKaggle.pipeline import create_pipeline

def test_LoadDataKaggle_pipeline_structure():
    pipeline = create_pipeline()
    assert pipeline is not None
    assert len(pipeline.nodes) > 0

def test_load_and_prepare_gtsrb_data(monkeypatch, tmp_path):
    # Simule l'appel à l'API Kaggle pour éviter un vrai téléchargement
    class MockKaggleAPI:
        def dataset_download_files(self, *args, **kwargs):
            zip_path = tmp_path / "traffic-sign-gtrb.zip"
            with open(zip_path, "wb") as f:
                f.write(b"faux contenu zip")
            print(f"[TEST] Fichier ZIP simulé à : {zip_path}")
    monkeypatch.setattr("orchestration_lm_bf.pipelines.LoadDataKaggle.nodes.kaggle.api", MockKaggleAPI())

    # Simule l'extraction du zip avec une classe contextuelle valide
    class MockZipFile:
        def __init__(self, file, mode='r'):
            pass
        def extractall(self, path):
            print(f"[TEST] Extraction simulée vers : {path}")
        def __enter__(self):
            return self
        def __exit__(self, exc_type, exc_val, exc_tb):
            pass
    monkeypatch.setattr("orchestration_lm_bf.pipelines.LoadDataKaggle.nodes.zipfile.ZipFile", MockZipFile)

    # Crée un faux chemin d'image simulé
    simulated_image_path = tmp_path / "train" / "test.png"
    simulated_image_path.parent.mkdir(parents=True, exist_ok=True)
    simulated_image_path.write_bytes(b"faux contenu image")

    # Simule la détection des images dans le dossier
    monkeypatch.setattr("orchestration_lm_bf.pipelines.LoadDataKaggle.nodes.glob.glob", lambda *args, **kwargs: [str(simulated_image_path)])

    # Simule la lecture d'une image avec cv2
    monkeypatch.setattr("orchestration_lm_bf.pipelines.LoadDataKaggle.nodes.cv2.imread", lambda path: [[255, 255, 255]])

    # Exécute la fonction de chargement
    result = load_and_prepare_gtsrb_data()
    print(f"[TEST] Images chargées : {result}")

    # Vérifie que le dictionnaire retourné est correct
    assert isinstance(result, dict)
    assert "test.png" in result
    assert result["test.png"] == [[255, 255, 255]]