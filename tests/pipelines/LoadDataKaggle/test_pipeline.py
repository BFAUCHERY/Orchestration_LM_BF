import pytest
from orchestration_lm_bf.pipelines.LoadDataKaggle.nodes import load_and_prepare_gtsrb_data

def test_load_and_prepare_gtsrb_data(monkeypatch, tmp_path):
    # Monkeypatching Kaggle API
    class MockKaggleAPI:
        def dataset_download_files(self, *args, **kwargs):
            # simulate creation of zip file
            zip_path = tmp_path / "traffic-sign-gtrb.zip"
            with open(zip_path, "wb") as f:
                f.write(b"Fake zip content")
            print(f"[TEST LOG] Simulated zip file written to: {zip_path}")

    monkeypatch.setattr("orchestration_lm_bf.pipelines.LoadDataKaggle.nodes.kaggle.api", MockKaggleAPI())

    # Monkeypatch zipfile to avoid actual decompression
    monkeypatch.setattr("orchestration_lm_bf.pipelines.LoadDataKaggle.nodes.zipfile.ZipFile.extractall", lambda self, path: print(f"[TEST LOG] Simulated extraction to: {path}"))
    monkeypatch.setattr("orchestration_lm_bf.pipelines.LoadDataKaggle.nodes.zipfile.ZipFile.__init__", lambda self, file, mode='r': None)
    
    # Monkeypatch glob to simulate image files
    simulated_image_path = tmp_path / "train" / "test.png"
    monkeypatch.setattr("orchestration_lm_bf.pipelines.LoadDataKaggle.nodes.glob.glob", lambda *args, **kwargs: [str(simulated_image_path)])
    
    # Monkeypatch cv2.imread to return a fake image
    monkeypatch.setattr("orchestration_lm_bf.pipelines.LoadDataKaggle.nodes.cv2.imread", lambda path: [[255, 255, 255]])

    result = load_and_prepare_gtsrb_data()
    print(f"[TEST LOG] Image data loaded: {result}")

    assert isinstance(result, dict)
    assert "test.png" in result
    assert result["test.png"] == [[255, 255, 255]]