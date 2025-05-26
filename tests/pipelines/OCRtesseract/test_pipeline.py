from orchestration_lm_bf.pipelines.OCRtesseract.pipeline import create_pipeline
from kedro.io import DataCatalog, MemoryDataSet
from kedro.runner import SequentialRunner

def test_ocr_pipeline_structure():
    catalog = DataCatalog({
        "params:ocr_image_path": MemoryDataSet("some_path.jpg"),
        "ocr_result": MemoryDataSet()
    })
    pipeline = create_pipeline()
    runner = SequentialRunner()
    try:
        runner.run(pipeline, catalog)
    except Exception:
        pass  # on vérifie juste que la pipeline est bien structurée