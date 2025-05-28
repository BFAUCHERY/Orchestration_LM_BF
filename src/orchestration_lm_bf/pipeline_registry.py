"""Project pipelines."""
from __future__ import annotations

from kedro.framework.project import find_pipelines
from kedro.pipeline import Pipeline
from orchestration_lm_bf.pipelines.LoadDataKaggle import pipeline as LoadDataKaggle
from orchestration_lm_bf.pipelines.detectionYOLO import pipeline as detectionYOLO
from orchestration_lm_bf.pipelines.evaluateYOLO import pipeline as evaluateYOLO
from orchestration_lm_bf.pipelines.evaluateModelAPI import pipeline as evaluateModelAPI
from orchestration_lm_bf.pipelines.OCRtesseract import pipeline as OCRtesseract
from orchestration_lm_bf.pipelines.ocrAPI import pipeline as ocrAPI



def register_pipelines() -> dict[str, Pipeline]:
    """Register the project's pipelines.

    Returns:
        A mapping from pipeline names to ``Pipeline`` objects.
    """
    pipelines = find_pipelines()

    pipelines["LoadDataKaggle"] = LoadDataKaggle.create_pipeline()
    pipelines["detectionYOLO"] = detectionYOLO.create_pipeline()
    pipelines["evaluateYOLO"] = evaluateYOLO.create_pipeline()
    pipelines["evaluateModelAPI"] = evaluateModelAPI.create_pipeline()
    pipelines["OCRtesseract"] = OCRtesseract.create_pipeline()
    pipelines["ocrAPI"] = ocrAPI.create_pipeline()


    pipelines["local_pipeline"] = (
        pipelines["evaluateYOLO"]
        + pipelines["OCRtesseract"]
    )

    pipelines["api_pipeline"] = (
        pipelines["evaluateModelAPI"]
        + pipelines["ocrAPI"]
    )

    pipelines["__default__"] = pipelines["api_pipeline"]
    return pipelines
