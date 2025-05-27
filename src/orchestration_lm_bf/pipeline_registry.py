"""Project pipelines."""
from __future__ import annotations

from kedro.framework.project import find_pipelines
from kedro.pipeline import Pipeline
from orchestration_lm_bf.pipelines.LoadDataKaggle import pipeline as LoadDataKaggle
from orchestration_lm_bf.pipelines.detectionYOLO import pipeline as detectionYOLO
from orchestration_lm_bf.pipelines.evaluateYOLO import pipeline as evaluateYOLO
from orchestration_lm_bf.pipelines.OCRtesseract import pipeline as OCRtesseract


def register_pipelines() -> dict[str, Pipeline]:
    """Register the project's pipelines.

    Returns:
        A mapping from pipeline names to ``Pipeline`` objects.
    """
    pipelines = find_pipelines()

    pipelines["LoadDataKaggle"] = LoadDataKaggle.create_pipeline()
    pipelines["detectionYOLO"] = detectionYOLO.create_pipeline()
    pipelines["evaluateYOLO"] = evaluateYOLO.create_pipeline()
    pipelines["OCRtesseract"] = OCRtesseract.create_pipeline()

    pipelines["local_pipeline"] = (
        pipelines["LoadDataKaggle"]
        + pipelines["detectionYOLO"]
        + pipelines["evaluateYOLO"]
        + pipelines["OCRtesseract"]
    )

    pipelines["__default__"] = pipelines["local_pipeline"]
    return pipelines
