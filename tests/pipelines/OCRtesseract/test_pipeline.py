import pytest
from unittest.mock import MagicMock, patch
from pathlib import Path
from PIL import Image
import numpy as np
import cv2
import os

from orchestration_lm_bf.pipelines.OCRtesseract.pipeline import create_pipeline

def test_OCRtesseract_pipeline_structure():
    pipeline = create_pipeline()
    assert pipeline is not None
    assert len(pipeline.nodes) > 0