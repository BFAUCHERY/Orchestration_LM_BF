from orchestration_lm_bf.pipeline_registry import register_pipelines

def test_register_pipelines_structure():
    pipelines = register_pipelines()
    assert "detectionYOLO" in pipelines
    assert "evaluate_yolov8_10" in pipelines