from orchestration_lm_bf.pipeline_registry import register_pipelines

def test_register_pipelines_returns_dict():
    pipelines = register_pipelines()
    assert isinstance(pipelines, dict)