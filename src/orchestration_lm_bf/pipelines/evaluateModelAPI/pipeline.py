
from kedro.pipeline import Pipeline, node, pipeline
from .nodes import evaluate_model_api_node

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=evaluate_model_api_node,
            inputs=[
                "params:image_source",  # Changé de image_folder à image_source
                "params:api_key",
                "params:project_id",
                "params:model_version",
                "params:confidence"
            ],
            outputs="roboflow_predictions_raw",
            name="evaluate_model_api_node",
        )
    ])