from kedro.pipeline import Pipeline, node, pipeline
from .nodes import train_yolo, evaluate_yolo

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=train_yolo,
            inputs="params:yolo_yaml_path",
            outputs="trained_model",
            name="train_yolo_node"
        ),
        node(
            func=evaluate_yolo,
            inputs="trained_model",
            outputs="evaluation_metrics",
            name="evaluate_yolo_node"
        ),
    ])