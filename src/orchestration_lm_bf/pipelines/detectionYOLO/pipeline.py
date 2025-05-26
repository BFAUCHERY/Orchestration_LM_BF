from kedro.pipeline import node, pipeline
from .nodes import train_yolo

def create_pipeline(**kwargs):
    return pipeline([
        node(
            func=train_yolo,
            inputs="params:yolo_yaml_path",  # référence à un paramètre Kedro
            outputs="trained_model_path",
            name="train_yolo_node"
        )
    ])