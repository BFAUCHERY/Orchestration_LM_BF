from kedro.pipeline import Pipeline, node, pipeline
from .nodes import evaluate_yolov8_10, predict_yolo

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
#        node(
#            func=evaluate_yolov8_10,
#            inputs=["params:evaluation_model_path", "params:data_yaml_path"],
#            outputs="evaluation_metrics",
#            name="evaluate_yolo_node"
#        ),
        node(
            func=predict_yolo,
            inputs=["params:evaluation_model_path","params:predict_image_path"],
            outputs="yolo_predictions",
            name="predict_yolo_node"
        ),
    ])