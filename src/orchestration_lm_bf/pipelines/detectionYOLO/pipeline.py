"""
This is a boilerplate pipeline 'detectionYOLO'
generated using Kedro 0.19.12
"""

from kedro.pipeline import node, Pipeline, pipeline  # noqa
from .nodes import load_data, preprocess_data, train_yolov8, evaluate_yolov8


def create_pipeline(**kwargs) -> Pipeline:

    return pipeline([
        node(
            func=load_data,
            inputs=["train_annotations", "params:image_root"],
            outputs="annotations_df",
            name="load_data_node"
            ),
        node(
            func=preprocess_data,
            inputs=["annotations_df","yolo_dataset"],
            outputs=None,
            name="preprocess_data_node"
        ),
        node(
            func=train_yolov8,
            inputs=["yolo_dataset","model_output_path"],
            outputs="trained_model_output",
            name="train_model_node"
        ),
        node(
            func=evaluate_yolov8,
            inputs=["trained_model_output","yolo_dataset"],
            outputs="map50_score",
            name="evaluate_model_node"
        ),
    ])
