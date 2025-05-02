"""
This is a boilerplate pipeline 'detectionYOLO'
generated using Kedro 0.19.12
"""

from kedro.pipeline import node, Pipeline, pipeline  # noqa


def create_pipeline(**kwargs) -> Pipeline:
    from kedro.pipeline import Pipeline, node, pipeline
    from .nodes import load_data, preprocess_data, train_yolov8, evaluate_yolov8

    return pipeline([
        node(
            func=load_data,
            inputs=dict(
                test_csv_path="train_annotations",
                image_dir="image_root"
            ),
            outputs="annotations_df",
            name="load_data_node"
        ),
        node(
            func=preprocess_data,
            inputs=dict(
                df="annotations_df",
                output_dir="yolo_dataset"
            ),
            outputs=None,
            name="preprocess_data_node"
        ),
        node(
            func=train_yolov8,
            inputs=dict(
                data_path="yolo_dataset",
                model_path="model_output_path"
            ),
            outputs="trained_model_output",
            name="train_model_node"
        ),
        node(
            func=evaluate_yolov8,
            inputs=dict(
                model_path="trained_model_output",
                data_path="yolo_dataset"
            ),
            outputs="map50_score",
            name="evaluate_model_node"
        ),
    ])
