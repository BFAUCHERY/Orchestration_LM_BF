from kedro.pipeline import Pipeline, node, pipeline
from .nodes import prepare_crops_from_roboflow, extract_text_from_crops

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=prepare_crops_from_roboflow,
            inputs=dict(predictions_dict="roboflow_predictions_raw", base_folder="params:images_folder"),
            outputs="ocr_crops",
            name="prepare_crops_from_roboflow_node"
        ),
        node(
            func=extract_text_from_crops,
            inputs="ocr_crops",
            outputs="ocr_output",
            name="extract_text_from_crops_node"
        )
    ])