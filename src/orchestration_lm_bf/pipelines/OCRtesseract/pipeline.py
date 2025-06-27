from kedro.pipeline import Pipeline, node, pipeline
from .nodes import extract_text, get_detections,clean_text

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=get_detections, 
            inputs=["model", "params:images_folder"], 
            outputs="detections", 
            name="get_detections"
            ),
        node(
            func=extract_text,
            inputs="detections",
            outputs="ocr_output",
            name="ocr_extraction_node"
        ),
        node(
            func=clean_text,
            inputs="ocr_clean",
            outputs="ocr_clean",
            name="clean_text_node"
        )
    ])