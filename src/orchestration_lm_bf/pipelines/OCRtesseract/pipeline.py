from kedro.pipeline import Pipeline, node, pipeline
from .nodes import get_detections, prepare_ocr_data, configure_tesseract, evaluate_ocr

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=configure_tesseract, 
            inputs=None, 
            outputs="tess_config", 
            name="configure_tesseract"
            ),
        node(
            func=get_detections, 
            inputs=["model", "params:images_folder"], 
            outputs="detections", 
            name="get_detections"
            ),
        node(
            func=prepare_ocr_data, 
            inputs=["detections"], 
            outputs="crops", 
            name="prepare_ocr_data"
            ),
        node(
            func=evaluate_ocr, 
            inputs=["crops", "tess_config"], 
            outputs="ocr_evaluation", 
            name="evaluate_ocr"
            )
    ])