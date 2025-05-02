from kedro.pipeline import Pipeline, node, pipeline
from .nodes import prepare_ocr_data, configure_tesseract, evaluate_ocr

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(configure_tesseract, inputs="params:ocr_language", outputs=None, name="configure_tesseract"),
        node(prepare_ocr_data, inputs=["yolo_data", "params:image_base_path"], outputs="ocr_images", name="prepare_ocr_data"),
        node(evaluate_ocr, inputs="ocr_images", outputs="ocr_cer", name="evaluate_ocr")
    ])