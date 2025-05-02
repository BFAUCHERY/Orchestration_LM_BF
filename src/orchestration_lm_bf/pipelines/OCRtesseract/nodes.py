import pytesseract
from PIL import Image
from typing import List
from pathlib import Path

def prepare_ocr_data(yolo_data: list, base_path: str) -> List[Image.Image]:
    """
    Charge les images depuis les chemins fournis et retourne une liste de ROI simulés.
    """
    images = []
    for path, _ in yolo_data:
        img = Image.open(Path(base_path) / path)
        images.append(img)  # En pratique, ici il faudrait extraire les ROI
    return images

def configure_tesseract(language: str = "eng") -> None:
    pytesseract.pytesseract.tesseract_cmd = "/usr/bin/tesseract"  # Adapter selon env
    return

def evaluate_ocr(images: List[Image.Image]) -> float:
    """
    Simule une lecture OCR sur des images.
    """
    results = [pytesseract.image_to_string(img) for img in images]
    print("Sample OCR results:", results[:3])
    return 0.12  # CER fictif
