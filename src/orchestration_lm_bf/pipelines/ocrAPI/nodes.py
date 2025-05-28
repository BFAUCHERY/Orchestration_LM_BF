# nodes.py
from PIL import Image
import cv2
import numpy as np
import easyocr
import os
from typing import List, Dict
import warnings
warnings.filterwarnings("ignore", message=".*NNPACK.*")

def prepare_crops_from_roboflow(predictions_dict: dict, base_folder: str) -> List[np.ndarray]:
    crops = []
    for image_filename, data in predictions_dict.get("predictions_by_image", {}).items():
        image_path = os.path.join(base_folder, image_filename)
        print(f"Processing image: {image_path}")
        img = cv2.imread(image_path)
        if img is None:
            print(f"Erreur lecture image: {image_path}")
            continue

        for i, pred in enumerate(data.get("predictions", [])):
            x = pred.get("x")
            y = pred.get("y")
            w = pred.get("width")
            h = pred.get("height")

            if None in (x, y, w, h):
                continue

            padding = 5
            x1 = max(int(x - w / 2) - padding, 0)
            y1 = max(int(y - h / 2) - padding, 0)
            x2 = int(x + w / 2) + padding
            y2 = int(y + h / 2) + padding

            print(f"Cropping coordinates with padding: ({x1},{y1}) to ({x2},{y2})")
            cropped = img[y1:y2, x1:x2]
            if cropped.size == 0:
                print(f"Image vide ou invalide: {image_filename}, box: ({x1},{y1},{x2},{y2})")
                continue

            print(f"Crop dimensions: {cropped.shape}")
            debug_crop_path = os.path.join("data/07_predict", f"crop_{i}_{image_filename}")
            cv2.imwrite(debug_crop_path, cropped)
            print(f"Saved debug crop to: {debug_crop_path}")

            crops.append(cropped)
            try:
                os.remove(debug_crop_path)
                print(f"Deleted crop image: {debug_crop_path}")
            except Exception as e:
                print(f"Error deleting crop image {debug_crop_path}: {e}")
        try:
            os.remove(image_path)
            print(f"Deleted processed image: {image_path}")
        except Exception as e:
            print(f"Error deleting image {image_path}: {e}")

    return crops

def extract_text_from_crops(crops: List[np.ndarray]) -> List[Dict]:
    print(f"Extracting text from {len(crops)} crops")
    reader = easyocr.Reader(['en'], gpu=False)
    results = []
    for crop in crops:
        text_results = reader.readtext(crop)
        crop_texts = []
        for bbox, text, confidence in text_results:
            print(f"Detected text: '{text}' with confidence {confidence}")
            crop_texts.append({
                "text": text,
                "confidence": float(confidence),
                "bbox": [float(coord) for coord in np.array(bbox).flatten()]
            })
        results.append(crop_texts)
    return results