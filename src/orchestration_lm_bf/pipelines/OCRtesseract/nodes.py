from PIL import Image
from pathlib import Path
import easyocr
import cv2
import numpy as np

def get_detections(model, images_folder: str):
    detections = []
    image_paths = list(Path(images_folder).glob("*.png"))  # ou png
    
    for img_path in image_paths:
        image = Image.open(img_path)
        results = model(image, conf=0.2)
        for result in results:
            boxes = result.boxes.xyxy.detach().numpy()
            scores = result.boxes.conf.detach().numpy()
            classes = result.boxes.cls.detach().numpy()
            for box, score, cls in zip(boxes, scores, classes):
                detection = {
                    "image_path": str(img_path),
                    "boxes": box.tolist(),  # [x1, y1, x2, y2]
                    "score": float(score),
                    "class": int(cls)
                }
                detections.append(detection)
    
    return detections

import os

def is_inside_docker():
    try:
        if os.path.exists("/.dockerenv"):
            return True
        with open("/proc/1/cgroup", "rt") as f:
            return any("docker" in line for line in f)
    except FileNotFoundError:
        return False

def extract_text(detections) -> list:
    if is_inside_docker():
        model_dir = Path("/home/kedro_docker/.easyocr")
    else:
        model_dir = Path("models/easyocr")

    model_dir.mkdir(parents=True, exist_ok=True)
    print(f"📁 Dossier des modèles EasyOCR: {model_dir.resolve()}")

    reader = easyocr.Reader(['en'], gpu=False, model_storage_directory=str(model_dir))
    results = []
    print(f"Number of detections: {len(detections)}")
    for detection in detections:
        img = cv2.imread(detection['image_path'])
        if img is None:
            print(f"Error reading image: {detection['image_path']}")
            continue
        x1, y1, x2, y2 = map(int, detection['boxes'])
        cropped = img[y1:y2, x1:x2]
        text_results = reader.readtext(cropped)
        clean_text = []
        for bbox, text, confidence in text_results:
            clean_bbox = [float(x) if isinstance(x, (np.floating, np.float32, np.float64)) else float(x) for x in np.array(bbox).flatten()]
            clean_confidence = float(confidence)
            clean_text.append({
                'bbox': clean_bbox,
                'text': text,
                'confidence': clean_confidence
            })
        
        results.append({
            'image_path': detection['image_path'],
            'text': clean_text
        })

    return results
