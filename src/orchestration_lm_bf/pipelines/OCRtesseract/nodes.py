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
            boxes = result.boxes.xyxy.cpu().numpy()
            scores = result.boxes.conf.cpu().numpy()
            classes = result.boxes.cls.cpu().numpy()
            for box, score, cls in zip(boxes, scores, classes):
                detection = {
                    "image_path": str(img_path),
                    "boxes": box.tolist(),  # [x1, y1, x2, y2]
                    "score": float(score),
                    "class": int(cls)
                }
                detections.append(detection)
    
    return detections

def extract_text(detections) -> list:
    reader = easyocr.Reader(['en'])
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
