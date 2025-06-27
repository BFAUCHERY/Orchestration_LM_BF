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

def clean_text(raw_text):
    corrections = {
        "CUP": "STOP",
        "5TOP": "STOP",
        "ST0P": "STOP",
        "SIOP": "STOP",
        "STOPP": "STOP",
    }
    text = corrections.get(raw_text, raw_text)
    if text.isdigit():
        return text
    digits = ''.join(filter(str.isdigit, text))
    if digits:
        return digits
    return text

def extract_text(detections) -> list:
    print("🔍 Début de l'extraction de texte avec Tesseract")
    import pytesseract
    results = []
    for detection in detections:
        img = cv2.imread(detection['image_path'])
        if img is None:
            print(f"Error reading image: {detection['image_path']}")
            continue
        print(f"📷 Image chargée: {detection['image_path']}")
        x1, y1, x2, y2 = map(int, detection['boxes'])
        cropped = img[y1:y2, x1:x2]
        print(f"✂️  Image rognée aux coordonnées: {(x1, y1, x2, y2)}")

        # Pré-traitement pour améliorer l'OCR
        gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
        thresh = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 11, 2
        )

        # Tesseract config
        custom_config = r'--oem 3 --psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
        text = pytesseract.image_to_string(thresh, config=custom_config).strip().upper()
        print(f"📝 Texte détecté (Tesseract): {text}")
        cleaned_text = clean_text(text)
        print(f"🧹 Texte nettoyé : {cleaned_text}")

        # Confiance en fonction du texte
        confidence = 0.0
        if cleaned_text == "STOP":
            confidence = 0.95
        elif cleaned_text in {"30", "50", "70", "90", "110", "130"}:
            confidence = 0.90
        elif cleaned_text.isdigit() and len(cleaned_text) <= 3:
            confidence = 0.85
        elif any(char.isdigit() for char in cleaned_text):
            confidence = 0.75
        else:
            confidence = 0.50 if cleaned_text else 0.0

        results.append({
            'image_path': detection['image_path'],
            'text': [{
                'bbox': [float(x1), float(y1), float(x2), float(y2)],
                'text': cleaned_text,
                'confidence': confidence
            }]
        })
    print("✅ Fin de l'extraction de texte avec Tesseract.")
    return results
