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
    print("🔍 Début de l'extraction de texte avec EasyOCR")
    if is_inside_docker():
        model_dir = Path("/home/kedro_docker/.easyocr")
    else:
        model_dir = Path("models/easyocr")
    print(f"📂 Utilisation du dossier modèle: {model_dir}")

    model_dir.mkdir(parents=True, exist_ok=True)
    print("✅ Dossier modèle créé ou déjà existant.")
    print(f"📁 Dossier des modèles EasyOCR: {model_dir.resolve()}")
    print(f"📁 Contenu du dossier modèle EasyOCR ({model_dir.resolve()}):")
    for file in model_dir.glob("**/*"):
        print(f"  - {file.relative_to(model_dir)}")

    print("🔧 Initialisation du lecteur EasyOCR...")
    reader = easyocr.Reader(['en'], gpu=False, model_storage_directory=str(model_dir))
    print("✅ EasyOCR prêt.")
    print(f"🔍 Début du traitement de {len(detections)} détection(s)")
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
        text_results = reader.readtext(cropped)
        print(f"📝 Texte détecté: {text_results}")
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
    print("✅ Fin de l'extraction de texte.")
    return results
