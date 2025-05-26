import pytesseract
from PIL import Image
from pathlib import Path
from jiwer import cer
import os

def configure_tesseract():
    tessdata_dir = "C:/Users/benoit.fauchery/AppData/Local/Programs/Tesseract-OCR/tessdata"
    os.environ['TESSDATA_PREFIX'] = tessdata_dir
    pytesseract.pytesseract.tesseract_cmd = 'C:/Users/benoit.fauchery/AppData/Local/Programs/Tesseract-OCR/tesseract.exe'

    return {'lang': 'eng', 'config': '--psm 6'}

def get_detections(model, images_folder: str):
    detections = []
    image_paths = list(Path(images_folder).glob("*.png"))  # ou png
    
    for img_path in image_paths:
        image = Image.open(img_path)
        results = model(image, conf=0)
        for result in results:
            boxes = result.boxes.xyxy.cpu().numpy()
            scores = result.boxes.conf.cpu().numpy()
            classes = result.boxes.cls.cpu().numpy()
            for box, score, cls in zip(boxes, scores, classes):
                detection = {
                    "image_path": str(img_path),
                    "bbox": box.tolist(),  # [x1, y1, x2, y2]
                    "score": float(score),
                    "class": int(cls)
                }
                detections.append(detection)
    
    return detections


def prepare_ocr_data(detections):
    crops = []
    for det in detections:
        img = Image.open(det['image_path'])
        bbox = det['bbox']  # (x1, y1, x2, y2)
        roi = img.crop(bbox)
        crops.append(roi)

    return crops

def evaluate_ocr(crops, tess_config, ground_truths=None):
    texts = []
    cer_scores = []
    for i, crop in enumerate(crops):
        if crop.width > 0 and crop.height > 0:
            text = pytesseract.image_to_string(crop, lang=tess_config['lang'], config=tess_config['config'])
            texts.append(text)

            if ground_truths:
                score = cer(ground_truths[i], text)
                cer_scores.append(score)

        else:
            print(f"Invalid crop size: {crop.width}x{crop.height}")
        
    average_cer = sum(cer_scores) / len(cer_scores) if cer_scores else None

    return {'texts': texts, 'average_cer': average_cer}
