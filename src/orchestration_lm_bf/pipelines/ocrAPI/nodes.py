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
    print("[PREP] Starting crop preparation from predictions")
    print(f"[PREP] Base folder: {base_folder}")
    print(f"[PREP] Total images to process: {len(predictions_dict.get('predictions_by_image', {}))}")
    crops = []
    for image_filename, data in predictions_dict.get("predictions_by_image", {}).items():
        image_path = os.path.join(base_folder, image_filename)
        print(f"Processing image: {image_path}")
        img = cv2.imread(image_path)
        print(f"[PREP] Image shape: {img.shape if img is not None else 'None'}")
        if img is None:
            print(f"Erreur lecture image: {image_path}")
            continue

        for i, pred in enumerate(data.get("predictions", [])):
            print(f"[PREP] Prediction {i}: {pred}")
            x = pred.get("x")
            y = pred.get("y")
            w = pred.get("width")
            h = pred.get("height")
            print(f"[PREP] Raw coords - x:{x}, y:{y}, w:{w}, h:{h}")

            if None in (x, y, w, h):
                print(f"[PREP] Invalid coordinates detected, skipping prediction {i}")
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

            print(f"[PREP] Appending crop #{len(crops)+1}")
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

    print(f"[PREP] Total crops prepared: {len(crops)}")
    return crops

def extract_text_from_crops(crops: List[np.ndarray]) -> List[Dict]:
    import threading
    import time
    
    print(f"[OCR] Starting text extraction from {len(crops)} crops")
    
    try:
        reader = easyocr.Reader(['en'], gpu=False, verbose=False)
        print("[OCR] EasyOCR reader created")
    except Exception as e:
        print(f"[OCR] Failed to create reader: {e}")
        return [[] for _ in crops]
    
    results = []
    
    for idx, crop in enumerate(crops):
        print(f"[OCR] Processing crop {idx + 1}/{len(crops)}")
        
        # Variables partagées pour le thread
        ocr_result = [None]
        ocr_error = [None]
        completed = [False]
        
        def run_ocr():
            try:
                print(f"[OCR] Thread started for crop {idx + 1}")
                ocr_result[0] = reader.readtext(crop)
                completed[0] = True
                print(f"[OCR] Thread completed for crop {idx + 1}")
            except Exception as e:
                print(f"[OCR] Thread error for crop {idx + 1}: {e}")
                ocr_error[0] = e
                completed[0] = True
        
        # Lancer OCR dans un thread avec timeout
        thread = threading.Thread(target=run_ocr)
        thread.daemon = True
        thread.start()
        
        # Attendre maximum 30 secondes
        timeout = 30
        start_time = time.time()
        
        while not completed[0] and (time.time() - start_time) < timeout:
            time.sleep(0.5)
            print(f"[OCR] Waiting... {time.time() - start_time:.1f}s")
        
        if not completed[0]:
            print(f"[OCR] ⏰ TIMEOUT après {timeout}s pour crop {idx + 1}")
            results.append([])
            continue
        
        if ocr_error[0]:
            print(f"[OCR] ❌ Error processing crop {idx + 1}: {ocr_error[0]}")
            results.append([])
            continue
        
        # Traiter les résultats
        text_results = ocr_result[0] or []
        crop_texts = []
        
        for bbox, text, confidence in text_results:
            print(f"[OCR] ✅ Detected text: '{text}' with confidence {confidence}")
            crop_texts.append({
                "text": text,
                "confidence": float(confidence),
                "bbox": [float(coord) for coord in np.array(bbox).flatten()]
            })
        
        results.append(crop_texts)
    
    print(f"[OCR] Completed processing {len(results)} crops")
    return results