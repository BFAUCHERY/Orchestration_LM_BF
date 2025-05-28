# nodes.py
from PIL import Image
import cv2
import numpy as np
import os
from typing import List, Dict
import warnings
import time
import threading

# Désactiver les avertissements
warnings.filterwarnings("ignore", message=".*NNPACK.*")
warnings.filterwarnings("ignore")

# Désactiver Kaggle avant d'importer EasyOCR
os.environ['KAGGLE_CONFIG_DIR'] = '/tmp/kaggle_disabled'
os.environ['KAGGLE_USERNAME'] = ''
os.environ['KAGGLE_KEY'] = ''

import easyocr

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

class TimeoutOCR:
    """Classe pour gérer OCR avec timeout sans signal"""
    
    def __init__(self, timeout_seconds=20):
        self.timeout_seconds = timeout_seconds
        self.result = None
        self.exception = None
        self.completed = False
        
    def run_ocr(self, reader, crop):
        """Fonction qui exécute l'OCR"""
        try:
            print(f"[OCR-Thread] Starting OCR processing...")
            self.result = reader.readtext(crop)
            self.completed = True
            print(f"[OCR-Thread] OCR completed successfully")
        except Exception as e:
            print(f"[OCR-Thread] OCR failed with error: {e}")
            self.exception = e
            self.completed = True
    
    def extract_with_timeout(self, reader, crop):
        """Lance OCR avec timeout"""
        # Lancer OCR dans un thread séparé
        thread = threading.Thread(target=self.run_ocr, args=(reader, crop))
        thread.daemon = True
        thread.start()
        
        # Attendre avec timeout
        thread.join(timeout=self.timeout_seconds)
        
        if not self.completed:
            print(f"[OCR] ⏰ Timeout après {self.timeout_seconds}s")
            return None
        
        if self.exception:
            print(f"[OCR] ❌ Erreur OCR: {self.exception}")
            return None
            
        return self.result

def extract_text_from_crops(crops: List[np.ndarray]) -> List[Dict]:
    """Fonction principale avec gestion robuste des erreurs"""
    print(f"[OCR] Starting text extraction from {len(crops)} crops")
    
    # Désactiver explicitement Kaggle
    os.environ['KAGGLE_CONFIG_DIR'] = '/tmp/kaggle_disabled'
    os.environ['KAGGLE_USERNAME'] = ''
    os.environ['KAGGLE_KEY'] = ''
    
    # Initialiser EasyOCR avec gestion d'erreurs
    reader = None
    try:
        print("[OCR] Initializing EasyOCR...")
        # Forcer l'utilisation d'un répertoire spécifique pour EasyOCR
        reader = easyocr.Reader(
            ['en'], 
            gpu=False, 
            verbose=False,
            download_enabled=True
        )
        print("[OCR] EasyOCR initialized successfully")
    except Exception as e:
        print(f"[OCR] ❌ Failed to initialize EasyOCR: {e}")
        print(f"[OCR] Error type: {type(e).__name__}")
        print(f"[OCR] Error details: {str(e)}")
        
        # Essayer avec des paramètres différents
        try:
            print("[OCR] Trying alternative EasyOCR initialization...")
            reader = easyocr.Reader(['en'], gpu=False, verbose=False)
            print("[OCR] Alternative EasyOCR initialization successful")
        except Exception as e2:
            print(f"[OCR] ❌ Alternative initialization also failed: {e2}")
            # Retourner des résultats vides si EasyOCR ne peut pas s'initialiser
            return [[] for _ in crops]
    
    if reader is None:
        print("[OCR] ❌ Could not initialize EasyOCR reader")
        return [[] for _ in crops]
    
    results = []
    for idx, crop in enumerate(crops):
        try:
            print(f"[OCR] Processing crop {idx + 1}/{len(crops)}")
            
            # Sauvegarder le crop pour debug
            debug_path = f"data/07_predict/debug_crop_{idx}.png"
            cv2.imwrite(debug_path, crop)
            print(f"[OCR] Debug crop saved to: {debug_path}")
            
            # Utiliser la classe TimeoutOCR
            timeout_ocr = TimeoutOCR(timeout_seconds=20)
            text_results = timeout_ocr.extract_with_timeout(reader, crop)
            
            # Nettoyer le fichier debug
            try:
                os.remove(debug_path)
            except:
                pass
            
            if text_results is None:
                print(f"[OCR] No results for crop {idx + 1}")
                results.append([])
                continue
            
            # Traiter les résultats
            crop_texts = []
            for bbox, text, confidence in text_results:
                print(f"[OCR] ✅ Detected text: '{text}' with confidence {confidence:.3f}")
                crop_texts.append({
                    "text": text,
                    "confidence": float(confidence),
                    "bbox": [float(coord) for coord in np.array(bbox).flatten()]
                })
            
            results.append(crop_texts)
            print(f"[OCR] Crop {idx + 1} processed successfully with {len(crop_texts)} text detections")
            
        except Exception as e:
            print(f"[OCR] ❌ Critical error processing crop {idx + 1}: {e}")
            print(f"[OCR] Error type: {type(e).__name__}")
            import traceback
            traceback.print_exc()
            results.append([])
    
    print(f"[OCR] ✅ Completed processing {len(results)} crops")
    print(f"[OCR] Total text detections: {sum(len(r) for r in results)}")
    
    return results