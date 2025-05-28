# nodes.py
from PIL import Image
import cv2
import numpy as np
import easyocr
import os
from typing import List, Dict
import warnings
import time
import threading
from concurrent.futures import ThreadPoolExecutor, TimeoutError

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

class TimeoutOCR:
    """Classe pour gérer OCR avec timeout sans signal"""
    
    def __init__(self, timeout_seconds=15):
        self.timeout_seconds = timeout_seconds
        self.result = None
        self.exception = None
        
    def run_ocr(self, reader, crop):
        """Fonction qui exécute l'OCR"""
        try:
            self.result = reader.readtext(crop)
        except Exception as e:
            self.exception = e
    
    def extract_with_timeout(self, reader, crop):
        """Lance OCR avec timeout"""
        # Lancer OCR dans un thread séparé
        thread = threading.Thread(target=self.run_ocr, args=(reader, crop))
        thread.daemon = True  # Thread daemon pour qu'il se ferme avec le programme
        thread.start()
        
        # Attendre avec timeout
        thread.join(timeout=self.timeout_seconds)
        
        if thread.is_alive():
            # Thread encore actif = timeout
            print(f"[OCR] ⏰ Timeout après {self.timeout_seconds}s")
            return None
        
        if self.exception:
            print(f"[OCR] ❌ Erreur OCR: {self.exception}")
            return None
            
        return self.result

def extract_text_from_crops_simple(crops: List[np.ndarray]) -> List[Dict]:
    """Version simplifiée et robuste pour Docker/Kedro"""
    print(f"[OCR] Starting simple text extraction from {len(crops)} crops")
    
    # Initialiser EasyOCR une seule fois
    try:
        print("[OCR] Initializing EasyOCR...")
        reader = easyocr.Reader(['en'], gpu=False, verbose=False)
        print("[OCR] EasyOCR initialized successfully")
    except Exception as e:
        print(f"[OCR] ❌ Failed to initialize EasyOCR: {e}")
        # Retourner des résultats vides si EasyOCR ne peut pas s'initialiser
        return [[] for _ in crops]
    
    results = []
    for idx, crop in enumerate(crops):
        try:
            print(f"[OCR] Processing crop {idx + 1}/{len(crops)}")
            
            # Utiliser la classe TimeoutOCR
            timeout_ocr = TimeoutOCR(timeout_seconds=15)
            text_results = timeout_ocr.extract_with_timeout(reader, crop)
            
            if text_results is None:
                # Timeout ou erreur
                results.append([])
                continue
            
            # Traiter les résultats
            crop_texts = []
            for bbox, text, confidence in text_results:
                print(f"[OCR] Detected text: '{text}' with confidence {confidence:.3f}")
                crop_texts.append({
                    "text": text,
                    "confidence": float(confidence),
                    "bbox": [float(coord) for coord in np.array(bbox).flatten()]
                })
            
            results.append(crop_texts)
            
        except Exception as e:
            print(f"[OCR] ❌ Critical error processing crop {idx + 1}: {e}")
            results.append([])
    
    print(f"[OCR] Completed processing {len(results)} crops")
    return results

def extract_text_from_crops_fallback(crops: List[np.ndarray]) -> List[Dict]:
    """Version ultra-simple sans timeout pour éviter tout problème"""
    print(f"[OCR] Starting fallback text extraction from {len(crops)} crops")
    
    try:
        reader = easyocr.Reader(['en'], gpu=False, verbose=False)
        print("[OCR] EasyOCR reader created")
    except Exception as e:
        print(f"[OCR] ❌ EasyOCR initialization failed: {e}")
        return [[] for _ in crops]
    
    results = []
    for idx, crop in enumerate(crops):
        try:
            print(f"[OCR] Processing crop {idx + 1}/{len(crops)} (fallback mode)")
            
            # OCR direct sans timeout - si ça bloque, ça bloque mais au moins ça essaie
            text_results = reader.readtext(crop)
            
            crop_texts = []
            for bbox, text, confidence in text_results:
                print(f"[OCR] Found: '{text}' (conf: {confidence:.3f})")
                crop_texts.append({
                    "text": text,
                    "confidence": float(confidence),
                    "bbox": [float(coord) for coord in np.array(bbox).flatten()]
                })
            
            results.append(crop_texts)
            
        except Exception as e:
            print(f"[OCR] ❌ Error processing crop {idx + 1}: {e}")
            results.append([])
    
    return results

def extract_text_from_crops(crops: List[np.ndarray]) -> List[Dict]:
    """Fonction principale avec plusieurs fallbacks"""
    print(f"[OCR] Starting text extraction from {len(crops)} crops")
    
    # Détecter l'environnement
    is_docker = os.environ.get('IN_DOCKER', False) or os.path.exists('/.dockerenv')
    
    if is_docker:
        print("[OCR] Docker detected, using simple extraction")
        
        # Essayer d'abord la version avec timeout
        try:
            return extract_text_from_crops_simple(crops)
        except Exception as e:
            print(f"[OCR] ❌ Simple extraction failed: {e}")
            print("[OCR] Trying fallback extraction...")
            
            # Si ça échoue, essayer la version ultra-simple
            try:
                return extract_text_from_crops_fallback(crops)
            except Exception as e:
                print(f"[OCR] ❌ Fallback extraction also failed: {e}")
                # Dernier recours : résultats vides
                return [[] for _ in crops]
    else:
        # En local, utiliser la version simple
        return extract_text_from_crops_simple(crops)