# nodes.py
from PIL import Image
import cv2
import numpy as np
import os
from typing import List, Dict
import warnings

# Supprimer tous les warnings
warnings.filterwarnings("ignore")

def prepare_crops_from_roboflow(predictions_dict: dict, base_folder: str) -> List[np.ndarray]:
    print("[PREP] Starting crop preparation from predictions")
    print(f"[PREP] Base folder: {base_folder}")
    
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

            # Découper l'image avec padding
            padding = 10
            x1 = max(int(x - w / 2) - padding, 0)
            y1 = max(int(y - h / 2) - padding, 0)
            x2 = min(int(x + w / 2) + padding, img.shape[1])
            y2 = min(int(y + h / 2) + padding, img.shape[0])
            
            cropped = img[y1:y2, x1:x2]
            if cropped.size == 0:
                continue

            print(f"Crop {i}: dimensions {cropped.shape}")
            crops.append(cropped)

        # Supprimer l'image traitée
        try:
            os.remove(image_path)
        except:
            pass

    print(f"[PREP] Total crops: {len(crops)}")
    return crops

def extract_text_from_crops(crops: List[np.ndarray]) -> List[Dict]:
    """Version ultra-simple sans complications"""
    print(f"[OCR] Starting simple OCR on {len(crops)} crops")
    
    if not crops:
        print("[OCR] No crops to process")
        return []
    
    results = []
    
    # Essayer d'abord EasyOCR
    try:
        print("[OCR] Attempting EasyOCR...")
        import easyocr
        
        # Initialisation simple
        reader = easyocr.Reader(['en'], gpu=False)
        print("[OCR] EasyOCR reader created")
        
        for idx, crop in enumerate(crops):
            try:
                print(f"[OCR] Processing crop {idx + 1}/{len(crops)}")
                
                # OCR simple et direct
                ocr_results = reader.readtext(crop)
                
                crop_texts = []
                for (bbox, text, conf) in ocr_results:
                    if conf > 0.5 and text.strip():  # Seulement si confiance > 50%
                        print(f"[OCR] Found: '{text}' (confidence: {conf:.2f})")
                        crop_texts.append({
                            "text": text.strip(),
                            "confidence": float(conf),
                            "bbox": [float(x) for x in np.array(bbox).flatten()]
                        })
                
                results.append(crop_texts)
                
            except Exception as e:
                print(f"[OCR] Error on crop {idx}: {e}")
                results.append([])
                
    except Exception as e:
        print(f"[OCR] EasyOCR failed: {e}")
        
        # Fallback: Tesseract si EasyOCR échoue
        try:
            print("[OCR] Trying Tesseract fallback...")
            import pytesseract
            
            for idx, crop in enumerate(crops):
                try:
                    # Convertir en PIL Image
                    pil_img = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
                    
                    # OCR avec Tesseract
                    text = pytesseract.image_to_string(pil_img, config='--psm 8').strip()
                    
                    if text:
                        print(f"[OCR-Tesseract] Found: '{text}'")
                        results.append([{
                            "text": text,
                            "confidence": 0.8,  # Confiance arbitraire
                            "bbox": [0, 0, crop.shape[1], 0, crop.shape[1], crop.shape[0], 0, crop.shape[0]]
                        }])
                    else:
                        results.append([])
                        
                except Exception as e:
                    print(f"[OCR-Tesseract] Error on crop {idx}: {e}")
                    results.append([])
                    
        except Exception as e:
            print(f"[OCR] Tesseract also failed: {e}")
            # Dernier recours: résultats vides
            results = [[] for _ in crops]
    
    print(f"[OCR] Completed. Found {sum(len(r) for r in results)} text elements")
    return results