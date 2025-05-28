# nodes.py
from PIL import Image
import cv2
import numpy as np
import os
from typing import List, Dict
import warnings

# Supprimer tous les warnings
warnings.filterwarnings("ignore")

# DÉSACTIVER KAGGLE COMPLÈTEMENT AVANT TOUT IMPORT
os.environ['KAGGLE_CONFIG_DIR'] = '/tmp'
os.environ['KAGGLE_USERNAME'] = 'disabled'
os.environ['KAGGLE_KEY'] = 'disabled'

# Créer un fichier kaggle.json vide pour éviter l'erreur
try:
    kaggle_dir = '/tmp'
    kaggle_file = os.path.join(kaggle_dir, 'kaggle.json')
    if not os.path.exists(kaggle_file):
        with open(kaggle_file, 'w') as f:
            f.write('{"username":"disabled","key":"disabled"}')
        os.chmod(kaggle_file, 0o600)
except:
    pass

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
    """Version avec fallback complet Tesseract"""
    print(f"[OCR] Starting OCR on {len(crops)} crops")
    
    if not crops:
        print("[OCR] No crops to process")
        return []
    
    results = []
    
    # Essayer Tesseract en premier (plus fiable dans Docker)
    try:
        print("[OCR] Attempting Tesseract OCR...")
        import pytesseract
        
        for idx, crop in enumerate(crops):
            try:
                print(f"[OCR] Processing crop {idx + 1}/{len(crops)} with Tesseract")
                
                # Convertir BGR (OpenCV) vers RGB (PIL)
                rgb_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(rgb_crop)
                
                # Préprocessing pour améliorer l'OCR
                # Convertir en grayscale et améliorer le contraste
                gray_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
                
                # Appliquer un seuil pour binariser l'image
                _, thresh_crop = cv2.threshold(gray_crop, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                
                # Convertir en PIL pour Tesseract
                thresh_pil = Image.fromarray(thresh_crop)
                
                # OCR avec différentes configurations
                configs = [
                    '--psm 8 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789',  # Lettres et chiffres seulement
                    '--psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789',  # Ligne de texte simple
                    '--psm 13',  # Raw line
                    '--psm 8'   # Single word
                ]
                
                best_text = ""
                best_conf = 0
                
                for config in configs:
                    try:
                        # Essayer sur l'image binarisée
                        text = pytesseract.image_to_string(thresh_pil, config=config).strip()
                        if text and len(text) > len(best_text):
                            best_text = text
                            best_conf = 0.8  # Confiance arbitraire pour Tesseract
                            break
                    except:
                        continue
                
                # Si pas de résultat, essayer sur l'image originale
                if not best_text:
                    try:
                        best_text = pytesseract.image_to_string(pil_img, config='--psm 8').strip()
                        best_conf = 0.7
                    except:
                        best_text = ""
                
                if best_text:
                    print(f"[OCR-Tesseract] Found: '{best_text}'")
                    results.append([{
                        "text": best_text,
                        "confidence": best_conf,
                        "bbox": [0, 0, crop.shape[1], 0, crop.shape[1], crop.shape[0], 0, crop.shape[0]]
                    }])
                else:
                    print(f"[OCR-Tesseract] No text found in crop {idx + 1}")
                    results.append([])
                    
            except Exception as e:
                print(f"[OCR-Tesseract] Error on crop {idx}: {e}")
                results.append([])
                
    except Exception as e:
        print(f"[OCR] Tesseract failed: {e}")
        
        # Dernier recours: EasyOCR avec désactivation Kaggle
        try:
            print("[OCR] Trying EasyOCR as fallback...")
            
            # Réessayer de désactiver Kaggle
            os.environ['KAGGLE_CONFIG_DIR'] = '/tmp'
            os.environ['KAGGLE_USERNAME'] = 'disabled'
            os.environ['KAGGLE_KEY'] = 'disabled'
            
            import easyocr
            
            # Forcer l'utilisation d'un répertoire local pour EasyOCR
            reader = easyocr.Reader(
                ['en'], 
                gpu=False, 
                verbose=False,
                download_enabled=False  # Désactiver les téléchargements
            )
            
            for idx, crop in enumerate(crops):
                try:
                    print(f"[OCR-EasyOCR] Processing crop {idx + 1}/{len(crops)}")
                    ocr_results = reader.readtext(crop)
                    
                    crop_texts = []
                    for (bbox, text, conf) in ocr_results:
                        if conf > 0.5 and text.strip():
                            print(f"[OCR-EasyOCR] Found: '{text}' (confidence: {conf:.2f})")
                            crop_texts.append({
                                "text": text.strip(),
                                "confidence": float(conf),
                                "bbox": [float(x) for x in np.array(bbox).flatten()]
                            })
                    
                    results.append(crop_texts)
                    
                except Exception as e:
                    print(f"[OCR-EasyOCR] Error on crop {idx}: {e}")
                    results.append([])
                    
        except Exception as e:
            print(f"[OCR] EasyOCR also failed: {e}")
            # Dernier recours: résultats vides
            results = [[] for _ in crops]
    
    print(f"[OCR] Completed. Found {sum(len(r) for r in results)} text elements")
    return results