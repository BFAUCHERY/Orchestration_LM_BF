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
    """OCR avec Tesseract - plus stable que EasyOCR dans Docker"""
    print(f"[OCR] Starting TESSERACT text extraction from {len(crops)} crops")
    
    if not crops:
        print("[OCR] No crops to process")
        return []
    
    results = []
    
    try:
        import pytesseract
        print("[OCR] ✅ Tesseract OCR loaded successfully")
        
        for idx, crop in enumerate(crops):
            try:
                print(f"[OCR] Processing crop {idx + 1}/{len(crops)} with Tesseract")
                
                # Convertir BGR (OpenCV) vers RGB (PIL)
                rgb_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(rgb_crop)
                
                # Préprocessing pour améliorer l'OCR
                # 1. Convertir en grayscale
                gray_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
                
                # 2. Améliorer le contraste
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
                enhanced = clahe.apply(gray_crop)
                
                # 3. Appliquer un seuil pour binariser l'image
                _, thresh_crop = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                
                # 4. Débruitage
                kernel = np.ones((2,2), np.uint8)
                cleaned = cv2.morphologyEx(thresh_crop, cv2.MORPH_CLOSE, kernel)
                
                # Convertir en PIL pour Tesseract
                thresh_pil = Image.fromarray(cleaned)
                
                # OCR avec différentes configurations Tesseract
                configs = [
                    '--psm 8 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789',  # Mot seul, lettres et chiffres
                    '--psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789',  # Ligne de texte
                    '--psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789',  # Bloc de texte uniforme
                    '--psm 13',  # Raw line
                    '--psm 8',   # Mot seul sans restriction
                    '--psm 10'   # Caractère seul
                ]
                
                best_text = ""
                best_conf = 0.0
                
                for config_idx, config in enumerate(configs):
                    try:
                        # Essayer sur l'image preprocessée
                        text = pytesseract.image_to_string(thresh_pil, config=config).strip()
                        
                        if text and len(text) >= len(best_text):
                            # Calculer une confiance basée sur la longueur et la configuration
                            conf = 0.9 - (config_idx * 0.1)  # Première config = meilleure confiance
                            if len(text) > 1:  # Bonus pour texte plus long
                                conf += 0.05
                            
                            best_text = text
                            best_conf = max(conf, 0.5)  # Minimum 50%
                            
                            print(f"[OCR] Config {config_idx}: '{text}' (conf: {conf:.2f})")
                            
                            # Si on trouve quelque chose de satisfaisant, on s'arrête
                            if len(text) >= 3:
                                break
                                
                    except Exception as e:
                        print(f"[OCR] Config {config_idx} failed: {e}")
                        continue
                
                # Si rien trouvé avec l'image preprocessée, essayer sur l'originale
                if not best_text:
                    try:
                        print(f"[OCR] Trying original image for crop {idx + 1}")
                        text = pytesseract.image_to_string(pil_img, config='--psm 8').strip()
                        if text:
                            best_text = text
                            best_conf = 0.7
                            print(f"[OCR] Original image: '{text}'")
                    except Exception as e:
                        print(f"[OCR] Original image OCR failed: {e}")
                
                if best_text:
                    # Nettoyer le texte (supprimer caractères indésirables)
                    cleaned_text = ''.join(c for c in best_text if c.isalnum() or c.isspace()).strip()
                    
                    if cleaned_text:
                        print(f"[OCR] ✅ Tesseract found: '{cleaned_text}' (confidence: {best_conf:.2f})")
                        results.append([{
                            "text": cleaned_text,
                            "confidence": best_conf,
                            "bbox": [0, 0, crop.shape[1], 0, crop.shape[1], crop.shape[0], 0, crop.shape[0]]
                        }])
                    else:
                        print(f"[OCR] Text found but empty after cleaning: '{best_text}'")
                        results.append([])
                else:
                    print(f"[OCR] No text found in crop {idx + 1}")
                    results.append([])
                    
            except Exception as e:
                print(f"[OCR] ❌ Error processing crop {idx + 1}: {e}")
                import traceback
                traceback.print_exc()
                results.append([])
                
    except ImportError:
        print("[OCR] ❌ Tesseract not available (pytesseract not installed)")
        results = [[] for _ in crops]
    except Exception as e:
        print(f"[OCR] ❌ Tesseract failed: {e}")
        import traceback
        traceback.print_exc()
        results = [[] for _ in crops]
    
    total_detections = sum(len(r) for r in results)
    print(f"[OCR] ✅ Completed processing {len(results)} crops with {total_detections} text detections")
    
    return results