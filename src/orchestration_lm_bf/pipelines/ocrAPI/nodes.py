# nodes.py
from PIL import Image
import cv2
import numpy as np
import os
from typing import List, Dict
import warnings

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

def preprocess_for_sign_ocr(crop):
    """Preprocessing spécialisé pour panneaux de signalisation"""
    
    # 1. Convertir en grayscale
    if len(crop.shape) == 3:
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    else:
        gray = crop.copy()
    
    # 2. Redimensionner pour améliorer la résolution (important pour Tesseract)
    height, width = gray.shape
    if height < 100 or width < 100:  # Si trop petit, agrandir
        scale_factor = max(150 / height, 150 / width)
        new_height = int(height * scale_factor)
        new_width = int(width * scale_factor)
        gray = cv2.resize(gray, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
        print(f"[OCR] Resized from {width}x{height} to {new_width}x{new_height}")
    
    # 3. Débruitage
    denoised = cv2.medianBlur(gray, 3)
    
    # 4. Amélioration du contraste adaptatif
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    enhanced = clahe.apply(denoised)
    
    # 5. Détection des contours pour isoler le texte
    # Utiliser différentes approches selon le type de panneau
    
    # Approche 1: Seuillage adaptatif (bon pour panneaux avec fond uniforme)
    adaptive_thresh = cv2.adaptiveThreshold(
        enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
    )
    
    # Approche 2: Seuillage d'Otsu (bon pour panneaux contrastés)
    _, otsu_thresh = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Approche 3: Seuillage pour texte blanc sur fond coloré
    _, inv_thresh = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # 6. Morphologie pour nettoyer
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    
    adaptive_clean = cv2.morphologyEx(adaptive_thresh, cv2.MORPH_CLOSE, kernel)
    otsu_clean = cv2.morphologyEx(otsu_thresh, cv2.MORPH_CLOSE, kernel)
    inv_clean = cv2.morphologyEx(inv_thresh, cv2.MORPH_CLOSE, kernel)
    
    return {
        'original': gray,
        'adaptive': adaptive_clean,
        'otsu': otsu_clean,
        'inverted': inv_clean,
        'enhanced': enhanced
    }

def extract_text_from_crops(crops: List[np.ndarray]) -> List[Dict]:
    """OCR avec Tesseract optimisé pour panneaux de signalisation"""
    print(f"[OCR] Starting OPTIMIZED TESSERACT text extraction from {len(crops)} crops")
    
    if not crops:
        print("[OCR] No crops to process")
        return []
    
    results = []
    
    try:
        import pytesseract
        print("[OCR] ✅ Tesseract OCR loaded successfully")
        
        for idx, crop in enumerate(crops):
            try:
                print(f"[OCR] Processing crop {idx + 1}/{len(crops)} with advanced preprocessing")
                
                # Préprocessing spécialisé
                processed_images = preprocess_for_sign_ocr(crop)
                
                # Configurations Tesseract optimisées pour panneaux
                configs = [
                    # Configuration 1: Texte de panneau standard (STOP, YIELD, etc.)
                    {
                        'config': '--psm 8 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ ',
                        'description': 'Mot seul - lettres majuscules seulement'
                    },
                    # Configuration 2: Panneaux avec chiffres (vitesse, distances)
                    {
                        'config': '--psm 8 -c tessedit_char_whitelist=0123456789KMHMPHABCDEFGHIJKLMNOPQRSTUVWXYZ ',
                        'description': 'Mot seul - lettres et chiffres'
                    },
                    # Configuration 3: Ligne de texte simple
                    {
                        'config': '--psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 ',
                        'description': 'Ligne de texte'
                    },
                    # Configuration 4: Mode caractère par caractère pour texte difficile
                    {
                        'config': '--psm 10 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ',
                        'description': 'Caractère seul'
                    },
                    # Configuration 5: Mode permissif pour capturer tout
                    {
                        'config': '--psm 8',
                        'description': 'Mot seul - tous caractères'
                    },
                    # Configuration 6: Bloc de texte pour panneaux complexes
                    {
                        'config': '--psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 ',
                        'description': 'Bloc de texte uniforme'
                    }
                ]
                
                best_results = []
                
                # Tester chaque configuration sur chaque image préprocessée
                for img_name, img in processed_images.items():
                    pil_img = Image.fromarray(img)
                    
                    for config_data in configs:
                        try:
                            text = pytesseract.image_to_string(
                                pil_img, 
                                config=config_data['config']
                            ).strip()
                            
                            if text and len(text) > 0:
                                # Nettoyer le texte
                                cleaned_text = ''.join(c for c in text if c.isalnum() or c.isspace()).strip()
                                
                                if cleaned_text and len(cleaned_text) >= 2:  # Au moins 2 caractères
                                    # Calculer un score de qualité
                                    quality_score = calculate_text_quality(cleaned_text, img_name, config_data['description'])
                                    
                                    best_results.append({
                                        'text': cleaned_text,
                                        'confidence': quality_score,
                                        'method': f"{img_name}+{config_data['description']}",
                                        'bbox': [0, 0, crop.shape[1], 0, crop.shape[1], crop.shape[0], 0, crop.shape[0]]
                                    })
                                    
                                    print(f"[OCR] {img_name} + {config_data['description']}: '{cleaned_text}' (score: {quality_score:.2f})")
                                    
                        except Exception as e:
                            continue
                
                # Sélectionner le meilleur résultat
                if best_results:
                    # Trier par confiance décroissante
                    best_results.sort(key=lambda x: x['confidence'], reverse=True)
                    best_result = best_results[0]
                    
                    print(f"[OCR] ✅ BEST: '{best_result['text']}' (conf: {best_result['confidence']:.2f}, method: {best_result['method']})")
                    results.append([best_result])
                else:
                    print(f"[OCR] ❌ No text found in crop {idx + 1}")
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

def calculate_text_quality(text, preprocessing_method, config_description):
    """Calcule un score de qualité pour le texte détecté"""
    score = 0.5  # Score de base
    
    # Bonus pour longueur appropriée
    if 1 <= len(text) <= 10:
        score += 0.2
    elif len(text) > 10:
        score -= 0.1
    
    # Bonus pour mots de panneaux courants
    common_sign_words = [
        'STOP', 'YIELD', 'SPEED', 'LIMIT', 'ZONE', 'SCHOOL', 'PEDESTRIAN',
        'CROSSING', 'DANGER', 'WARNING', 'CAUTION', 'EXIT', 'ENTER',
        'ONE', 'WAY', 'DO', 'NOT', 'TURN', 'LEFT', 'RIGHT', 'AHEAD'
    ]
    
    # Bonus pour vitesses courantes (panneaux de limitation)
    common_speeds = ['20', '30', '40', '50', '60', '70', '80', '90', '100', '110', '130']
    
    if text.upper() in common_sign_words:
        score += 0.3
        print(f"[OCR] Bonus for common sign word: {text}")
    elif text in common_speeds:
        score += 0.4  # Bonus encore plus élevé pour les vitesses
        print(f"[OCR] Bonus for speed limit: {text}")
    
    # Bonus pour chiffres seuls (panneaux de vitesse)
    if text.isdigit():
        digit_value = int(text)
        if 20 <= digit_value <= 130 and digit_value % 10 == 0:  # Vitesses réalistes
            score += 0.3
            print(f"[OCR] Bonus for realistic speed: {text}")
        elif 1 <= digit_value <= 99:  # Autres chiffres valides
            score += 0.2
    
    # Bonus pour texte tout en majuscules (typique des panneaux)
    if text.isupper() and len(text) > 1:
        score += 0.1
    
    # Bonus selon la méthode de preprocessing
    if preprocessing_method == 'adaptive':
        score += 0.05
    elif preprocessing_method == 'enhanced':
        score += 0.03
    elif preprocessing_method == 'inverted':  # Bon pour panneaux de vitesse
        score += 0.04
    
    # Malus pour caractères étranges
    strange_chars = sum(1 for c in text if not (c.isalnum() or c.isspace()))
    score -= strange_chars * 0.1
    
    return min(1.0, max(0.1, score))  # Entre 0.1 et 1.0