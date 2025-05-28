# nodes.py
from PIL import Image
import cv2
import numpy as np
import easyocr
import os
from typing import List, Dict
import warnings
import signal
import time
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, TimeoutError

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

def ocr_worker(crop_data):
    """Fonction worker pour OCR dans un processus séparé"""
    try:
        import easyocr
        import numpy as np
        
        # Désérialiser le crop
        crop = np.frombuffer(crop_data['data'], dtype=crop_data['dtype']).reshape(crop_data['shape'])
        idx = crop_data['idx']
        
        print(f"[OCR-Worker] Processing crop {idx + 1}")
        
        # Initialiser EasyOCR dans le processus worker
        reader = easyocr.Reader(['en'], gpu=False, verbose=False)
        
        # Timeout alarm handler
        def timeout_handler(signum, frame):
            raise TimeoutError("OCR timeout")
        
        # Set timeout de 30 secondes
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(30)
        
        try:
            text_results = reader.readtext(crop)
            signal.alarm(0)  # Cancel timeout
            
            crop_texts = []
            for bbox, text, confidence in text_results:
                print(f"[OCR-Worker] Detected text: '{text}' with confidence {confidence}")
                crop_texts.append({
                    "text": text,
                    "confidence": float(confidence),
                    "bbox": [float(coord) for coord in np.array(bbox).flatten()]
                })
            return crop_texts
            
        except TimeoutError:
            print(f"[OCR-Worker] ⏰ Timeout pour crop {idx + 1}")
            return []
        except Exception as e:
            print(f"[OCR-Worker] ❌ Error processing crop {idx + 1}: {e}")
            return []
        finally:
            signal.alarm(0)  # Cancel any remaining alarm
            
    except Exception as e:
        print(f"[OCR-Worker] ❌ Critical error in worker: {e}")
        return []

def extract_text_from_crops_simple(crops: List[np.ndarray]) -> List[Dict]:
    """Version simplifiée sans multiprocessing pour Docker"""
    print(f"[OCR] Starting simple text extraction from {len(crops)} crops")
    
    # Fallback simple si problème avec EasyOCR
    try:
        # Test rapide d'EasyOCR
        reader = easyocr.Reader(['en'], gpu=False, verbose=False)
        
        # Test sur une petite image
        test_img = np.ones((50, 50, 3), dtype=np.uint8) * 255
        reader.readtext(test_img)
        print("[OCR] EasyOCR test passed")
        
    except Exception as e:
        print(f"[OCR] ❌ EasyOCR failed, using fallback: {e}")
        # Retourner des résultats vides en cas d'échec
        return [[] for _ in crops]
    
    results = []
    for idx, crop in enumerate(crops):
        try:
            print(f"[OCR] Processing crop {idx + 1}/{len(crops)}")
            
            # Timeout simple avec alarm
            def timeout_handler(signum, frame):
                raise TimeoutError("OCR timeout")
            
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(20)  # 20 secondes timeout
            
            try:
                text_results = reader.readtext(crop)
                signal.alarm(0)  # Cancel timeout
                
                crop_texts = []
                for bbox, text, confidence in text_results:
                    print(f"[OCR] Detected text: '{text}' with confidence {confidence}")
                    crop_texts.append({
                        "text": text,
                        "confidence": float(confidence),
                        "bbox": [float(coord) for coord in np.array(bbox).flatten()]
                    })
                results.append(crop_texts)
                
            except TimeoutError:
                print(f"[OCR] ⏰ Timeout pour crop {idx + 1}")
                results.append([])
            except Exception as e:
                print(f"[OCR] ❌ Error processing crop {idx + 1}: {e}")
                results.append([])
            finally:
                signal.alarm(0)
                
        except Exception as e:
            print(f"[OCR] ❌ Critical error processing crop {idx + 1}: {e}")
            results.append([])
    
    return results

def extract_text_from_crops(crops: List[np.ndarray]) -> List[Dict]:
    """Version principale avec multiprocessing et fallback"""
    print(f"[OCR] Starting text extraction from {len(crops)} crops")
    
    # Vérifier si on est dans Docker ou si multiprocessing pose problème
    is_docker = os.environ.get('IN_DOCKER', False) or os.path.exists('/.dockerenv')
    
    if is_docker:
        print("[OCR] Docker detected, using simple extraction")
        return extract_text_from_crops_simple(crops)
    
    # Essayer avec multiprocessing
    try:
        # Préparer les données pour les workers
        crop_data_list = []
        for idx, crop in enumerate(crops):
            crop_data = {
                'data': crop.tobytes(),
                'dtype': crop.dtype,
                'shape': crop.shape,
                'idx': idx
            }
            crop_data_list.append(crop_data)
        
        # Utiliser ProcessPoolExecutor avec timeout
        results = []
        with ProcessPoolExecutor(max_workers=2) as executor:
            try:
                # Timeout global de 60 secondes
                futures = [executor.submit(ocr_worker, crop_data) for crop_data in crop_data_list]
                
                for i, future in enumerate(futures):
                    try:
                        result = future.result(timeout=30)  # 30s par crop
                        results.append(result)
                        print(f"[OCR] Completed crop {i+1}/{len(crops)}")
                    except TimeoutError:
                        print(f"[OCR] ⏰ Timeout for crop {i+1}")
                        results.append([])
                    except Exception as e:
                        print(f"[OCR] ❌ Error for crop {i+1}: {e}")
                        results.append([])
                        
            except Exception as e:
                print(f"[OCR] ❌ ProcessPool error: {e}")
                return extract_text_from_crops_simple(crops)
                
        return results
        
    except Exception as e:
        print(f"[OCR] ❌ Multiprocessing failed, using simple fallback: {e}")
        return extract_text_from_crops_simple(crops)