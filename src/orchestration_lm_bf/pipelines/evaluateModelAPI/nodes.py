import requests
import base64
import os
from typing import Dict, Any, List, Union
import logging
from pathlib import Path
import shutil

logger = logging.getLogger(__name__)

def evaluate_model_api_node(
    image_source: Union[str, Dict[str, str]],
    api_key: str, 
    project_id: str,
    model_version: str,
    confidence: float
) -> Dict[str, Any]:
    """
    Évalue le modèle Roboflow sur un dossier d'images ou une image unique.
    
    Args:
        image_source: Soit un chemin vers un dossier, soit un dict avec le chemin d'une image unique
        api_key: Clé API Roboflow
        project_id: ID du projet (graduation_project)
        model_version: Version du modèle (1)
        confidence: Seuil de confiance
    
    Returns:
        Dict contenant tous les résultats de prédiction
    """
    results = {
        "predictions_by_image": {},
        "summary": {},
        "errors": []
    }
    
    # URL de l'API pour votre modèle spécifique
    url = f"https://detect.roboflow.com/{project_id}/{model_version}"
    
    # Paramètres de la requête
    params = {
        "api_key": api_key,
        "confidence": confidence,
        "overlap": 0.5
    }
    
    # Headers
    headers = {
        "Content-Type": "application/x-www-form-urlencoded"
    }
    
    # Déterminer si c'est un fichier unique ou un dossier
    image_files = []
    base_folder = ""
    
    if isinstance(image_source, dict) and 'single_image_path' in image_source:
        # Cas d'une image unique (depuis l'interface web)
        single_image_path = image_source['single_image_path']
        if os.path.exists(single_image_path):
            image_files = [os.path.basename(single_image_path)]
            base_folder = os.path.dirname(single_image_path)
            logger.info(f"Mode image unique: {single_image_path}")
        else:
            logger.error(f"Image non trouvée: {single_image_path}")
            results["errors"].append(f"Image non trouvée: {single_image_path}")
            return results
    else:
        # Cas d'un dossier (comportement original)
        image_folder = str(image_source)
        if not os.path.exists(image_folder):
            logger.error(f"Dossier non trouvé: {image_folder}")
            results["errors"].append(f"Dossier non trouvé: {image_folder}")
            return results
            
        base_folder = image_folder
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
        image_files = [f for f in os.listdir(image_folder) 
                      if any(f.lower().endswith(ext) for ext in image_extensions)]
        logger.info(f"Mode dossier: {len(image_files)} images dans {image_folder}")
    
    if not image_files:
        logger.warning("Aucune image à traiter")
        results["errors"].append("Aucune image trouvée")
        return results
    
    logger.info(f"Traitement de {len(image_files)} image(s)")
    
    total_detections = 0
    successful_predictions = 0
    
    for image_file in image_files:
        image_path = os.path.join(base_folder, image_file)

        # S'assurer que l'image est bien copiée dans data/07_predict pour l'étape d'OCR
        predict_dir = Path("data/07_predict")
        predict_dir.mkdir(parents=True, exist_ok=True)
        target_path = predict_dir / image_file
        if not target_path.exists():
            try:
                shutil.copy(image_path, target_path)
                logger.info(f"Image copiée dans {target_path}")
            except Exception as copy_err:
                logger.error(f"Erreur lors de la copie de l'image: {copy_err}")

        try:
            # Lire et encoder l'image en base64
            with open(image_path, "rb") as img_file:
                image_data = base64.b64encode(img_file.read()).decode('utf-8')
            
            # Faire la requête POST
            response = requests.post(url, params=params, data=image_data, headers=headers)
            response.raise_for_status()
            
            # Traiter la réponse
            result = response.json()
            predictions = result.get("predictions", [])
            
            # PRINTS pour afficher les résultats
            print(f"\n=== RÉSULTATS pour {image_file} ===")
            print(f"Nombre de détections: {len(predictions)}")
            
            if predictions:
                for i, pred in enumerate(predictions, 1):
                    print(f"\nDétection {i}:")
                    print(f"  - Classe: {pred.get('class', 'inconnue')}")
                    print(f"  - Confiance: {pred.get('confidence', 0):.3f} ({pred.get('confidence', 0)*100:.1f}%)")
                    print(f"  - Position X: {pred.get('x', 0):.1f}")
                    print(f"  - Position Y: {pred.get('y', 0):.1f}")
                    print(f"  - Largeur: {pred.get('width', 0):.1f}")
                    print(f"  - Hauteur: {pred.get('height', 0):.1f}")
                    print(f"  - ID détection: {pred.get('detection_id', 'N/A')}")
            else:
                print("  Aucune détection trouvée")
            
            print(f"Temps d'inférence: {result.get('time', 0):.3f}s")
            print("=" * 50)
            
            results["predictions_by_image"][image_file] = {
                "predictions": predictions,
                "image_info": result.get("image", {}),
                "inference_time": result.get("time", 0),
                "detection_count": len(predictions)
            }
            
            total_detections += len(predictions)
            successful_predictions += 1
            
            logger.info(f"Image {image_file}: {len(predictions)} détections")
            
        except requests.exceptions.RequestException as e:
            error_msg = f"Erreur API pour {image_file}: {e}"
            logger.error(error_msg)
            results["errors"].append(error_msg)
            
        except Exception as e:
            error_msg = f"Erreur inattendue pour {image_file}: {e}"
            logger.error(error_msg)
            results["errors"].append(error_msg)
    
    # Créer le résumé
    results["summary"] = {
        "total_images_processed": len(image_files),
        "successful_predictions": successful_predictions,
        "failed_predictions": len(image_files) - successful_predictions,
        "total_detections": total_detections,
        "average_detections_per_image": total_detections / successful_predictions if successful_predictions > 0 else 0,
        "error_count": len(results["errors"])
    }
    
    # PRINT du résumé final
    print(f"\n{'='*60}")
    print("RÉSUMÉ FINAL")
    print(f"{'='*60}")
    print(f"Images traitées: {results['summary']['successful_predictions']}/{results['summary']['total_images_processed']}")
    print(f"Total détections: {results['summary']['total_detections']}")
    print(f"Moyenne détections/image: {results['summary']['average_detections_per_image']:.2f}")
    if results["errors"]:
        print(f"Erreurs: {results['summary']['error_count']}")
    print(f"{'='*60}\n")
    
    logger.info(f"Traitement terminé. {successful_predictions}/{len(image_files)} images traitées avec succès")
    
    return results


def evaluate_single_image_api(
    image_path: str,
    api_key: str, 
    project_id: str,
    model_version: str,
    confidence: float
) -> Dict[str, Any]:
    """
    Version simplifiée pour évaluer une seule image via l'API Roboflow.
    Utilisée directement par l'interface web.
    
    Args:
        image_path: Chemin vers l'image unique
        api_key: Clé API Roboflow
        project_id: ID du projet 
        model_version: Version du modèle
        confidence: Seuil de confiance
    
    Returns:
        Dict contenant les résultats de prédiction pour cette image
    """
    # Utiliser la fonction principale avec le format dict pour une image unique
    return evaluate_model_api_node(
        image_source={'single_image_path': image_path},
        api_key=api_key,
        project_id=project_id,
        model_version=model_version,
        confidence=confidence
    )