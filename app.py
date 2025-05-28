from pathlib import Path
from flask import Flask, request, jsonify, render_template
import os
from PIL import Image
import io
import base64
import random
from kedro.framework.session import KedroSession
from kedro.framework.startup import bootstrap_project
import json
import requests
import shutil
import sys
import time

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Initialisation du projet Kedro
bootstrap_project(Path.cwd())

# Ajouter le chemin src au PYTHONPATH pour importer les nodes
sys.path.append(str(Path.cwd() / "src"))

# Extensions autorisées
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'webp'}

# Configuration Roboflow (à adapter selon vos paramètres)
ROBOFLOW_CONFIG = {
    'api_key': 'XDn3ZffdMQJCAvokyyG1',  # À remplacer par votre vraie clé
    'project_id': 'graduation_project',
    'model_version': '1',
    'confidence': 0.5
}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def analyze_sign_local():
    """
    Analyse locale avec vos pipelines Kedro existants
    Basée sur l'ancien code qui fonctionnait
    """
    start_time = time.perf_counter()
    
    try:
        print("🖥️ Lancement de la pipeline evaluateYOLO...")
        
        # Classification avec YOLO (exactement comme dans l'ancien code)
        run_pipelines(["evaluateYOLO"])
        
        # Lire les résultats YOLO
        yolo_results_path = 'data/08_outputs/yolo_predictions.json'
        with open(yolo_results_path, 'r', encoding='utf-8') as f:
            data = json.load(f) 

        all_classes = []
        confidences = []
        for item in data:
            all_classes.extend(item.get('classes', [])) 
            confidences.extend(item.get('scores', []))
        
        if all_classes and confidences:
            category = all_classes[0]
            confidence = confidences[0]
        else:
            category = "Aucune classe trouvée."
            confidence = 0.0
        
        # Extraction du texte avec OCR
        detected_text = "Texte non disponible"
        confidence_ocr = 0.0
        try:
            print("🔍 Lancement OCRtesseract...")
            run_pipelines(["OCRtesseract"])
            
            ocr_path = 'data/08_outputs/ocr_output.json'
            with open(ocr_path, 'r', encoding='utf-8') as f:
                ocr_data = json.load(f)
                
            if ocr_data and len(ocr_data) > 0 and 'text' in ocr_data[0]:
                if len(ocr_data[0]['text']) > 0:
                    detected_text = ocr_data[0]['text'][0]['text']
                    confidence_ocr = ocr_data[0]['text'][0]["confidence"]
        except Exception as e:
            print(f"⚠️ OCR non disponible: {e}")
            # Continuer sans OCR
        
        processing_time_ms = int((time.perf_counter() - start_time) * 1000)
        
        return {
            'category': category,
            'category_name': f"Détection {category}",
            'category_description': f"Panneau détecté par YOLO: {category}",
            'detected_text': detected_text,
            'confidence_yolo': confidence,
            'confidence_ocr': confidence_ocr,
            'analysis_details': {
                'image_processed': True,
                'text_regions_found': 1 if detected_text != "Texte non disponible" else 0,
                'processing_time_ms': processing_time_ms
            }
        }
        
    except Exception as e:
        print(f"❌ Erreur lors de l'analyse locale: {e}")
        return {
            'category': "Erreur",
            'category_name': "Erreur d'analyse",
            'category_description': f"Erreur: {str(e)}",
            'detected_text': "N/A",
            'confidence_yolo': 0.0,
            'confidence_ocr': 0.0,
            'analysis_details': {
                'image_processed': False,
                'text_regions_found': 0,
                'processing_time_ms': 0
            }
        }

def analyze_sign_api_direct(image_path):
    """
    Analyse directe avec l'API Roboflow avec récupération de l'image annotée
    """
    try:
        print("🌐 Analyse directe via API Roboflow...")
        
        # URL de l'API
        url = f"https://detect.roboflow.com/{ROBOFLOW_CONFIG['project_id']}/{ROBOFLOW_CONFIG['model_version']}"
        
        # Paramètres
        params = {
            "api_key": ROBOFLOW_CONFIG['api_key'],
            "confidence": ROBOFLOW_CONFIG['confidence'],
            "overlap": 0.5,
            "format": "json",
        }
        
        # Headers
        headers = {
            "Content-Type": "application/x-www-form-urlencoded"
        }
        
        # Lire et encoder l'image
        with open(image_path, "rb") as img_file:
            image_data = base64.b64encode(img_file.read()).decode('utf-8')
        
        # Faire la première requête pour obtenir les prédictions JSON
        response = requests.post(url, params=params, data=image_data, headers=headers)
        response.raise_for_status()
        
        result = response.json()
        predictions = result.get("predictions", [])
        
        # Sauvegarder les prédictions pour la pipeline OCR
        roboflow_output_path = 'data/05_model_output/roboflow_predictions.json'
        os.makedirs(os.path.dirname(roboflow_output_path), exist_ok=True)
        
        # Formatter les prédictions pour la pipeline OCR - IMPORTANT: utiliser le bon nom de fichier
        image_filename = os.path.basename(image_path)  # Utiliser le vrai nom du fichier
        predictions_formatted = {
            "predictions_by_image": {
                image_filename: {  # Utiliser le nom réel du fichier
                    "predictions": predictions,
                    "image_info": result.get("image", {}),
                    "inference_time": result.get("time", 0),
                    "detection_count": len(predictions)
                }
            },
            "summary": {
                "total_images_processed": 1,
                "successful_predictions": 1,
                "total_detections": len(predictions)
            },
            "errors": []
        }
        
        with open(roboflow_output_path, 'w') as f:
            json.dump(predictions_formatted, f)
        
        # Faire une deuxième requête pour obtenir l'image annotée
        annotated_image_base64 = None
        if predictions:  # Seulement si on a des détections
            params_image = {
                "api_key": ROBOFLOW_CONFIG['api_key'],
                "confidence": ROBOFLOW_CONFIG['confidence'],
                "overlap": 0.5,
                "format": "image",  # Pour obtenir l'image annotée
                "labels": "on",     # Afficher les labels
                "stroke": "2"       # Épaisseur des boîtes
            }
            
            try:
                response_image = requests.post(url, params=params_image, data=image_data, headers=headers)
                response_image.raise_for_status()
                
                # L'image est retournée directement en base64
                annotated_image_base64 = base64.b64encode(response_image.content).decode('utf-8')
                print("✅ Image annotée récupérée")
            except Exception as e:
                print(f"⚠️ Impossible de récupérer l'image annotée: {e}")
                # Continuer sans l'image annotée
        
        # Afficher les résultats
        print(f"\n=== RÉSULTATS API Roboflow ===")
        print(f"Nombre de détections: {len(predictions)}")
        
        if predictions:
            for i, pred in enumerate(predictions, 1):
                print(f"\nDétection {i}:")
                print(f"  - Classe: {pred.get('class', 'inconnue')}")
                print(f"  - Confiance: {pred.get('confidence', 0)*100:.1f}%")
        
        return {
            'mode': 'api',
            'success': True,
            'predictions': predictions,
            'inference_time': result.get('time', 0),
            'total_detections': len(predictions),
            'annotated_image': annotated_image_base64,
            'summary': predictions_formatted['summary']
        }
        
    except Exception as e:
        print(f"❌ Erreur lors de l'analyse API: {e}")
        return {
            'mode': 'api',
            'success': False,
            'error': str(e),
            'predictions': []
        }

def analyze_sign_api(image_path):
    """
    Analyse avec l'API Roboflow + OCR via les pipelines Kedro
    """
    start_time = time.perf_counter()
    
    try:
        # Nettoyer les anciens fichiers de prédictions
        old_predictions_path = 'data/05_model_output/roboflow_predictions.json'
        if os.path.exists(old_predictions_path):
            os.remove(old_predictions_path)
            print(f"🗑️ Ancien fichier de prédictions supprimé")
        
        # Première étape : Analyse avec Roboflow
        analysis = analyze_sign_api_direct(image_path)
        
        if not analysis['success']:
            return analysis
        
        # Deuxième étape : OCR sur les détections
        detected_text = "Texte non disponible"
        confidence_ocr = 0.0
        
        try:
            print("🔍 Lancement pipeline OCR API...")
            
            # Vérifier que le fichier de prédictions existe
            if not os.path.exists(old_predictions_path):
                print(f"❌ Fichier de prédictions non trouvé: {old_predictions_path}")
                raise FileNotFoundError("Fichier de prédictions non trouvé")
            
            # Lancer la pipeline OCR qui utilise les prédictions Roboflow
            run_pipelines(["ocrAPI"])
            
            # Lire les résultats OCR
            ocr_path = 'data/08_outputs/ocr_output.json'
            if os.path.exists(ocr_path):
                with open(ocr_path, 'r', encoding='utf-8') as f:
                    ocr_data = json.load(f)
                
                # Extraire le texte et la confiance
                all_texts = []
                all_confidences = []
                
                for crop_results in ocr_data:
                    if isinstance(crop_results, list):
                        for text_detection in crop_results:
                            if 'text' in text_detection and text_detection['text']:
                                all_texts.append(text_detection['text'])
                                all_confidences.append(text_detection.get('confidence', 0))
                
                if all_texts:
                    # Prendre le texte avec la meilleure confiance
                    best_idx = all_confidences.index(max(all_confidences))
                    detected_text = all_texts[best_idx]
                    confidence_ocr = all_confidences[best_idx]
                    print(f"✅ Texte détecté: '{detected_text}' (confiance: {confidence_ocr:.2f})")
            
        except Exception as e:
            print(f"⚠️ Erreur OCR: {e}")
            print(f"⚠️ L'OCR a échoué, mais l'analyse continue sans le texte")
            # Continuer sans OCR si erreur
        
        # Calculer le temps de traitement
        processing_time_ms = int((time.perf_counter() - start_time) * 1000)
        
        # Préparer la réponse finale
        predictions = analysis.get('predictions', [])
        
        # Trouver la meilleure prédiction pour la catégorie
        category = "Aucune détection"
        confidence_yolo = 0.0
        if predictions:
            best_pred = max(predictions, key=lambda x: x.get('confidence', 0))
            category = best_pred.get('class', 'unknown')
            confidence_yolo = best_pred.get('confidence', 0)
        
        return {
            'success': True,
            'mode': 'api',
            'predictions': predictions,
            'inference_time': analysis.get('inference_time', 0),
            'total_detections': len(predictions),
            'annotated_image': analysis.get('annotated_image'),
            'category': category,
            'detected_text': detected_text,
            'confidence_yolo': confidence_yolo,
            'confidence_ocr': confidence_ocr,
            'analysis_details': {
                'image_processed': True,
                'text_regions_found': 1 if detected_text != "Texte non disponible" else 0,
                'processing_time_ms': processing_time_ms
            }
        }
        
    except Exception as e:
        print(f"❌ Erreur lors de l'analyse API: {e}")
        return {
            'success': False,
            'mode': 'api',
            'error': str(e),
            'predictions': []
        }

def run_pipelines(pipelines):
    for pipeline_name in pipelines:
        print(f"Running pipeline: {pipeline_name}")
        with KedroSession.create(project_path=Path.cwd()) as session:
            session.run(pipeline_name=pipeline_name)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/health')
@app.route('/health')
def health_check():
    return jsonify({
        'status': 'healthy',
        'service': 'Traffic Sign Analyzer API',
        'version': '2.0.0',
        'modes': ['local', 'api']
    }), 200

@app.route('/predict', methods=['POST'])
def predict():
    print("\n🔍 Nouvelle requête reçue sur /predict")
    
    try:
        # Vérifier si un fichier a été envoyé
        if 'image' not in request.files:
            print("❌ Aucune image dans la requête")
            return jsonify({'success': False, 'error': 'Aucune image fournie'}), 400
        
        file = request.files['image']
        mode = request.form.get('mode', 'local')  # Par défaut mode local
        
        print(f"📁 Fichier reçu: {file.filename}")
        print(f"🔧 Mode sélectionné: {mode}")
        
        if file.filename == '':
            return jsonify({'success': False, 'error': 'Aucun fichier sélectionné'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({'success': False, 'error': 'Format de fichier non supporté'}), 400
        
        # Sauvegarder le fichier temporairement
        filename = "image_to_predict.png"
        project_root = os.path.dirname(os.path.abspath(__file__))
        data_folder = os.path.join(project_root, 'data/07_predict/')
        os.makedirs(data_folder, exist_ok=True)
        filepath = os.path.join(data_folder, filename)
        
        # Supprimer le fichier existant s'il existe
        if os.path.exists(filepath):
            os.remove(filepath)
        file.save(filepath)
        print(f"💾 Image sauvegardée temporairement: {filepath}")
        
        try:
            # Vérifier que c'est bien une image valide
            with Image.open(filepath) as img:
                # Convertir en RGB si nécessaire
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                
                # Redimensionner si trop grande
                max_size = (800, 600)
                img.thumbnail(max_size, Image.Resampling.LANCZOS)
                img.save(filepath, format='JPEG', quality=85)
                
                # Analyser selon le mode choisi
                if mode == 'api':
                    print(f"🌐 Analyse avec l'API Roboflow + OCR...")
                    analysis_result = analyze_sign_api(filepath)
                    
                    if analysis_result['success']:
                        # Encoder l'image originale pour l'affichage (si pas d'image annotée)
                        if not analysis_result.get('annotated_image'):
                            buffer = io.BytesIO()
                            img.save(buffer, format='JPEG', quality=85)
                            img_base64 = base64.b64encode(buffer.getvalue()).decode()
                            analysis_result['image_preview'] = f"data:image/jpeg;base64,{img_base64}"
                        
                        response = analysis_result
                        response['filename'] = file.filename
                        print(f"✅ Analyse API réussie")
                    else:
                        response = {
                            'success': False,
                            'mode': 'api',
                            'error': analysis_result.get('error', 'Erreur API Roboflow'),
                            'predictions': []
                        }
                        print(f"❌ Erreur API: {analysis_result.get('error')}")
                        
                else:
                    print(f"🖥️ Analyse avec le modèle local...")
                    analysis_result = analyze_sign_local()
                    
                    # Encoder l'image pour l'affichage
                    buffer = io.BytesIO()
                    img.save(buffer, format='JPEG', quality=85)
                    img_base64 = base64.b64encode(buffer.getvalue()).decode()
                    
                    response = {
                        'success': True,
                        'mode': 'local',
                        'result': analysis_result,
                        'image_preview': f"data:image/jpeg;base64,{img_base64}",
                        'filename': file.filename
                    }
                    print(f"✅ Analyse locale réussie")
                
                return jsonify(response)
                
        except Exception as e:
            print(f"❌ Erreur lors du traitement: {e}")
            import traceback
            traceback.print_exc()
            return jsonify({
                'success': False, 
                'error': f'Erreur lors du traitement de l\'image: {str(e)}'
            }), 400
        
        finally:
            # Ne pas supprimer le fichier immédiatement car les pipelines peuvent en avoir besoin
            # Le supprimer après un délai ou dans un processus de nettoyage séparé
            pass
    
    except Exception as e:
        print(f"❌ Erreur serveur: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False, 
            'error': f'Erreur serveur: {str(e)}'
        }), 500

@app.route('/api/analyze', methods=['POST'])
def api_analyze():
    """Endpoint de compatibilité - redirige vers /predict"""
    return predict()

@app.errorhandler(413)
def too_large(e):
    return jsonify({'success': False, 'error': 'Fichier trop volumineux (max 16MB)'}), 413

@app.errorhandler(404)
def not_found(e):
    return jsonify({'success': False, 'error': 'Endpoint non trouvé'}), 404

@app.errorhandler(500)
def internal_error(e):
    return jsonify({'success': False, 'error': 'Erreur interne du serveur'}), 500

if __name__ == '__main__':
    print("🚦 Démarrage de l'API d'analyse de panneaux de signalisation...")
    print("📍 Interface disponible sur: http://localhost:5001")
    print("🖥️  Mode local: Kedro pipelines")
    print("🌐 Mode API: Roboflow + OCR")
    
    # Vérifier la configuration Roboflow
    if ROBOFLOW_CONFIG['api_key'] == 'VOTRE_CLE_API':
        print("⚠️  ATTENTION: Veuillez configurer votre clé API Roboflow dans ROBOFLOW_CONFIG")
    else:
        print("✅ Clé API Roboflow configurée")
    
    # Créer les dossiers nécessaires
    os.makedirs('data/07_predict', exist_ok=True)
    os.makedirs('data/05_model_output', exist_ok=True)
    os.makedirs('data/08_outputs', exist_ok=True)
    os.makedirs('templates', exist_ok=True)
    
app.run(debug=True, host='0.0.0.0', port=5001)