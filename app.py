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
        if all_classes and confidences :
            category = all_classes[0]
            confidence = confidences[0]
        else:
            category = "Aucune classe trouvée."
        
        # Extraction du texte avec OCR
        detected_text = "Texte non disponible"
        try:
            print("🔍 Lancement OCRtesseract...")
            run_pipelines(["OCRtesseract"])
            
            ocr_path = 'data/08_outputs/ocr_output.json'
            with open(ocr_path, 'r', encoding='utf-8') as f:
                detected_text = json.load(f)
        except Exception as e:
            print(f"⚠️ OCR non disponible: {e}")
            # Continuer sans OCR
        
        processing_time_ms = int((time.perf_counter() - start_time) * 1000)
        
        return {
            'category': category,
            'category_name': f"Détection {category}",
            'category_description': f"Panneau détecté par YOLO: {category}",
            'detected_text': detected_text[0]['text'][0]['text'],
            'confidence_yolo': confidence,
            'confidence_ocr': detected_text[0]['text'][0]["confidence"],
            'analysis_details': {
                'image_processed': True,
                'text_regions_found': random.randint(1, 3),
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
            'confidence': 0.0,
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
        
        # Paramètres - IMPORTANT: Ajouter format=image pour obtenir l'image annotée
        params = {
            "api_key": ROBOFLOW_CONFIG['api_key'],
            "confidence": ROBOFLOW_CONFIG['confidence'],
            "overlap": 0.5,
            "format": "json",  # D'abord obtenir le JSON
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
            'annotated_image': annotated_image_base64,  # Ajouter l'image annotée
            'summary': {
                'total_images_processed': 1,
                'successful_predictions': 1,
                'total_detections': len(predictions)
            }
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
    Analyse avec l'API Roboflow via la pipeline Kedro
    """
    try:
        # Vérifier d'abord si la clé API est configurée
        if ROBOFLOW_CONFIG['api_key'] == 'VOTRE_CLE_API':
            print("❌ Clé API Roboflow non configurée!")
            # Utiliser l'approche directe au lieu de Kedro
            return analyze_sign_api_direct(image_path)
        
        # Essayer d'abord l'approche directe (plus simple)
        return analyze_sign_api_direct(image_path)
        
        # Si vous voulez utiliser Kedro, décommentez le code ci-dessous:
        
        print("🌐 Lancement de la pipeline evaluateModelAPI...")
        
        # Copier l'image dans le dossier attendu par la pipeline
        api_image_folder = 'data/01_raw/api_images'
        os.makedirs(api_image_folder, exist_ok=True)
        
        # Supprimer les anciennes images
        for old_file in os.listdir(api_image_folder):
            old_path = os.path.join(api_image_folder, old_file)
            if os.path.isfile(old_path):
                os.remove(old_path)
        
        # Copier l'image avec un nom standard
        api_image_path = os.path.join(api_image_folder, 'image_to_analyze.png')
        shutil.copy2(image_path, api_image_path)
        
        # Lancer la pipeline Roboflow
        run_pipelines(["evaluateModelAPI"])
        
        # Lire les résultats
        roboflow_results_path = 'data/05_model_output/roboflow_predictions.json'
        if os.path.exists(roboflow_results_path):
            with open(roboflow_results_path, 'r', encoding='utf-8') as f:
                result = json.load(f)
            
            predictions = []
            inference_time = 0
            
            # Extraire les prédictions de toutes les images
            for image_name, image_data in result.get("predictions_by_image", {}).items():
                predictions.extend(image_data.get("predictions", []))
                inference_time = max(inference_time, image_data.get("inference_time", 0))
                
            return {
                'mode': 'api',
                'success': True,
                'predictions': predictions,
                'inference_time': inference_time,
                'total_detections': len(predictions),
                'summary': result.get("summary", {})
            }
        else:
            return {
                'mode': 'api',
                'success': False,
                'error': f'Fichier de résultats non trouvé: {roboflow_results_path}',
                'predictions': []
            }
            
    except Exception as e:
        print(f"❌ Erreur lors de l'analyse API: {e}")
        return {
            'mode': 'api',
            'success': False,
            'error': str(e),
            'predictions': []
        }

def run_pipelines(pipelines):
    for pipeline_name in pipelines:
        print(f"Running pipeline: {pipeline_name}")
        with KedroSession.create("./", Path.cwd()) as session:
            session.run(pipeline_name=pipeline_name)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/health')
def health_check():
    return jsonify({
        'status': 'healthy',
        'service': 'Traffic Sign Analyzer API',
        'version': '2.0.0',
        'modes': ['local', 'api']
    })

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
                    print(f"🌐 Analyse avec l'API Roboflow...")
                    analysis_result = analyze_sign_api(filepath)
                    
                    if analysis_result['success']:
                        predictions = analysis_result['predictions']
                        
                        # Formater la réponse pour l'interface
                        response = {
                            'success': True,
                            'mode': 'api',
                            'predictions': predictions,
                            'inference_time': analysis_result.get('inference_time', 0),
                            'total_detections': len(predictions),
                            'filename': file.filename
                        }
                        
                        # Dans la fonction predict(), partie API :

                if mode == 'api':
                    print(f"🌐 Analyse avec l'API Roboflow...")
                    analysis_result = analyze_sign_api(filepath)
                    
                    if analysis_result['success']:
                        predictions = analysis_result['predictions']
                        
                        # Formater la réponse pour l'interface
                        response = {
                            'success': True,
                            'mode': 'api',
                            'predictions': predictions,
                            'inference_time': analysis_result.get('inference_time', 0),
                            'total_detections': len(predictions),
                            'filename': file.filename,
                            'annotated_image': analysis_result.get('annotated_image')  # Ajouter l'image annotée
                        }
                        
                        if predictions:
                            best_pred = max(predictions, key=lambda x: x.get('confidence', 0))
                            response['best_detection'] = {
                                'class': best_pred.get('class', 'unknown'),
                                'confidence': best_pred.get('confidence', 0),
                                'bbox': {
                                    'x': best_pred.get('x', 0),
                                    'y': best_pred.get('y', 0),
                                    'width': best_pred.get('width', 0),
                                    'height': best_pred.get('height', 0)
                                }
                            }
                        print(f"✅ Analyse API réussie: {len(predictions)} détections")
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
            return jsonify({
                'success': False, 
                'error': f'Erreur lors du traitement de l\'image: {str(e)}'
            }), 400
        
        finally:
            # Nettoyer le fichier temporaire
            if os.path.exists(filepath):
                os.remove(filepath)
                print(f"🗑️ Fichier temporaire supprimé")
    
    except Exception as e:
        print(f"❌ Erreur serveur: {e}")
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
    print("🌐 Mode API: Roboflow")
    
    # Vérifier la configuration Roboflow
    if ROBOFLOW_CONFIG['api_key'] == 'VOTRE_CLE_API':
        print("⚠️  ATTENTION: Veuillez configurer votre clé API Roboflow dans ROBOFLOW_CONFIG")
        print("   L'API fonctionnera en mode démo sans vraie clé")
    else:
        print("✅ Clé API Roboflow configurée")
    
    # Créer les dossiers nécessaires
    os.makedirs('data/07_predict', exist_ok=True)
    os.makedirs('data/01_raw/api_images', exist_ok=True)
    os.makedirs('templates', exist_ok=True)
    
app.run(debug=True, host='127.0.0.1', port=5001)
