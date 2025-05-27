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

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Initialisation du projet Kedro (permet de charger le contexte du projet)
bootstrap_project(Path.cwd())

# Extensions autorisées
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'webp'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def analyze_sign():
    """
    Simule l'analyse d'un panneau de signalisation
    Dans une vraie application, ici on utiliserait un modèle de ML et OCR
    """
    # Classification
    run_pipelines(["evaluateYOLO"])
    
    with open('data/08_outputs/yolo_predictions.json', 'r', encoding='utf-8') as f:
        data = json.load(f) 

    all_classes = []
    for item in data:
        all_classes.extend(item.get('classes', [])) 

    if all_classes:
        category = all_classes[0]
    else:
        category = "Aucune classe trouvée."
    
    # Extraction du texte
    run_pipelines(["OCRtesseract"])
    with open('data/08_outputs/ocr_text.txt', 'r', encoding='utf-8') as f:
        detected_text = f.read().strip()
    
    # Simulation du niveau de confiance
    confidence = round(random.uniform(0.75, 0.98), 2)
    
    return {
        'category': category,
        'category_name': "sample name",
        'category_description': "sample description",
        'detected_text': detected_text,
        'confidence': confidence,
        'analysis_details': {
            'image_processed': True,
            'text_regions_found': random.randint(1, 3),
            'processing_time_ms': random.randint(150, 800)
        }
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
        'version': '1.0.0'
    })

@app.route('/api/analyze', methods=['POST'])
def api_analyze():
    try:
        # Vérifier si un fichier a été envoyé
        if 'image' not in request.files:
            return jsonify({'error': 'Aucune image fournie'}), 400
        
        file = request.files['image']
        
        if file.filename == '':
            return jsonify({'error': 'Aucun fichier sélectionné'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({'error': 'Format de fichier non supporté'}), 400
        
        # Sauvegarder le fichier
        filename = "image_to_predict.png"
        project_root = os.path.dirname(os.path.abspath(__file__))
        data_folder = os.path.join(project_root, 'data/07_predict/')
        os.makedirs(data_folder, exist_ok=True)
        data_folder = os.path.abspath(data_folder)
        filepath = os.path.join(data_folder, filename)
        # Supprimer le fichier existant s'il existe déjà
        if os.path.exists(filepath):
            os.remove(filepath)
        file.save(filepath)
        print(f"Fichier sauvegardé à: {filepath}")
        try:
            # Vérifier que c'est bien une image valide
            with Image.open(filepath) as img:
                # Convertir en RGB si nécessaire
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                
                # Redimensionner si trop grande (pour optimiser)
                max_size = (800, 600)
                img.thumbnail(max_size, Image.Resampling.LANCZOS)
                
                # Analyser l'image (simulation)
                analysis_result = analyze_sign()
                
                # Encoder l'image en base64 pour l'affichage
                buffer = io.BytesIO()
                img.save(buffer, format='JPEG', quality=85)
                img_base64 = base64.b64encode(buffer.getvalue()).decode()
                
                response = {
                    'success': True,
                    'result': analysis_result,
                    'image_preview': f"data:image/jpeg;base64,{img_base64}",
                    'filename': file.filename
                }
                
                return jsonify(response)
                
        except Exception as e:
            return jsonify({'error': f'Erreur lors du traitement de l\'image: {str(e)}'}), 400
        
        finally:
            # Nettoyer le fichier temporaire
            if os.path.exists(filepath):
                os.remove(filepath)
    
    except Exception as e:
        return jsonify({'error': f'Erreur serveur: {str(e)}'}), 500

@app.route('/api/categories')
def get_categories():
    """Retourne la liste des catégories de panneaux supportées"""
    return jsonify({
        'categories': SIGN_CATEGORIES,
        'total_categories': len(SIGN_CATEGORIES)
    })

@app.errorhandler(413)
def too_large(e):
    return jsonify({'error': 'Fichier trop volumineux (max 16MB)'}), 413

@app.errorhandler(404)
def not_found(e):
    return jsonify({'error': 'Endpoint non trouvé'}), 404

@app.errorhandler(500)
def internal_error(e):
    return jsonify({'error': 'Erreur interne du serveur'}), 500

if __name__ == '__main__':
    print("🚦 Démarrage de l'API d'analyse de panneaux de signalisation...")
    print("📍 Interface disponible sur: http://localhost:5000")
    print("🔗 API endpoint: http://localhost:5000/api/analyze")
    app.run(debug=True, host='0.0.0.0', port=5000)
