from flask import Flask, request, jsonify, render_template, send_from_directory
from werkzeug.utils import secure_filename
import os
import uuid
from PIL import Image
import io
import base64
import random

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['UPLOAD_FOLDER'] = 'uploads'

# Créer le dossier uploads s'il n'existe pas
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Extensions autorisées
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'webp'}

# Catégories de panneaux simulées
SIGN_CATEGORIES = {
    'interdiction': {
        'name': 'Panneau d\'interdiction',
        'description': 'Panneau rond avec bordure rouge',
        'examples': ['STOP', 'SENS INTERDIT', 'INTERDICTION DE TOURNER', 'VITESSE LIMITÉE']
    },
    'obligation': {
        'name': 'Panneau d\'obligation',
        'description': 'Panneau rond bleu',
        'examples': ['DIRECTION OBLIGATOIRE', 'PISTE CYCLABLE', 'VOIE RÉSERVÉE']
    },
    'danger': {
        'name': 'Panneau de danger',
        'description': 'Panneau triangulaire avec bordure rouge',
        'examples': ['VIRAGE DANGEREUX', 'PASSAGE À NIVEAU', 'TRAVAUX', 'ÉCOLE']
    },
    'indication': {
        'name': 'Panneau d\'indication',
        'description': 'Panneau rectangulaire informatif',
        'examples': ['SORTIE', 'PARKING', 'HÔPITAL', 'CENTRE VILLE']
    },
    'priorite': {
        'name': 'Panneau de priorité',
        'description': 'Panneau de priorité et cédez le passage',
        'examples': ['CÉDEZ LE PASSAGE', 'PRIORITÉ À DROITE', 'ROUTE PRIORITAIRE']
    }
}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def simulate_sign_analysis(image_path):
    """
    Simule l'analyse d'un panneau de signalisation
    Dans une vraie application, ici on utiliserait un modèle de ML et OCR
    """
    # Simulation de la classification
    category_key = random.choice(list(SIGN_CATEGORIES.keys()))
    category = SIGN_CATEGORIES[category_key]
    
    # Simulation de l'extraction de texte
    detected_text = random.choice(category['examples'])
    
    # Simulation du niveau de confiance
    confidence = round(random.uniform(0.75, 0.98), 2)
    
    return {
        'category': category_key,
        'category_name': category['name'],
        'category_description': category['description'],
        'detected_text': detected_text,
        'confidence': confidence,
        'analysis_details': {
            'image_processed': True,
            'text_regions_found': random.randint(1, 3),
            'processing_time_ms': random.randint(150, 800)
        }
    }

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
def analyze_sign():
    try:
        # Vérifier si un fichier a été envoyé
        if 'image' not in request.files:
            return jsonify({'error': 'Aucune image fournie'}), 400
        
        file = request.files['image']
        
        if file.filename == '':
            return jsonify({'error': 'Aucun fichier sélectionné'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({'error': 'Format de fichier non supporté'}), 400
        
        # Sauvegarder le fichier temporairement
        filename = secure_filename(f"{uuid.uuid4()}_{file.filename}")
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
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
                analysis_result = simulate_sign_analysis(filepath)
                
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
