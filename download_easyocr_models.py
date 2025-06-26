import easyocr
from pathlib import Path

# Créer le dossier local de destination si nécessaire
model_dir = Path("models/easyocr")
model_dir.mkdir(parents=True, exist_ok=True)

# Lancer le téléchargement dans ce dossier
reader = easyocr.Reader(['en'], gpu=False, model_storage_directory=str(model_dir))

print("✅ Modèles EasyOCR téléchargés dans 'models/easyocr'")