from roboflow import Roboflow
import os

# Remplace ici par ta vraie API Key : https://app.roboflow.com > Settings > API Key
rf = Roboflow(api_key="umHpbGUZiPoJhrKn6T7g")

# Récupère ton workspace et projet
workspace = rf.workspace("iabdlmbf")
project = workspace.project("road-signs-sbjnd")

# 📂 Dossier contenant les .jpg + .txt (plats)
upload_dir = "roboflow_upload/train"
tag = "train"  # ou "test" selon ce que tu veux marquer

print("⚠️ Info : création automatique de version non supportée. Les images seront ajoutées à la version la plus récente.")

# Parcours tous les fichiers du dossier et upload
for file in os.listdir(upload_dir):
    if file.endswith(".png"):
        image_path = os.path.join(upload_dir, file)
        base_name = os.path.splitext(file)[0]
        annotation_path = os.path.join(upload_dir, base_name + ".txt")
        if os.path.exists(annotation_path):
            print(f"📤 Upload : {file} + {base_name}.txt")
            project.upload(image_path, annotation_path=annotation_path, num_retry_uploads=3, tags=[tag])
        else:
            print(f"⚠️  Annotation manquante pour : {file}, ignorée")

print("✅ Upload terminé.")