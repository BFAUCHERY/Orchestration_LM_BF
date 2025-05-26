from roboflow import Roboflow
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

# Remplace ici par ta vraie API Key : https://app.roboflow.com > Settings > API Key
rf = Roboflow(api_key="umHpbGUZiPoJhrKn6T7g")

# Récupère ton workspace et projet
workspace = rf.workspace("iabdlmbf")
project = workspace.project("road-signs-sbjnd")

# 📂 Dossier contenant les .jpg + .txt (plats)
upload_dir = "roboflow_upload/train"
tag = "train"  # ou "test" selon ce que tu veux marquer

print("⚠️ Info : création automatique de version non supportée. Les images seront ajoutées à la version la plus récente.")

# Récupère les images déjà présentes sur Roboflow (version 1)
existing_images = project.version("1").images()
existing_filenames = set(img.name for img in existing_images)
print(f"🔍 {len(existing_filenames)} images déjà présentes sur Roboflow.")

def upload_image(file):
    if file.endswith(".png"):
        if file in existing_filenames:
            print(f"⏩ {file} déjà présent, ignoré")
            return
        image_path = os.path.join(upload_dir, file)
        base_name = os.path.splitext(file)[0]
        annotation_path = os.path.join(upload_dir, base_name + ".txt")
        if os.path.exists(annotation_path):
            print(f"📤 Upload : {file} + {base_name}.txt")
            project.upload(image_path, annotation_path=annotation_path, num_retry_uploads=3, tags=[tag])
        else:
            print(f"⚠️  Annotation manquante pour : {file}, ignorée")

with ThreadPoolExecutor(max_workers=25) as executor:
    futures = [executor.submit(upload_image, file) for file in os.listdir(upload_dir)]
    for _ in as_completed(futures):
        pass

print("✅ Upload terminé.")