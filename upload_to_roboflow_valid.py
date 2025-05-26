from roboflow import Roboflow
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

# Remplace ici par ta vraie API Key : https://app.roboflow.com > Settings > API Key
rf = Roboflow(api_key="umHpbGUZiPoJhrKn6T7g")

# Récupère ton workspace et projet
workspace = rf.workspace("iabdlmbf")
project = workspace.project("road-signs-sbjnd")

# 📂 Dossier contenant les .jpg + .txt (plats)
upload_dir = "roboflow_upload/valid"
tag = "valid"  # ou "test" selon ce que tu veux marquer
split_name="valid"

print("Info : création automatique de version non supportée. Les images seront ajoutées à la version la plus récente.")

def upload_image(file):
    if file.endswith(".png"):
        image_path = os.path.join(upload_dir, file)
        base_name = os.path.splitext(file)[0]
        annotation_path = os.path.join(upload_dir, base_name + ".txt")
        if os.path.exists(annotation_path):
            print(f"Upload : {file} + {base_name}.txt")
            project.upload(image_path, annotation_path=annotation_path, num_retry_uploads=3, split=split_name)
        else:
            print(f"Annotation manquante pour : {file}, ignorée")

with ThreadPoolExecutor(max_workers=12) as executor:
    futures = [executor.submit(upload_image, file) for file in os.listdir(upload_dir)]
    for _ in as_completed(futures):
        pass

print("Upload terminé.")