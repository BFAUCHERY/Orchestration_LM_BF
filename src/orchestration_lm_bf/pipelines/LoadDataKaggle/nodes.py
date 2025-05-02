"""
This is a boilerplate pipeline 'LoadDataKaggle'
generated using Kedro 0.19.12
"""

"""
This is a boilerplate pipeline 'LoadDataKaggle'
generated using Kedro 0.19.12
"""

import os
import zipfile
import glob
import cv2
import kaggle

def load_and_prepare_gtsrb_data() -> dict:
    """
    Télécharge et charge les images du dataset GTSRB depuis Kaggle,
    si elles ne sont pas déjà présentes localement.
    Retourne un dictionnaire {filename: image}.
    """
    dataset_dir = "data/01_raw/gtsrb/GTSRB/Final_Training/Images/"
    zip_path = "data/01_raw/gtsrb-german-traffic-sign.zip"
    download_flag = False

    # Vérifie si les images sont déjà présentes
    if not os.path.exists(dataset_dir) or len(glob.glob(os.path.join(dataset_dir, "**/*.ppm"), recursive=True)) == 0:
        print("Téléchargement du dataset GTSRB depuis Kaggle...")
        kaggle.api.dataset_download_files('meowmeowmeowmeowmeow/gtsrb-german-traffic-sign', path="data/01_raw", unzip=False)
        download_flag = True

    # Décompression si nécessaire
    if download_flag and os.path.exists(zip_path):
        print("Décompression du fichier ZIP...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall("data/01_raw")
        os.remove(zip_path)

    # Chargement des images
    print("Chargement des images GTSRB...")
    data = {}
    class_dirs = [d for d in glob.glob(os.path.join(dataset_dir, "*")) if os.path.isdir(d)]

    for class_dir in class_dirs:
        images = glob.glob(os.path.join(class_dir, "*.ppm"))
        for img_path in images:
            img = cv2.imread(img_path)
            if img is not None:
                filename = os.path.basename(img_path)
                data[filename] = img

    return data