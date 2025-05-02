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
    Télécharge et charge les images du dataset GTSRB depuis Kaggle (doganozcan/traffic-sign-gtrb).
    Retourne un dictionnaire {filename: image}.
    """
    dataset_dir = "data/01_raw/traffic-sign-gtrb/"
    zip_path = "data/01_raw/traffic-sign-gtrb.zip"

    print("Téléchargement du dataset GTSRB depuis Kaggle (doganozcan)...")
    kaggle.api.dataset_download_files('doganozcan/traffic-sign-gtrb', path="data/01_raw", unzip=False)

    if os.path.exists(zip_path):
        print("Décompression du fichier ZIP...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall("data/01_raw")
        os.remove(zip_path)

    print("Chargement des images GTSRB...")
    data = {}
    image_paths = glob.glob(os.path.join(dataset_dir, "**/*.png"), recursive=True)

    for img_path in image_paths:
        img = cv2.imread(img_path)
        if img is not None:
            filename = os.path.basename(img_path)
            data[filename] = img

    return data