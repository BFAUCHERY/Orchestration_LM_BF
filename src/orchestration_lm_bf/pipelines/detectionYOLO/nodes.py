import os
import cv2
import pandas as pd

from ultralytics import YOLO
from sklearn.model_selection import train_test_split


def load_data(test_csv_path: str, image_dir: str) -> pd.DataFrame:
    """
    Charge les annotations et les chemins d'images depuis le fichier CSV de test.
    """
    df = pd.read_csv(test_csv_path)
    df["full_path"] = df["Path"].apply(lambda p: os.path.join(image_dir, os.path.basename(p)))

    return df


def preprocess_data(df: pd.DataFrame, output_dir: str) -> None:
    """
    Convertit les annotations en format YOLOv8 et enregistre les images + labels.
    """
    image_out = os.path.join(output_dir, "images")
    label_out = os.path.join(output_dir, "labels")
    os.makedirs(image_out, exist_ok=True)
    os.makedirs(label_out, exist_ok=True)

    for _, row in df.iterrows():
        img = cv2.imread(row["full_path"])
        if img is None:
            continue
        h, w = img.shape[:2]
        x1, y1, x2, y2 = row["Roi.X1"], row["Roi.Y1"], row["Roi.X2"], row["Roi.Y2"]
        x_center = ((x1 + x2) / 2) / w
        y_center = ((y1 + y2) / 2) / h
        bbox_width = (x2 - x1) / w
        bbox_height = (y2 - y1) / h
        class_id = int(row["ClassId"])

        filename = os.path.splitext(os.path.basename(row["full_path"]))[0]
        cv2.imwrite(os.path.join(image_out, f"{filename}.jpg"), img)

        with open(os.path.join(label_out, f"{filename}.txt"), "w") as f:
            f.write(f"{class_id} {x_center} {y_center} {bbox_width} {bbox_height}\n")


def train_yolov8(data_path: str, model_path: str) -> str:
    """
    Entraîne un modèle YOLOv8 à partir du dataset formaté.
    """
    model = YOLO("yolov8n.pt")  # modèle pré-entraîné
    model.train(data=data_path, epochs=5, imgsz=640)
    model.save(model_path)
    return model_path


def evaluate_yolov8(model_path: str, data_path: str) -> float:
    """
    Évalue le modèle YOLOv8 et retourne une métrique (ex: mAP50).
    """
    model = YOLO(model_path)
    metrics = model.val(data=data_path)
    return metrics.box.map50  # mAP@0.5
