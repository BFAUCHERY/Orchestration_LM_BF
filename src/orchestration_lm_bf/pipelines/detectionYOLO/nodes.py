from ultralytics import YOLO
import pandas as pd

def train_yolo(raw_data:pd.DataFrame):
    model = YOLO('yolov8n.pt')  # ou yolov8s.pt selon les ressources disponibles
    model.train(data=raw_data["path"], epochs=20, imgsz=640)
    return model

def evaluate_yolo(model):
    metrics = model.val()  # utilise les données de validation dans le dossier `val` de data.yaml
    return metrics