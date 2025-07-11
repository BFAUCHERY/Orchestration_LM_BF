from ultralytics import YOLO

def train_yolo(data_yaml_path: str) -> str:
    """
    Entraîne YOLOv8 pendant 1 époque sur un dataset YOLO (via fichier data.yaml).

    Args:
        data_yaml_path: Chemin vers le fichier data.yaml

    Returns:
        Chemin vers le modèle entraîné (model.pt)
    """
    model = YOLO("model/yolov8n.pt")
    model.train(data=data_yaml_path, epochs=1, imgsz=160, batch=4, device="cpu", patience=0) 
    
    # Retourne un modèle qui ne sera pas utilisé, on garde le modèle déjà utilisé
    return "model/model.pt"