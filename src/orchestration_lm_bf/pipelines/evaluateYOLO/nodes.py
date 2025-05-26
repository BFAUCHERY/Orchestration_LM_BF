from ultralytics import YOLO

def evaluate_yolov8_10(model_path: str, data_yaml_path: str) -> dict:
    """
    Évalue le modèle YOLOv8 sur tes propres données.

    Args:
        model_path: Chemin vers le modèle pré-entraîné (.pt)
        data_yaml_path: Chemin vers le fichier data.yaml de ton dataset

    Returns:
        Dictionnaire des métriques (mAP, recall, precision, etc.)
    """
    model = YOLO(model_path)
    metrics = model.val(data=data_yaml_path)
    return metrics.results_dict