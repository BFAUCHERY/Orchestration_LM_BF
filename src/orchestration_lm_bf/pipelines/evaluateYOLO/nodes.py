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


def predict_yolo(model_path: str, image_path: str) -> list:
    """
    Fait une prédiction avec un modèle YOLOv8 sur une image donnée.

    Args:
        model_path: Chemin vers le fichier .pt du modèle YOLO entraîné.
        image_path: Chemin vers l'image sur laquelle faire la prédiction.

    Returns:
        Liste des résultats de prédiction (bounding boxes, classes, scores, etc.)
    """
    # Charge le modèle
    model = YOLO(model_path)
    
    # Fait la prédiction
    results = model.predict(image_path, device="cpu")
    
    # On peut transformer les résultats pour ne retourner que ce qui est nécessaire
    predictions = []
    for result in results:
        boxes = result.boxes.xyxy.cpu().numpy().tolist()  # coordonnées des bounding boxes
        scores = result.boxes.conf.cpu().numpy().tolist()  # scores de confiance
        classes = result.boxes.cls.cpu().numpy().tolist()  # classes prédites
        category_names = []

        for i in range(len(classes)):
            category_names.append(result.names[int(classes[i])])  # noms des classes
        
        predictions.append({
            "boxes": boxes,
            "scores": scores,
            "classes": category_names
        })
    
    return predictions