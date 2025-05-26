from kedro.io import AbstractDataset
from ultralytics import YOLO

class YOLOModelDataset(AbstractDataset):
    def __init__(self, filepath):
        self._filepath = filepath

    def _load(self):
        return YOLO(self._filepath)

    def _save(self, model):
        model.save(self._filepath)

    def _describe(self):
        return dict(filepath=self._filepath)
