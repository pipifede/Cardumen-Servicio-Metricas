from ultralytics import YOLO
from pathlib import Path
import numpy as np
import cv2

class YOLOModel:
    def __init__(self, model_path: str):
        self.model = YOLO(model_path)

    def process_image(self, image: np.ndarray):
        """Procesa un frame de imagen con YOLO"""
        results = self.model(image)
        return results[0].plot()  # Devuelve la imagen con las detecciones dibujadas

    def process_video(self, video_path: str, output_path: str):
        """Procesa un video completo"""
        results = self.model(video_path, show=False, save=True, save_txt=False, project=output_path)
        return results