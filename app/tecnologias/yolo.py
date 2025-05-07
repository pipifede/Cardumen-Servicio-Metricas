from ultralytics import YOLO
from pathlib import Path
import numpy as np

class YOLOModel:
    def __init__(self, model_path: str):
        self.model = YOLO(model_path)

    def predict(self, image_path: str):
        results = self.model(image_path)
        return results

    def load_model(self):
        # Load the model if needed, or perform any initialization
        pass

    def process_image(self, image: np.ndarray):
        results = self.predict(image)
        detections = []
        for result in results:
            for detection in result.boxes:
                detections.append({
                    "class_name": detection.cls,
                    "confidence": detection.conf,
                    "bbox": detection.xyxy.tolist()  # Convertir a lista para serialización JSON
                })
        return results[0].plot()  # Devuelve la imagen procesada con las detecciones dibujadas
    
    # funcion para devolver el video procesado
    def process_video(self, video_path: str, output_path: str):
        results = self.model(video_path, show=False, save=True, save_txt=False, project=output_path)
        return results