from pathlib import Path
import os

UPLOAD_DIR = Path("app/static/uploads")
PROCESSED_DIR = Path("app/static/processed")
YOLO_MODEL_PATH = Path("app/tecnologias/yoloModels")
MEDIAPIPE_MODEL_PATH = Path("app/tecnologias/mediapipeModels")

# Crear directorios si no existen
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(UPLOAD_DIR / "videos", exist_ok=True)
os.makedirs(PROCESSED_DIR / "videos", exist_ok=True)
os.makedirs(YOLO_MODEL_PATH, exist_ok=True)
os.makedirs(MEDIAPIPE_MODEL_PATH, exist_ok=True)

MODELOSYOLO = ['yolo11n.pt', 'yolo11s.pt', 'yolo11m.pt', 'yolo11l.pt', 'yolo11x.pt']
MODELOS_MEDIAPIPE = ["efficientdet_lite0_pf32.tflite","efficientdet_lite0_int8.tflite", "efficientdet_lite0_pf16.tflite", "ssd_mobilenet_v2_int8.tflite", "ssd_mobilenet_v2_pf32.tflite" ]
