from ultralytics import YOLO
from pathlib import Path
import numpy as np
import cv2
import time
import psutil
from ultralytics.utils.plotting import Annotator
import torch

class KalmanFilter:
    def __init__(self, initial_x=0, initial_y=0):
        self.kf = cv2.KalmanFilter(4, 2)
        self.kf.measurementMatrix = np.array([[1, 0, 0, 0],
                                              [0, 1, 0, 0]], np.float32)
        self.kf.transitionMatrix = np.array([[1, 0, 1, 0],
                                             [0, 1, 0, 1],
                                             [0, 0, 1, 0],
                                             [0, 0, 0, 1]], np.float32)
        self.kf.processNoiseCov = np.array([[1, 0, 0, 0],
                                            [0, 1, 0, 0],
                                            [0, 0, 1, 0],
                                            [0, 0, 0, 1]], np.float32) * 0.001
        self.kf.measurementNoiseCov = np.array([[1, 0],
                                                 [0, 1]], np.float32) * 10
        
        self.kf.statePost = np.array([[initial_x], [initial_y], [0], [0]], np.float32) 
        
        self.kf.errorCovPost = np.eye(4, dtype=np.float32) * 0.1

        # -----------------------------------

    def predict(self):
        return self.kf.predict()

    def correct(self, measurement):
        return self.kf.correct(measurement)

class YOLOModel:
    def __init__(self, model_path: str):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = YOLO(model_path).to(device)
        print(f"GPU disponible: {torch.cuda.is_available()}")
        if device == "cuda":
            print(f"  - Nombre de GPU: {torch.cuda.get_device_name(0)}")
            print(f"  - Total memoria: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
            print(f"  - Modelo en: {next(self.model.model.parameters()).device}")
        else:
            print("  - Ejecutando en CPU")
        self.metrics = {
            'total_frames': 0,
            'total_inference_time': 0,
            'total_processing_time': 0,
            'confidences': [],
            'cpu_usage': []
        }
        self.trackers = {}
        self.trajectories = {}
        # self.avg_box_dims = {} # ELIMINADO: No es necesario para tamaño fijo

    def start_metrics(self):
        """Reinicia las métricas para cada procesamiento"""
        self.metrics = {
            'total_frames': 0,
            'total_inference_time': 0,
            'total_processing_time': 0,
            'confidences': [],
            'cpu_usage': [], # Re-agregado aquí ya que es necesario para calcular el promedio
            'start_time': time.time()
        }
        self.trackers = {} # Reiniciar trackers al inicio de cada video
        self.trajectories = {} # Reiniciar trayectorias

    def get_metrics(self):
        """Calcula las métricas finales"""
        elapsed = time.time() - self.metrics['start_time']
        
        return {
            'cpu_usage': np.mean(self.metrics['cpu_usage']) if self.metrics['cpu_usage'] else 0,
            'cpu_time': time.process_time(),
            'wall_clock_time': elapsed,
            'confidence': np.mean(self.metrics['confidences']) if self.metrics['confidences'] else 0,
            'avg_inference_time': self.metrics['total_inference_time'] / self.metrics['total_frames'] if self.metrics['total_frames'] > 0 else 0,
            'avg_processing_time': self.metrics['total_processing_time'] / self.metrics['total_frames'] if self.metrics['total_frames'] > 0 else 0,
            'total_frames': self.metrics['total_frames']
        }

    def update_frame_metrics(self, inference_time, processing_time, confidences):
        """Actualiza las métricas con los datos del frame actual"""
        self.metrics['total_frames'] += 1
        self.metrics['total_inference_time'] += inference_time
        self.metrics['total_processing_time'] += processing_time
        self.metrics['confidences'].extend(confidences)
        self.metrics['cpu_usage'].append(psutil.cpu_percent())
        self.metrics['last_frame_time'] = time.time()


    def process_image(self, image: np.ndarray, current_frame: int = 1, total_frames: int = 1):
        print(f"\r Procesando frame {current_frame}/{total_frames}", end="", flush=True)
        start_time = time.time()

        results = self.model.track(image, persist=True, verbose=False)
        inference_time = time.time() - start_time

        processed_frame = image.copy()
        annotator = Annotator(processed_frame, line_width=2)
        confidences = []
        frame_boxes = []  # Lista para almacenar las cajas procesadas

        if results[0].boxes is not None:
            boxes = results[0].boxes
            for box in boxes:
                try:
                    if (
                        not isinstance(box.xyxy, torch.Tensor) or
                        box.xyxy.ndim == 0 or
                        box.xyxy.shape[0] == 0 or
                        box.xyxy[0].ndim == 0
                    ):
                        print(f"DEBUG: Box malformado o vacío en frame {current_frame}, salteando.")
                        continue
                    # Obtener coordenadas originales
                    x1_raw, y1_raw, x2_raw, y2_raw = map(int, box.xyxy[0].cpu().numpy())
                    width_raw = x2_raw - x1_raw
                    height_raw = y2_raw - y1_raw
                    
                    cls = int(box.cls.cpu().item())
                    conf = float(box.conf.cpu().item())
                    track_id = int(box.id.cpu().item()) if box.id is not None and box.id.numel() > 0 else -1

                    if cls == 0:  # persona
                        center_x_raw = (x1_raw + x2_raw) // 2
                        center_y_raw = (y1_raw + y2_raw) // 2

                        # FILTRO DE KALMAN PARA LA POSICIÓN DEL CENTRO DEL BOX
                        if track_id not in self.trackers:
                            self.trackers[track_id] = KalmanFilter(initial_x=center_x_raw, initial_y=center_y_raw)
                            self.trajectories[track_id] = []

                        prediction = self.trackers[track_id].predict()
                        
                        measurement = np.array([[center_x_raw], [center_y_raw]], np.float32)
                        corrected_state = self.trackers[track_id].correct(measurement)
                        
                        predicted_x, predicted_y = int(corrected_state[0]), int(corrected_state[1])
                        
                        # Guardar la posición filtrada por Kalman
                        self.trajectories[track_id].append((predicted_x, predicted_y))

                        # RECONSTRUIR EL BOUNDING BOX USANDO EL CENTRO FILTRADO PERO LAS DIMENSIONES ORIGINALES
                        smoothed_x1 = predicted_x - width_raw // 2
                        smoothed_y1 = predicted_y - height_raw // 2
                        smoothed_x2 = predicted_x + width_raw // 2
                        smoothed_y2 = predicted_y + height_raw // 2

                        # Guardar datos de la caja procesada para el JSON
                        frame_boxes.append({
                            'xyxy': [smoothed_x1, smoothed_y1, smoothed_x2, smoothed_y2],  # Usar centro filtrado pero dimensiones originales
                            'conf': conf,
                            'cls': cls,
                            'id': track_id
                        })

                        label = f"ID {track_id} ({conf*100:.1f}%) ({predicted_x},{predicted_y})"
                        
                        # Dibujar el box con las coordenadas filtradas pero dimensiones originales
                        annotator.box_label([smoothed_x1, smoothed_y1, smoothed_x2, smoothed_y2], label, color=(255, 0, 0))

                        # Dibujar la trayectoria
                        if len(self.trajectories[track_id]) > 1:
                            for i in range(1, len(self.trajectories[track_id])):
                                cv2.line(processed_frame, self.trajectories[track_id][i-1], self.trajectories[track_id][i], (0, 255, 0), 2)
                        
                        confidences.append(conf)

                except Exception as e:
                    print(f"\nError procesando una box en el frame {current_frame}: {e}")
                    print(f"DEBUG: Problematic box details - type(box): {type(box)}")
                    if hasattr(box, 'xyxy'):
                        print(f"DEBUG: box.xyxy type: {type(box.xyxy)}, shape: {box.xyxy.shape if isinstance(box.xyxy, torch.Tensor) else 'N/A'}, numel: {box.xyxy.numel() if isinstance(box.xyxy, torch.Tensor) else 'N/A'}, value: {box.xyxy}")
                    if hasattr(box, 'cls'):
                        print(f"DEBUG: box.cls type: {type(box.cls)}, shape: {box.cls.shape if isinstance(box.cls, torch.Tensor) else 'N/A'}, numel: {box.cls.numel() if isinstance(box.cls, torch.Tensor) else 'N/A'}, value: {box.cls}")
                    if hasattr(box, 'id'):
                        print(f"DEBUG: box.id type: {type(box.id)}, numel: {box.id.numel() if isinstance(box.id, torch.Tensor) else 'N/A'}, value: {box.id}")
                    continue

        processing_time = time.time() - start_time
        self.update_frame_metrics(inference_time, processing_time, confidences)
        
        return {
            "processed_frame": processed_frame,
            "results": results,
            "frame_boxes": frame_boxes  # Añadir las cajas procesadas al resultado
        }

    def get_current_metrics(self):
        """Devuelve las métricas calculadas hasta el momento"""
        elapsed = time.time() - self.metrics['start_time']
        return {
            'cpu_usage': np.mean(self.metrics['cpu_usage']) if self.metrics['cpu_usage'] else 0,
            'wall_clock_time': elapsed,
            'confidence': np.mean(self.metrics['confidences']) if self.metrics['confidences'] else 0,
            'avg_inference_time': self.metrics['total_inference_time'] / self.metrics['total_frames'] if self.metrics['total_frames'] > 0 else 0,
            'avg_processing_time': self.metrics['total_processing_time'] / self.metrics['total_frames'] if self.metrics['total_frames'] > 0 else 0,
            'total_frames': self.metrics['total_frames'],
            'current_fps': self.metrics['total_frames'] / elapsed if elapsed > 0 else 0
        }

    def process_video(self, video_path: str, output_path: str):
        """Procesa un video completo y devuelve métricas"""
        self.start_metrics()
        
        # Configurar video de entrada y salida
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Crear VideoWriter para salida
        output_video = str(Path(output_path) / "output.avi")
        out = cv2.VideoWriter(output_video, cv2.VideoWriter_fourcc(*'XVID'), fps, (width, height))
        frame_number = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame_number += 1
            print(f"\r Procesando frame {frame_number}/{total_frames}", end="", flush=True)
            
            # Procesamiento del frame
            # Asegúrate de que process_image retorne un diccionario con 'processed_frame'
            processed_data = self.process_image(frame, frame_number, total_frames)
            processed_frame = processed_data["processed_frame"] # Extraer el frame del diccionario
            
            # Escribir frame procesado
            out.write(processed_frame)
            
        cap.release()
        out.release()
        print(f"\n✅ Procesamiento completado - {self.metrics['total_frames']} frames procesados")
        print(f"📁 Video guardado en: {output_video}")
        return {
            "output_path": output_video,
            "metrics": self.get_metrics()
        }