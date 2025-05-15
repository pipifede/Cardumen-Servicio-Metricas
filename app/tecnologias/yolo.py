from ultralytics import YOLO
from pathlib import Path
import numpy as np
import cv2
import time
import psutil

class YOLOModel:
    def __init__(self, model_path: str):
        self.model = YOLO(model_path)
        self.metrics = {
            'total_frames': 0,
            'total_inference_time': 0,
            'total_processing_time': 0,
            'confidences': [],
            'cpu_usage': []
        }

    def start_metrics(self):
        """Reinicia las métricas para cada procesamiento"""
        self.metrics = {
            'total_frames': 0,
            'total_inference_time': 0,
            'total_processing_time': 0,
            'confidences': [],
            'cpu_usage': [],
            'start_time': time.time()
        }

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
        """Procesa un frame individual"""
        print(f"\r Procesando frame {current_frame}/{total_frames}", end="", flush=True)
        start_time = time.time()
        
        # Procesamiento con YOLO
        results = self.model.track(image, persist=True, verbose=False)
        inference_time = time.time() - start_time
        
        # Post-procesamiento
        processed_frame = results[0].plot()
        processing_time = time.time() - start_time
        
        # Obtener confianzas
        confidences = []
        if results[0].boxes is not None:
            confidences = results[0].boxes.conf.cpu().numpy().tolist()
        
        # Actualizar métricas
        self.update_frame_metrics(inference_time, processing_time, confidences)
        
        return processed_frame

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
            start_time = time.time()
            
            # Inferencia
            results = self.model.track(frame, persist=True, verbose=False)
            
            # Tiempo de inferencia
            inference_time = time.time() - start_time
            self.metrics['total_inference_time'] += inference_time
            
            # Post-procesamiento
            processed_frame = results[0].plot()
            
            # Tiempo de procesamiento total
            processing_time = time.time() - start_time
            self.metrics['total_processing_time'] += processing_time
            
            # Confianzas de detecciones
            if results[0].boxes is not None:
                self.metrics['confidences'].extend(results[0].boxes.conf.cpu().numpy().tolist())
            
            # Escribir frame procesado
            out.write(processed_frame)
            
            # Métricas adicionales
            self.metrics['cpu_usage'].append(psutil.cpu_percent())
            self.metrics['total_frames'] += 1

        cap.release()
        out.release()
        print(f"\n✅ Procesamiento completado - {self.metrics['total_frames']} frames procesados")
        print(f"📁 Video guardado en: {output_video}")
        return {
            "output_path": output_video,
            "metrics": self.get_metrics()
        }