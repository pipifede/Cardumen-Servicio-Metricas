from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import ObjectDetectorOptions
from mediapipe.tasks.python.core.base_options import BaseOptions
from mediapipe.tasks.python.vision.object_detector import ObjectDetector
from mediapipe.tasks.python.vision.core.vision_task_running_mode import VisionTaskRunningMode
import mediapipe as mp
import numpy as np
import cv2
import os
import time
import psutil

class MediaPipeObjectDetector:
    def __init__(self, model_path: str):
        options = ObjectDetectorOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=VisionTaskRunningMode.VIDEO,
            max_results=25,
            score_threshold=0.2
        )
        self.detector = ObjectDetector.create_from_options(options)
        self.metrics = {
            'total_frames': 0,
            'total_inference_time': 0,
            'total_postprocess_time': 0,
            'confidences': [],
            'cpu_usage': [],
        }

    def start_metrics(self):
        """Reinicia las métricas para cada procesamiento"""
        self.metrics = {
            'total_frames': 0,
            'total_inference_time': 0,
            'total_postprocess_time': 0,
            'confidences': [],
            'cpu_usage': [],
            'start_time': time.time()
        }

    def _get_metrics(self):
        """Calcula las métricas finales"""
        elapsed = time.time() - self.metrics['start_time']
        
        return {
            'cpu_usage': np.mean(self.metrics['cpu_usage']) if self.metrics['cpu_usage'] else 0,
            'cpu_time': time.process_time(),
            'wall_clock_time': elapsed,
            'confidence': np.mean(self.metrics['confidences']) if self.metrics['confidences'] else 0,
            'avg_inference_time': self.metrics['total_inference_time'] / self.metrics['total_frames'] if self.metrics['total_frames'] > 0 else 0,
            'avg_postprocess_time': self.metrics['total_postprocess_time'] / self.metrics['total_frames'] if self.metrics['total_frames'] > 0 else 0,
            'total_frames': self.metrics['total_frames']
        }

    def update_frame_metrics(self, inference_time, postprocess_time, confidences):
        """Actualiza las métricas con los datos del frame actual"""
        self.metrics['total_frames'] += 1
        self.metrics['total_inference_time'] += inference_time
        self.metrics['total_postprocess_time'] += postprocess_time
        self.metrics['confidences'].extend(confidences)
        self.metrics['cpu_usage'].append(psutil.cpu_percent())
        self.metrics['last_frame_time'] = time.time()
    
    def process_image(self, image: np.ndarray, timestamp_ms: int = 0, current_frame: int = 1, total_frames: int = 1):
        """Procesa un frame individual"""
        print(f"\r Procesando frame {current_frame}/{total_frames}", end="", flush=True)
        start_time = time.time()
        confidences = []

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)
        inference_start = time.time()
        result = self.detector.detect_for_video(mp_image, timestamp_ms)
        inference_time = time.time() - inference_start

        # Post-procesamiento
        postprocess_start = time.time()
        detections = result.detections
        for detection in detections:
            category = detection.categories[0]
            confidences.append(category.score)

            # Dibujar bounding box
            bbox = detection.bounding_box
            start_point = (int(bbox.origin_x), int(bbox.origin_y))
            end_point = (int(bbox.origin_x + bbox.width), int(bbox.origin_y + bbox.height))

            cv2.rectangle(image, start_point, end_point, (0, 255, 0), 2)
            label = f"{category.category_name} {category.score:.2f}"
            cv2.putText(image, label, (start_point[0], start_point[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        postprocess_time = time.time() - postprocess_start

        # Actualizar métricas
        self.update_frame_metrics(inference_time, postprocess_time, confidences)

        return image

    def get_current_metrics(self):
        """Devuelve las métricas calculadas hasta el momento"""
        elapsed = time.time() - self.metrics['start_time']
        return {
            'cpu_usage': np.mean(self.metrics['cpu_usage']) if self.metrics['cpu_usage'] else 0,
            'wall_clock_time': elapsed,
            'confidence': np.mean(self.metrics['confidences']) if self.metrics['confidences'] else 0,
            'avg_inference_time': self.metrics['total_inference_time'] / self.metrics['total_frames'] if self.metrics['total_frames'] > 0 else 0,
            'avg_postprocess_time': self.metrics['total_postprocess_time'] / self.metrics['total_frames'] if self.metrics['total_frames'] > 0 else 0,
            'total_frames': self.metrics['total_frames'],
            'current_fps': self.metrics['total_frames'] / elapsed if elapsed > 0 else 0
        }
    def process_video(self, video_path: str, output_path: str):
        """Procesa un video completo y devuelve métricas"""
        self.start_metrics()
        process = psutil.Process()
        
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        os.makedirs(output_path, exist_ok=True)
        out_path = os.path.join(output_path, "result.avi")
        out = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'XVID'), fps, (width, height))
        frame_count = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Medición de tiempos
            timestamp_ms = int((frame_count / fps) * 1000)
            
            # Procesamiento del frame
            start_time = time.time()
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
            
            # Inferencia
            inference_start = time.time()
            result = self.detector.detect_for_video(mp_image, timestamp_ms)
            self.metrics['total_inference_time'] += time.time() - inference_start
            
            # Post-procesamiento
            postprocess_start = time.time()
            detections = result.detections
            confidences = []
            
            for detection in detections:
                category = detection.categories[0]
                
                # Capturar confianzas
                confidences.append(category.score)
                
                # Dibujar bounding boxes
                bbox = detection.bounding_box
                start_point = (bbox.origin_x, bbox.origin_y)
                end_point = (bbox.origin_x + bbox.width, bbox.origin_y + bbox.height)
                cv2.rectangle(frame, start_point, end_point, (0, 255, 0), 2)
                
                # Etiqueta
                label = f"{category.category_name} {category.score:.2f}"
                cv2.putText(frame, label, (start_point[0], start_point[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            self.metrics['total_postprocess_time'] += time.time() - postprocess_start
            self.metrics['confidences'].extend(confidences)
            
            # Escribir frame
            out.write(frame)
            
            # Métricas adicionales
            self.metrics['cpu_usage'].append(psutil.cpu_percent())
            self.metrics['total_frames'] += 1
            frame_count += 1
            print(f"\r Procesando frame {frame_count}/{total_frames}", end="", flush=True)

        cap.release()
        out.release()
        
        return {
            "output_path": out_path,
            "metrics": self._get_metrics()
        }
    