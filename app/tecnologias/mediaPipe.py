from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import ObjectDetectorOptions
from mediapipe.tasks.python.core.base_options import BaseOptions
from mediapipe.tasks.python.vision.object_detector import ObjectDetector
from mediapipe.tasks.python.vision.core.vision_task_running_mode import VisionTaskRunningMode
import mediapipe as mp
import numpy as np
import cv2
import os


class MediaPipeObjectDetector:
    def __init__(self, model_path: str):
        options = ObjectDetectorOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=VisionTaskRunningMode.VIDEO,
            max_results=25
        )
        self.detector = ObjectDetector.create_from_options(options)

    def process_image(self, image: np.ndarray, timestamp_ms: int = 0):
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)
        result = self.detector.detect_for_video(mp_image, timestamp_ms)
        detections = result.detections

        for detection in detections:
            category = detection.categories[0]
            if category.category_name.lower() != "person":
                continue

            bbox = detection.bounding_box
            start_point = (int(bbox.origin_x), int(bbox.origin_y))
            end_point = (int(bbox.origin_x + bbox.width), int(bbox.origin_y + bbox.height))

            cv2.rectangle(image, start_point, end_point, (0, 255, 0), 2)
            label = f"{category.category_name} {category.score:.2f}"
            cv2.putText(image, label, (start_point[0], start_point[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        return image

    def process_video(self, video_path: str, output_path: str):
        cap = cv2.VideoCapture(video_path)
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        os.makedirs(output_path, exist_ok=True)
        out_path = os.path.join(output_path, "result.avi")

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30

        out = cv2.VideoWriter(out_path, fourcc, fps, (width, height))
        frame_count = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            timestamp_ms = int((frame_count / fps) * 1000)
            frame_count += 1

            processed = self.process_image(frame, timestamp_ms)
            out.write(processed)

        cap.release()
        out.release()
        return out_path

