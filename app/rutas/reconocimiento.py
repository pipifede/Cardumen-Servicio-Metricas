from fastapi import APIRouter, File, UploadFile, WebSocket, WebSocketException, status, WebSocketDisconnect, Request, Form 
from fastapi.responses import StreamingResponse
from pathlib import Path
from app.tecnologias.yolo import YOLOModel
from app.tecnologias.mediaPipe import MediaPipeObjectDetector
from app.config import UPLOAD_DIR, PROCESSED_DIR, YOLO_MODEL_PATH, MODELOSYOLO, MODELOS_MEDIAPIPE, MEDIAPIPE_MODEL_PATH
import os
import shutil
import uuid
import cv2
import numpy as np
import base64
import mimetypes
import moviepy
from time import time

router = APIRouter()

def checkModelo(modelo: str, tecnologia: str):
    if tecnologia == "yolo":
        return modelo in MODELOSYOLO
    elif tecnologia == "mediapipe":
        return modelo in MODELOS_MEDIAPIPE
    return False

def convert_avi_to_mp4(avi_file_path):
    if not os.path.exists(avi_file_path):
        raise FileNotFoundError(avi_file_path)
    clip = moviepy.VideoFileClip(avi_file_path)
    path, file_name = os.path.split(avi_file_path)
    output_name = os.path.join(path, 'tmp_' + os.path.splitext(file_name)[0] + '.mp4')
    clip.write_videofile(output_name, codec="libx264")
    return output_name

@router.post("/upload/video")
async def upload_video(request: Request, file: UploadFile = File(...), tecnologia: str = Form("yolo"), modelo: str = Form(MODELOSYOLO[0])):
    print("\n\n✅ TECNOLOGÍA RECIBIDA:", tecnologia)
    if tecnologia not in ["yolo", "mediapipe"]:
        return {"error": f"Tecnología no soportada: {tecnologia}"}

    if not checkModelo(modelo, tecnologia):
        return {"error": f"Modelo '{modelo}' no es válido para la tecnología '{tecnologia}'"}

    if not file.filename.endswith(('.jpg', '.jpeg', '.png', '.mp4')):
        return {"error": "Tipo de archivo no soportado"}

    uuid_str = str(uuid.uuid4())
    input_path = Path(UPLOAD_DIR) / "videos" / f"{uuid_str}{Path(file.filename).suffix}"
    input_path.parent.mkdir(parents=True, exist_ok=True)

    with open(input_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    output_path = Path(PROCESSED_DIR) / "videos" / uuid_str
    output_path.mkdir(parents=True, exist_ok=True)

    if tecnologia == "yolo":
        model = YOLOModel(Path(YOLO_MODEL_PATH) / modelo)
        model.process_video(str(input_path), str(output_path))
        avi_output_dir = output_path / "predict"
        output_files = list(avi_output_dir.glob("*.avi"))
        if not output_files:
            return {"error": "No se encontró archivo de salida"}
        avi_file = output_files[0]
    elif tecnologia == "mediapipe":
        model_path = Path(MEDIAPIPE_MODEL_PATH) / modelo
        model = MediaPipeObjectDetector(str(model_path))
        avi_file = model.process_video(str(input_path), str(output_path))

    file_path = convert_avi_to_mp4(str(avi_file))
    media_type, _ = mimetypes.guess_type(str(file_path))
    media_type = media_type or "application/octet-stream"
    file_size = os.path.getsize(file_path)
    range_header = request.headers.get('range')

    def iterfile(start=0, end=None):
        with open(file_path, "rb") as f:
            f.seek(start)
            remaining = end - start + 1 if end is not None else file_size - start
            while remaining > 0:
                chunk_size = min(1024 * 1024, remaining)
                data = f.read(chunk_size)
                if not data:
                    break
                yield data
                remaining -= len(data)

    if range_header:
        start_str, end_str = range_header.strip().lower().replace("bytes=", "").split("-")
        start = int(start_str)
        end = int(end_str) if end_str else file_size - 1
        content_length = end - start + 1

        return StreamingResponse(
            iterfile(start, end),
            status_code=206,
            media_type="video/mp4",
            headers={
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Expose-Headers": "Content-Range, Accept-Ranges, X-Tecnologia",
                "Content-Range": f"bytes {start}-{end}/{file_size}",
                "Accept-Ranges": "bytes",
                "Content-Length": str(content_length),
                "X-Tecnologia": tecnologia
            }
        )
    else:
        return StreamingResponse(
            iterfile(),
            status_code=200,
            media_type="video/mp4",
            headers={
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Expose-Headers": "Content-Range, Accept-Ranges, X-Tecnologia",
                "Accept-Ranges": "bytes",
                "Content-Length": str(file_size),
                "X-Tecnologia": tecnologia
            }
        )

@router.websocket("/ws/image")
async def websocket_image(websocket: WebSocket, tecnologia: str = "yolo", modelo: str = MODELOSYOLO[0]):
    tecnologia = tecnologia.lower()

    if tecnologia not in ["yolo", "mediapipe"]:
        raise WebSocketException(reason="TECNOLOGIA INVALIDA")
    if not checkModelo(modelo, tecnologia):
        raise WebSocketException(reason=f"Modelo '{modelo}' no es válido para la tecnología '{tecnologia}'")

    await websocket.accept()
    print("[WebSocket] Cliente conectado")

    if tecnologia == "yolo":
        model = YOLOModel(YOLO_MODEL_PATH / modelo)
    elif tecnologia == "mediapipe":
        model_path = Path(MEDIAPIPE_MODEL_PATH) / modelo
        model = MediaPipeObjectDetector(str(model_path))

    try:
        timestamp_ms = 0
        while True:
            data = await websocket.receive_text()
            image_data = base64.b64decode(data)
            nparr = np.frombuffer(image_data, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            timestamp_ms += 1
            processed_image = model.process_image(image, timestamp_ms)

            _, buffer = cv2.imencode('.jpg', processed_image)
            processed_image_base64 = base64.b64encode(buffer).decode('utf-8')

            await websocket.send_text(processed_image_base64)
    except WebSocketDisconnect:
        print("[WebSocket] Cliente desconectado")
