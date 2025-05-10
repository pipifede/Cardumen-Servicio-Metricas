from fastapi import APIRouter, File, UploadFile, WebSocket, WebSocketDisconnect, Request, Form 
from fastapi.responses import StreamingResponse, JSONResponse
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
import asyncio
import json

router = APIRouter()

# Diccionario para mantener las conexiones WebSocket activas
active_connections = {}

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
    output_name = os.path.join(path, 'processed_' + os.path.splitext(file_name)[0] + '.mp4')
    clip.write_videofile(output_name, codec="libx264")
    return output_name

async def process_and_stream_video(file_path: str, output_path: str, tecnologia: str, modelo: str, task_id: str):
    """Procesa el video y envía updates por WebSocket"""
    try:
        if tecnologia == "yolo":
            model = YOLOModel(Path(YOLO_MODEL_PATH) / modelo)
            model.process_video(str(file_path), str(output_path))
            avi_output_dir = output_path / "predict"
            output_files = list(avi_output_dir.glob("*.avi"))
            if not output_files:
                raise FileNotFoundError("No se encontró archivo de salida YOLO")
            avi_file = output_files[0]
        else:
            model_path = Path(MEDIAPIPE_MODEL_PATH) / modelo
            model = MediaPipeObjectDetector(str(model_path))
            avi_file = model.process_video(str(file_path), str(output_path))

        mp4_file = convert_avi_to_mp4(str(avi_file))
        
        # Notificar finalización
        if task_id in active_connections:
            await active_connections[task_id].send_text(json.dumps({
                "type": "complete",
                "output_path": Path(mp4_file).name,
                "task_id": task_id
            }))
            
        return mp4_file
    except Exception as e:
        if task_id in active_connections:
            await active_connections[task_id].send_text(json.dumps({
                "type": "error",
                "message": str(e)
            }))
        raise

@router.post("/upload/video")
async def upload_video(
    request: Request, 
    file: UploadFile = File(...), 
    tecnologia: str = Form("yolo"), 
    modelo: str = Form(MODELOSYOLO[0])
):
    print(f"\n✅ Procesando video con {tecnologia} - Modelo: {modelo}")
    
    if tecnologia not in ["yolo", "mediapipe"]:
        return JSONResponse({"error": f"Tecnología no soportada: {tecnologia}"}, status_code=400)

    if not checkModelo(modelo, tecnologia):
        return JSONResponse(
            {"error": f"Modelo '{modelo}' no válido para {tecnologia}"}, 
            status_code=400
        )

    if not file.filename.lower().endswith(('.mp4', '.avi', '.mov')):
        return JSONResponse({"error": "Formato de video no soportado"}, status_code=400)

    task_id = str(uuid.uuid4())
    input_path = Path(UPLOAD_DIR) / "videos" / f"{task_id}{Path(file.filename).suffix}"
    input_path.parent.mkdir(parents=True, exist_ok=True)

    with open(input_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    output_path = Path(PROCESSED_DIR) / "videos" / task_id
    output_path.mkdir(parents=True, exist_ok=True)

    # Iniciar procesamiento en segundo plano
    asyncio.create_task(process_and_stream_video(
        input_path, output_path, tecnologia, modelo, task_id
    ))

    return JSONResponse({
        "status": "processing", 
        "task_id": task_id,
        "message": "El video está siendo procesado"
    })

@router.websocket("/ws/progress/{task_id}")
async def progress_websocket(websocket: WebSocket, task_id: str):
    await websocket.accept()
    active_connections[task_id] = websocket
    
    try:
        while True:
            # Mantener la conexión activa
            await websocket.receive_text()
    except WebSocketDisconnect:
        print(f"WebSocket desconectado para task_id: {task_id}")
    finally:
        if task_id in active_connections:
            del active_connections[task_id]

@router.websocket("/ws/image")
async def websocket_image(websocket: WebSocket, tecnologia: str = "yolo", modelo: str = MODELOSYOLO[0]):
    tecnologia = tecnologia.lower()
    await websocket.accept()
    print(f"[WebSocket] Cliente conectado - Tecnología: {tecnologia}, Modelo: {modelo}")

    try:
        # Inicializar modelo
        if tecnologia == "yolo":
            model = YOLOModel(Path(YOLO_MODEL_PATH) / modelo)
        else:
            model = MediaPipeObjectDetector(str(Path(MEDIAPIPE_MODEL_PATH) / modelo))

        timestamp_ms = 0
        while True:
            data = await websocket.receive_text()
            
            # Decodificar imagen
            image_data = base64.b64decode(data)
            nparr = np.frombuffer(image_data, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            # Procesar frame (manejo diferente para YOLO y MediaPipe)
            if tecnologia == "yolo":
                processed_image = model.process_image(image)  # YOLO solo necesita la imagen
            else:
                timestamp_ms += 1
                processed_image = model.process_image(image, timestamp_ms)  # MediaPipe necesita timestamp
            
            # Codificar y enviar resultado
            _, buffer = cv2.imencode('.jpg', processed_image)
            await websocket.send_text(base64.b64encode(buffer).decode('utf-8'))
            
    except WebSocketDisconnect:
        print("[WebSocket] Cliente desconectado")
    except Exception as e:
        print(f"[WebSocket] Error: {str(e)}")
        await websocket.close(code=1011)  # Código 1011 = Internal Error