from fastapi import FastAPI, UploadFile, Form, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse
import os
import uuid
from pathlib import Path
import asyncio
import json
import cv2
import numpy as np
import base64
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = "uploads"
PROCESSED_DIR = "processed"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)

active_connections = {}

async def process_video_real_time(file_path: str, task_id: str):
    """Procesamiento real del video con envío de frames procesados"""
    cap = cv2.VideoCapture(file_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_count += 1
        
        # Aquí va tu procesamiento real con YOLO/MediaPipe
        processed_frame = process_frame_with_model(frame, framework, model)  # Implementa esta función
        
        # Convertir frame procesado a base64
        _, buffer = cv2.imencode('.jpg', processed_frame)
        frame_base64 = base64.b64encode(buffer).decode('utf-8')
        
        # Enviar frame procesado al frontend
        if task_id in active_connections:
            await active_connections[task_id].send_text(json.dumps({
                "type": "frame",
                "frame": frame_base64,
                "progress": (frame_count / total_frames) * 100,
                "frame_count": frame_count,
                "total_frames": total_frames
            }))
        
        await asyncio.sleep(1/fps)  # Mantener el ritmo del video original
    
    cap.release()
    
    # Enviar mensaje de finalización
    if task_id in active_connections:
        await active_connections[task_id].send_text(json.dumps({
            "type": "complete",
            "output_path": f"processed_{Path(file_path).name}",
            "task_id": task_id
        }))

@app.websocket("/ws/progress/{task_id}")
async def progress_websocket(websocket: WebSocket, task_id: str):
    await websocket.accept()
    active_connections[task_id] = websocket
    
    try:
        while True:
            await websocket.receive_text()  # Mantener conexión activa
    except WebSocketDisconnect:
        print(f"Conexión cerrada para {task_id}")
    finally:
        if task_id in active_connections:
            del active_connections[task_id]

@app.post("/upload/")
async def upload_video(file: UploadFile = File(...), tecnologia: str = Form(...)):
    if not file.filename.lower().endswith(('.mp4', '.avi', '.mov')):
        return {"error": "Formato de video no soportado"}
    
    task_id = str(uuid.uuid4())
    input_path = Path(UPLOAD_DIR) / file.filename
    
    with open(input_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    # Iniciar procesamiento en segundo plano
    asyncio.create_task(process_video_real_time(str(input_path), task_id))
    
    return {"status": "processing", "task_id": task_id}

@app.get("/video/{filename}")
async def get_video(filename: str):
    video_path = Path(PROCESSED_DIR) / filename
    if not video_path.exists():
        return {"error": "Video no encontrado"}
    
    def iterfile():
        with open(video_path, mode="rb") as file_like:
            yield from file_like
    
    return StreamingResponse(iterfile(), media_type="video/mp4")