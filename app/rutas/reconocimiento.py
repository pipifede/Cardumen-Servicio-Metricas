from fastapi import APIRouter, File, UploadFile, WebSocket, WebSocketDisconnect, Request, Form 
from fastapi.responses import StreamingResponse, JSONResponse, FileResponse
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
active_connections = {}
active_tasks = {}
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
    output_name = os.path.join(path, 'processed_output.mp4')
    clip.write_videofile(output_name, codec="libx264")
    return output_name

async def process_video_real_time(
    file_path: str, task_id: str, tecnologia: str, modelo: str
):
    try:
        try:
            if tecnologia == "yolo":
                model = YOLOModel(Path(YOLO_MODEL_PATH) / modelo)
            else:
                model = MediaPipeObjectDetector(str(Path(MEDIAPIPE_MODEL_PATH) / modelo))
        except Exception as e:
            print(f"Error inicializando el modelo: {str(e)}")
            if task_id in active_connections:
                await active_connections[task_id].send_text(json.dumps({
                    "type": "error",
                    "message": "Error inicializando el modelo"
                }))
            return

        active_tasks[task_id] = asyncio.current_task()
        cap = cv2.VideoCapture(file_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_duration_ms = 1000 / fps if fps > 0 else 1

        frame_count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            frame_count += 1
            
            if tecnologia == "yolo":
                processed_frame = model.process_image(frame)
            else:
                timestamp_ms = int(frame_count * frame_duration_ms)
                processed_frame = model.process_image(frame, timestamp_ms)
            
            _, buffer = cv2.imencode('.jpg', processed_frame)
            frame_base64 = base64.b64encode(buffer).decode('utf-8')
            
            if task_id in active_connections:
                await active_connections[task_id].send_text(json.dumps({
                    "type": "frame",
                    "frame": frame_base64,
                    "progress": (frame_count / total_frames) * 100,
                    "frame_count": frame_count,
                    "total_frames": total_frames
                }))
            
            # Mantener ritmo real (opcional)
            if fps > 0:
                await asyncio.sleep(1 / fps)
        
        cap.release()
        
        # Enviar mensaje de finalización
        if task_id in active_connections:
            await active_connections[task_id].send_text(json.dumps({
                "type": "complete",
                "output_path": f"processed_{Path(file_path).name}",
                "task_id": task_id
            }))
    except asyncio.CancelledError:
        print(f"Procesamiento cancelado para {task_id}")
        if task_id in active_connections:
            await active_connections[task_id].send_json({
                "type": "error",
                "message": "Proceso cancelado por el usuario"
            })
    finally:
         # Limpieza final segura
        active_tasks.pop(task_id, None)
        active_connections.pop(task_id, None)
        if 'cap' in locals():
            cap.release()
        print(f"Limpieza completada para {task_id}")

@router.post("/upload") ## ENDPOINT PARA IR RECIBIENDO LOS FRAMES A MEDIDA QUE SE PROCESAN (EN PROCESO)
async def upload_video(
    file: UploadFile = File(...),
    tecnologia: str = Form(...),
    modelo: str = Form(...)
):
    if not file.filename.lower().endswith(('.mp4', '.avi', '.mov')):
        return {"error": "Formato de video no soportado"}
    
    if not checkModelo(modelo, tecnologia):
        return JSONResponse(
            {"error": "Modelo no válido para la tecnología seleccionada"},
            status_code=400
        )
    
    task_id = str(uuid.uuid4())
    input_path = Path(UPLOAD_DIR) / file.filename
    
    with open(input_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    asyncio.create_task(process_video_real_time(
        str(input_path), task_id, tecnologia, modelo
    ))
    
    return {"status": "processing", "task_id": task_id}

@router.post("/upload/video")
async def upload_video(request: Request, file: UploadFile = File(...), tecnologia: str = Form("yolo"), modelo: str = Form(MODELOSYOLO[0])):
    print("\n\n✅ TECNOLOGÍA RECIBIDA:", tecnologia)
    print("\n\n✅ MODELO RECIBIDO:", modelo)
    if tecnologia not in ["yolo", "mediapipe"]:
        return {"error": f"Tecnología no soportada: {tecnologia}"}

    if not checkModelo(modelo, tecnologia):
        return {"error": f"Modelo '{modelo}' no es válido para la tecnología '{tecnologia}'"}

    if not checkModelo(modelo, tecnologia):
        return {"error": f"Modelo '{modelo}' no es válido para la tecnología '{tecnologia}'"}
    if not file.filename.endswith(('.mp4', '.avi', '.mov')):
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
        processed_data = model.process_video(str(input_path), str(output_path))
        metrics = processed_data["metrics"]
        avi_file = processed_data["output_path"]
    elif tecnologia == "mediapipe":
        model_path = Path(MEDIAPIPE_MODEL_PATH) / modelo
        model = MediaPipeObjectDetector(str(model_path))
        processed_data = model.process_video(str(input_path), str(output_path))
        metrics = processed_data["metrics"]
        avi_file = processed_data["output_path"]
    file_path = convert_avi_to_mp4(str(avi_file))

    #guardar metricas 
    if (metrics):
        metrics_path = output_path / "metrics.json"
        with open(metrics_path, 'w') as file:
            json.dump(metrics, file)
    # Enviar el video procesado directamente en la respuesta
    def iterfile():
        with open(file_path, "rb") as f:
            while chunk := f.read(1024 * 1024):  # Leer en chunks de 1MB
                yield chunk

    return StreamingResponse(
        iterfile(),
        media_type="video/mp4",
        headers={
            "uuid": uuid_str,
            "Access-Control-Expose-Headers": "Content-Disposition"
        }
    )
@router.get("/videos/{filename}")
async def get_video(filename: str):
    video_path = Path(PROCESSED_DIR)/ 'videos' / filename /  'processed_output.mp4'
    if not video_path.exists():
        return {"error": "Video no encontrado"}
    
    def iterfile():
        with open(video_path, mode="rb") as file_like:
            yield from file_like
    
    return StreamingResponse(iterfile(), media_type="video/mp4")

@router.get("/videos/{filename}/metrics")
async def get_metrics(filename: str):
    metrics_path = Path(PROCESSED_DIR) / "videos" / filename / "metrics.json"
    
    if not metrics_path.exists():
        return JSONResponse(
            status_code=404,
            content={"error": "Métricas no encontradas"}
        )
    
    with open(metrics_path, 'r') as f:
        metrics = json.load(f)
    
    return JSONResponse(content=metrics)

@router.websocket("/ws/progress/{task_id}")
async def progress_websocket(websocket: WebSocket, task_id: str):
    await websocket.accept()
    active_connections[task_id] = websocket
    print(f"Conexión creada para {task_id}")
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        print(f"Conexión cerrada para {task_id}")
    finally:
        # Limpieza segura con verificación de existencia
        active_connections.pop(task_id, None)
        
        # Manejar la tarea asociada
        task = active_tasks.pop(task_id, None)
        if task:
            try:
                task.cancel()
                await task
            except asyncio.CancelledError:
                print(f"Tarea {task_id} cancelada correctamente")
            except Exception as e:
                print(f"Error cancelando tarea {task_id}: {str(e)}")
        
        print(f"Recursos liberados para {task_id}")

@router.websocket("/ws/image")
async def websocket_image(websocket: WebSocket, tecnologia: str = "yolo", modelo: str = MODELOSYOLO[0]):
    await websocket.accept()
    print(f"Conexión WebSocket para {tecnologia} - {modelo}")

    try:
        if tecnologia == "yolo":
            model = YOLOModel(Path(YOLO_MODEL_PATH) / modelo)
        else:
            model = MediaPipeObjectDetector(str(Path(MEDIAPIPE_MODEL_PATH) / modelo))

        timestamp_ms = 0
        while True:
            data = await websocket.receive_text()
            image_data = base64.b64decode(data)
            image = cv2.imdecode(np.frombuffer(image_data, np.uint8), cv2.IMREAD_COLOR)
            
            if tecnologia == "yolo":
                processed = model.process_image(image)
            else:
                timestamp_ms += 1
                processed = model.process_image(image, timestamp_ms)

            _, buffer = cv2.imencode('.jpg', processed)
            await websocket.send_text(base64.b64encode(buffer).decode('utf-8'))
    except WebSocketDisconnect:
        print("Cliente desconectado")
    except Exception as e:
        print(f"Error: {str(e)}")
        await websocket.close(code=1011)