from fastapi import APIRouter, File, UploadFile, WebSocket, WebSocketDisconnect, Request, Form, Body
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
import moviepy as moviepy
from time import time
import asyncio
import json
from typing import Dict
from fastapi.middleware.cors import CORSMiddleware
import yt_dlp
from fastapi import Query
from pydantic import BaseModel
from typing import List
from fastapi import HTTPException 
from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
    status,
    UploadFile,
    File,
    Form,
    WebSocket,
    Request,
    Body
)
from fastapi.responses import JSONResponse, StreamingResponse

COLOR_PALETTE = [
  "#1f77b4", "#4e79a7", "#5c8fd8", "#7eb0d5", "#a6cee3",
  "#e377c2", "#d62728", "#ff7f0e", "#ff9da7", "#ff9896",
  "#2ca02c", "#8c564b", "#98df8a", "#17becf", "#bcbd22",
  "#f7b6d2", "#ffbb78", "#c5b0d5", "#c49c94", "#dbdb8d",
  "#9edae5", "#393b79", "#637939", "#843c39", "#e7ba52",
	"#ad494a", "#a55194", "#6b6ecf"
];

def get_color_for_id(box_id: int) -> str:
    """Implementación idéntica a la del frontend para asignación de colores"""
    str_id = str(box_id)
    hash_val = 0
    for char in str_id:
        hash_val = (hash_val << 5) - hash_val + ord(char)
        hash_val = hash_val & 0xFFFFFFFF  # Convertir a entero de 32 bits
    
    index = abs(hash_val) % len(COLOR_PALETTE)
    return COLOR_PALETTE[index]

class BoxTrajectory(BaseModel):
    id: int
    x: float
    y: float

class FrameTrajectory(BaseModel):
    frame: int
    boxes: List[BoxTrajectory]

class TrajectoryResponse(BaseModel):
    detections: List[FrameTrajectory]

router = APIRouter()

# Estructuras para manejar conexiones y tareas activas
active_connections: Dict[str, WebSocket] = {}
active_tasks: Dict[str, asyncio.Task] = {}
processed_videos: Dict[str, str] = {}  # Mapeo de task_id -> ruta del video procesado

class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
        self.last_frame: Dict[str, str] = {}  # Almacena el último frame para cada conexión

    async def connect(self, task_id: str, websocket: WebSocket):
        await websocket.accept()
        self.active_connections[task_id] = websocket
        self.last_frame[task_id] = None

    def disconnect(self, task_id: str):
        if task_id in self.active_connections:
            del self.active_connections[task_id]
        if task_id in self.last_frame:
            del self.last_frame[task_id]

    async def send_message(self, task_id: str, message: dict):
        if task_id in self.active_connections:
            try:
                await self.active_connections[task_id].send_json(message)
            except Exception as e:
                print(f"Error enviando mensaje a {task_id}: {str(e)}")
                self.disconnect(task_id)

    # Guardar último frame para enviarlo en caso de procesamiento lento
    def save_last_frame(self, task_id: str, frame_base64: str):
        if task_id in self.active_connections:
            self.last_frame[task_id] = frame_base64

    # Obtener último frame guardado
    def get_last_frame(self, task_id: str):
        return self.last_frame.get(task_id)

manager = ConnectionManager()
def checkModelo(modelo: str, tecnologia: str):
    if tecnologia == "yolo":
        return modelo in MODELOSYOLO
    elif tecnologia == "mediapipe":
        return modelo in MODELOS_MEDIAPIPE
    return False

def convert_avi_to_mp4(avi_file_path, task_id=None):
    if not os.path.exists(avi_file_path):
        raise FileNotFoundError(avi_file_path)
    
    try:
        clip = moviepy.VideoFileClip(avi_file_path)
        path, file_name = os.path.split(avi_file_path)
        output_name = os.path.join(path, f'processed_output.mp4')
        
        # Usar un bitrate más alto para mejor calidad
        clip.write_videofile(output_name, codec="libx264", bitrate="8000k", 
                            fps=clip.fps, threads=4, preset='fast')
        
        # Registrar el video procesado si hay un task_id
        if task_id:
            processed_videos[task_id] = output_name
            
        return output_name
    except Exception as e:
        print(f"Error converting video: {str(e)}")
        raise


import requests

def mjpeg_frame_generator(url: str):
    """
    Lee un MJPEG multipart stream y produce frames OpenCV.
    """
    stream = requests.get(url, stream=True)
    bytes_buffer = b""
    
    for chunk in stream.iter_content(chunk_size=1024):
        bytes_buffer += chunk
        a = bytes_buffer.find(b'\xff\xd8')  # JPEG start
        b = bytes_buffer.find(b'\xff\xd9')  # JPEG end
        if a != -1 and b != -1:
            jpg = bytes_buffer[a:b+2]
            bytes_buffer = bytes_buffer[b+2:]
            frame = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)
            yield frame

def resize_with_aspect_ratio(original_width, original_height, max_width, max_height):
    aspect_ratio = original_width / original_height

    # Ajustamos al alto máximo primero
    new_height = min(original_height, max_height)
    new_width = int(new_height * aspect_ratio)

    # Si nos pasamos del ancho, corregimos
    if new_width > max_width:
        new_width = max_width
        new_height = int(new_width / aspect_ratio)

    return new_width, new_height

def serialize_boxes(boxes):
    """Convert YOLO boxes object to a JSON-serializable format"""
    if boxes is None:
        return None
    
    serialized = []
    for box in boxes:
        box_data = {
            'xyxy': box.xyxy.tolist()[0],  # Convert numpy array to list
            'conf': float(box.conf),  # Convert numpy float to Python float
            'cls': int(box.cls),  # Convert numpy int to Python int
            'id': int(box.id) if box.id is not None else None
        }
        serialized.append(box_data)
    return serialized

def calculate_trajectories(boxes_data):
    """
    Procesa los datos de cajas para calcular trayectorias.
    Devuelve un diccionario donde las claves son los IDs y los valores son listas de puntos (x,y,frame).
    """
    trajectories = {}
    for frame_data in boxes_data.get('detections', []):
        frame_num = frame_data['frame_number']
        for box in frame_data.get('boxes', []):
            if box.get('id') is not None:
                box_id = box['id']
                x1, y1, x2, y2 = box['xyxy']
                center_x = (x1 + x2) / 2
                center_y = (y1 + y2) / 2
                
                if box_id not in trajectories:
                    trajectories[box_id] = []
                trajectories[box_id].append((center_x, center_y, frame_num))
    return trajectories

async def process_video_real_time(file_path: str, task_id: str, tecnologia: str, modelo: str, new_fps:str, new_res:str):
    temp_output_dir = None
    output_writer = None
    all_boxes = []  # List to store boxes for each frame
    
    try:
        # Inicializar modelo
        try:
            if tecnologia == "yolo":
                model = YOLOModel(Path(YOLO_MODEL_PATH) / modelo)
            elif tecnologia == "mediapipe":
                model = MediaPipeObjectDetector(str(Path(MEDIAPIPE_MODEL_PATH) / modelo))
            else:
                raise ValueError(f"Tecnología no reconocida: {tecnologia}")
            model.start_metrics()
        except Exception as e:
            print(f"Error inicializando el modelo: {str(e)}")
            await manager.send_message(task_id, {
                "type": "error",
                "message": "Error inicializando el modelo"
            })
            return

        # Configurar captura de video
        is_mjpeg = (
            file_path.startswith("http")
            and any(ext in file_path for ext in [".mjpg", ".cgi", ".jpg", "faststream", "GetOneShot"])
        )


        if is_mjpeg:
            cap = None
            frame_generator = mjpeg_frame_generator(file_path)
            fps = 15 
            width, height = 640, 480 
            total_frames = None 
        else:
            cap = cv2.VideoCapture(file_path)
            if not cap.isOpened():
                await manager.send_message(task_id, {
                    "type": "error",
                    "message": "No se pudo abrir el archivo de video"
                })
                return
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            aspect_ratio = width/height
            
        """ # Crear directorio temporal para el video de salida
        temp_output_dir = Path(PROCESSED_DIR) / f"temp_{task_id}"
        temp_output_dir.mkdir(parents=True, exist_ok=True) """
        
        output_path = Path(PROCESSED_DIR) / "videos" / task_id
        output_path.mkdir(parents=True, exist_ok=True)
        output_file = Path(PROCESSED_DIR) / "videos" / task_id / "processed_output_frames.mp4"
                
        
        #Si new_fps es 0 i es mayor que los FPS originales, no se modifica nada
        if new_fps == "0" or int(new_fps) > fps: 
            new_fps = fps
        
        if new_res != "0":
            new_width, new_height = new_res.split("x")
            new_width = int(new_width)
            new_height = int(new_height)
            new_width, new_height = resize_with_aspect_ratio(width, height, new_width, new_height)
        else:
            new_width = width
            new_height = height

        new_fps = float(new_fps)
        
        # Configurar escritor de video
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        output_writer = cv2.VideoWriter(
            str(output_file), 
            fourcc, 
            new_fps, 
            (new_width, new_height)
        )

        frame_count = 0
        last_progress_update = 0
        buffer_size = 10  # Número de frames a almacenar en buffer
        frame_buffer = []
        
        # Tiempo para control de envío de frames (evitar saturación)
        last_frame_time = time()
        min_frame_interval = 1 / 30  # Máximo 30 fps para WebSocket
        
        while True:
            # Verificar si la tarea fue cancelada
            if task_id not in active_tasks:
                raise asyncio.CancelledError(f"Tarea cancelada por el cliente: {task_id}")

            if is_mjpeg:
                try:
                    frame = next(frame_generator)
                except StopIteration:
                    break
            else:
                ret, frame = cap.read()
                if not ret:
                    break


            frame_count += 1
            progress = (frame_count / total_frames) * 100 if total_frames else 0
            
            try:
                if task_id not in active_tasks:
                    await manager.send_message(task_id, {"type": "cancelled"})
                    break
                # Procesar frame
                if tecnologia == "yolo":
                    result = model.process_image(frame, frame_count, total_frames)
                    processed_frame = result["processed_frame"]
                    results = result["results"]
                    # Store boxes data
                    frame_boxes = serialize_boxes(results[0].boxes)
                    all_boxes.append({
                        'frame_number': frame_count,
                        'boxes': frame_boxes
                    })
                else:
                    timestamp_ms = int(frame_count * (1000 / fps))
                    processed_frame = model.process_image(frame, timestamp_ms, frame_count, total_frames)
                
                # Guardar frame procesado para el video final
                if output_writer is not None:
                    #Rescala el frame procesado segun la nueva res para guardarlo bien.
                    if processed_frame.shape[1] != new_width or processed_frame.shape[0] != new_height:
                        processed_frame = cv2.resize(processed_frame, (new_width, new_height))
                    output_writer.write(processed_frame)
                
                # Añadir al buffer circular
                if len(frame_buffer) >= buffer_size:
                    frame_buffer.pop(0)
                frame_buffer.append(processed_frame)
                
                # Enviar frame cada cierto intervalo para evitar saturación
                current_time = time()
                if current_time - last_frame_time >= min_frame_interval:
                    # Codificar frame para enviar
                    _, buffer = cv2.imencode('.jpg', processed_frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                    frame_base64 = base64.b64encode(buffer).decode('utf-8')
                    
                    # Guardar este frame como el último conocido
                    manager.save_last_frame(task_id, frame_base64)
                    
                    # Enviar actualización solo si el progreso ha cambiado significativamente
                    if total_frames is None or abs(progress - last_progress_update) >= 1:
                        await manager.send_message(task_id, {
                            "type": "frame",
                            "frame": frame_base64,
                            "progress": progress,
                            "frame_count": frame_count,
                            "total_frames": total_frames
                        })
                        last_progress_update = progress
                    
                    last_frame_time = current_time
                    await asyncio.sleep(0)
                
                # Pequeña pausa para permitir otras operaciones asíncronas
                if frame_count % 10 == 0:  # Cada 10 frames
                    await asyncio.sleep(0.001)  # Pausa mínima

            except Exception as e:
                print(f"Error procesando frame: {str(e)}")
                # Intentar recuperarse usando el último frame del buffer
                if frame_buffer:
                    # Usar el último frame procesado correctamente
                    processed_frame = frame_buffer[-1]
                    if output_writer is not None:
                        output_writer.write(processed_frame)
                continue

        # Cerrar y liberar recursos
        if cap is not None:
            cap.release()
        if output_writer is not None:
            output_writer.release()

        # Guardar métricas finales
        final_metrics = model.get_current_metrics() if model else {}
        
        final_metrics["original_fps"] = fps
        final_metrics["processed_fps"] = new_fps
        final_metrics["original_resolution"] = {"width": width, "height": height}
        final_metrics["processed_resolution"] = {"width": new_width, "height": new_height}
        
        metrics_path = output_path / "metrics.json"
        with open(metrics_path, 'w') as f:
            json.dump(final_metrics, f)
        
        # Save boxes data alongside the video
        if tecnologia == "yolo" and all_boxes:
            boxes_path = output_path / "detection_boxes.json"
            with open(boxes_path, 'w') as f:
                json.dump({
                    'model': modelo,
                    'total_frames': total_frames,
                    'fps': fps,
                    'detections': all_boxes
                }, f)
        
        # Convertir video final a mp4 de alta calidad
        if output_file.exists():
            try:
                final_output = convert_avi_to_mp4(str(output_file), task_id)
                relative_path = relative_path = "videos/" + task_id
                
                # Enviar mensaje de finalización con ruta relativa
                if task_id in active_tasks:
                    await manager.send_message(task_id, {
                        "type": "complete",
                        "output_path": relative_path,
                        "metrics": final_metrics,
                        "task_id": task_id
                    })
            except Exception as e:
                print(f"Error en conversión final: {str(e)}")
                await manager.send_message(task_id, {
                    "type": "error",
                    "message": f"Error en la conversión final: {str(e)}"
                })

    except asyncio.CancelledError:
        print(f"Procesamiento cancelado para {task_id}")
        await manager.send_message(task_id, {
            "type": "cancelled",
            "message": "Proceso cancelado por el usuario"
        })
    except Exception as e:
        print(f"Error en process_video_real_time: {str(e)}")
        await manager.send_message(task_id, {
            "type": "error",
            "message": f"Error en el procesamiento: {str(e)}"
        })
    finally:
        # Limpieza final segura
        if task_id in active_tasks:
            del active_tasks[task_id]
        manager.disconnect(task_id)
        if 'cap' in locals() and cap is not None:
            cap.release()
        if output_writer is not None:
            output_writer.release()
        print(f"Limpieza completada para {task_id}")

@router.post("/upload")
async def upload_video(
    file: UploadFile = File(...),
    tecnologia: str = Form(...),
    modelo: str = Form(...),
    fps: str = Form(...),
    res: str = Form(...)
):
    if not file.filename.lower().endswith(('.mp4', '.avi', '.mov')):
        return {"error": "Formato de video no soportado"}
    
    if not checkModelo(modelo, tecnologia):
        return JSONResponse(
            {"error": "Modelo no válido para la tecnología seleccionada"},
            status_code=400
        )
    task_id = str(uuid.uuid4())
    input_path = Path(UPLOAD_DIR) / "videos" / f"{task_id}{Path(file.filename).suffix}"
    
    # Asegurar que el directorio existe
    input_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(input_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Crear y almacenar la tarea
    task = asyncio.create_task(process_video_real_time(
        str(input_path), task_id, tecnologia, modelo, fps, res
    ))
    active_tasks[task_id] = task
    
    return {"status": "processing", "task_id": task_id}

@router.post("/upload/video")
async def upload_video(request: Request, file: UploadFile = File(...), tecnologia: str = Form("yolo"), modelo: str = Form(MODELOSYOLO[0])):
    print("\n\n✅ TECNOLOGÍA RECIBIDA:", tecnologia)
    print("\n\n✅ MODELO RECIBIDO:", modelo)
    if tecnologia not in ["yolo", "mediapipe"]:
        return {"error": f"Tecnología no soportada: {tecnologia}"}

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
    try:
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
    except Exception as e:
        return JSONResponse(
            {"error": f"Error procesando video: {str(e)}"},
            status_code=500
        )
@router.get("/videos/{filename}")
async def get_video(filename: str, request: Request):
    video_path = Path(PROCESSED_DIR)/ 'videos' / filename /  'processed_output.mp4'
    if not video_path.exists():
        return {"error": "Video no encontrado"}
    
    file_size = video_path.stat().st_size
    range_header = request.headers.get('range')
    
    # Common headers for both range and non-range requests
    common_headers = {
        "Accept-Ranges": "bytes",
        "Content-Disposition": f"inline; filename={os.path.basename(filename)}",
        "Cache-Control": "no-cache",
        "Access-Control-Allow-Origin": "*",
        "Access-Control-Allow-Headers": "Range, Content-Type, Accept",
        "Access-Control-Expose-Headers": "Content-Range, Content-Length, Accept-Ranges",
        "Content-Type": "video/mp4; codecs=\"avc1.42E01E, mp4a.40.2\""
    }
    
    if range_header:
        try:
            # Parse range header
            range_type, range_value = range_header.split('=')
            if range_type != 'bytes':
                return JSONResponse(
                    status_code=400,
                    content={"error": "Invalid range type"}
                )
            
            # Parse range values
            start, end = range_value.split('-')
            start = int(start) if start else 0
            end = int(end) if end else file_size - 1
            
            # Validate range
            if start >= file_size or end >= file_size or start > end:
                return JSONResponse(
                    status_code=416,
                    content={"error": "Requested range not satisfiable"}
                )
            
            # Calculate content length
            content_length = end - start + 1
            
            async def range_response():
                with open(video_path, 'rb') as file:
                    file.seek(start)
                    remaining = content_length
                    while remaining > 0:
                        chunk_size = min(8192, remaining)
                        chunk = file.read(chunk_size)
                        if not chunk:
                            break
                        yield chunk
                        remaining -= len(chunk)
            
            return StreamingResponse(
                range_response(),
                media_type="video/mp4",
                headers={
                    **common_headers,
                    "Content-Range": f"bytes {start}-{end}/{file_size}",
                    "Content-Length": str(content_length),
                },
                status_code=206
            )
            
        except Exception as e:
            return JSONResponse(
                status_code=400,
                content={"error": f"Invalid range request: {str(e)}"}
            )
    else:
        # No range header, return full file
        async def full_response():
            with open(video_path, 'rb') as file:
                while chunk := file.read(8192):
                    yield chunk
        
        return StreamingResponse(
            full_response(),
            media_type="video/mp4",
            headers={
                **common_headers,
                "Content-Length": str(file_size),
            }
        )

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

@router.post("/cancel/{task_id}")
async def cancel_processing(task_id: str):
    task = active_tasks.get(task_id)
    if task:
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        return {"status": "cancelled", "task_id": task_id}
    return {"status": "task_not_found", "task_id": task_id}

@router.websocket("/ws/progress/{task_id}")
async def progress_websocket(websocket: WebSocket, task_id: str):
    await manager.connect(task_id, websocket)
    print(f"Conexión WebSocket establecida para {task_id}")
    
    try:
        # Verificar si hay un video procesado para este task_id
        if task_id in processed_videos:
            video_path = processed_videos[task_id]
            relative_path = "videos/" + task_id
            
            # Enviar mensaje al cliente con la ruta del video procesado
            await websocket.send_json({
                "type": "complete",
                "output_path": relative_path,
                "task_id": task_id
            })
        
        # Mantener la conexión abierta
        while True:
            data = await websocket.receive_text()
            data = json.loads(data)
            # Si el cliente envía "ping", responder con el último frame
            if data == "ping":
                last_frame = manager.get_last_frame(task_id)
                if last_frame:
                    await websocket.send_json({
                        "type": "frame",
                        "frame": last_frame
                    })
            
    except WebSocketDisconnect:
        print(f"Conexión WebSocket cerrada para {task_id}")
    except Exception as e:
        print(f"Error en WebSocket {task_id}: {str(e)}")
    finally:
        manager.disconnect(task_id)
        print(f"Recursos liberados para {task_id}")
        if task_id in active_tasks:
            print(f"[WS] Cliente se desconectó, cancelando tarea {task_id}")
            active_tasks[task_id].cancel()
            del active_tasks[task_id]


@router.post("/upload/stream")
async def upload_stream(
    stream_url: str = Form(...),
    tecnologia: str = Form(...),
    modelo: str = Form(...)
):
    if not checkModelo(modelo, tecnologia):
        return JSONResponse({"error": "Modelo no válido para la tecnología seleccionada"}, status_code=400)
    
    task_id = str(uuid.uuid4())
    # Lanzar la tarea de análisis en background con la URL del stream
    task = asyncio.create_task(process_video_real_time(stream_url, task_id, tecnologia, modelo, "0", "0"))
    active_tasks[task_id] = task

    return {"status": "processing", "task_id": task_id}


@router.get("/get_youtube_stream_url")
async def get_youtube_stream_url(youtube_url: str = Query(...)):

    ydl_opts = {
        "format": "best[ext=mp4]/best",
        "quiet": True,
        "no_warnings": True,
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(youtube_url, download=False)
            # Buscar la URL del stream
            formats = info.get("formats", [])
            stream_url = None
            # Intentar obtener el URL de formato HLS (m3u8) o el mejor mp4
            for f in formats:
                if f.get("protocol") == "m3u8_native":
                    stream_url = f.get("url")
                    break
            if not stream_url and formats:
                stream_url = formats[-1].get("url")  # fallback al último formato
            if not stream_url:
                return {"error": "No se encontró URL de stream para esta transmisión"}

            return {"stream_url": stream_url}

    except Exception as e:
        return {"error": str(e)}

@router.websocket("/ws/image")
async def websocket_image(
    websocket: WebSocket, 
    tecnologia: str = "yolo", 
    modelo: str = MODELOSYOLO[0],
    max_latency: int = 500
):
    await websocket.accept()
    print(f"Conexión WebSocket para tiempo real: {tecnologia} - {modelo}")

    try:
        # Inicializar modelo
        if tecnologia == "yolo":
            model = YOLOModel(Path(YOLO_MODEL_PATH) / modelo)
        else:
            model = MediaPipeObjectDetector(str(Path(MEDIAPIPE_MODEL_PATH) / modelo))

        timestamp_ms = 0
        last_frame = None
        frame_times = []  # Lista para almacenar los últimos tiempos de frame
        
        while True:
            # Recibir frame del cliente
            data = await websocket.receive_text()
            data = json.loads(data)
            current_time = time()
            
            if current_time - data["timestamp"] < (max_latency / 1000):
                # Calcular FPS
                frame_times.append(current_time)
                if len(frame_times) > 30:  # Mantener solo los últimos 30 frames para el cálculo
                    frame_times.pop(0)
                
                if len(frame_times) > 1:
                    fps = len(frame_times) / (frame_times[-1] - frame_times[0])
                else:
                    fps = 0
                
                # Procesar imagen
                try:
                    image_data = base64.b64decode(data["frame"])
                    image = cv2.imdecode(np.frombuffer(image_data, np.uint8), cv2.IMREAD_COLOR)
                    
                    if tecnologia == "yolo":
                        result = model.process_image(image)
                        processed = result["processed_frame"]
                    else:
                        timestamp_ms += 1
                        processed = model.process_image(image, timestamp_ms)

                    # Comprimir con mejor calidad
                    _, buffer = cv2.imencode('.jpg', processed, [cv2.IMWRITE_JPEG_QUALITY, 85])
                    frame_base64 = base64.b64encode(buffer).decode('utf-8')
                    
                    # Guardar este frame como el último conocido
                    last_frame = frame_base64
                    
                    # Enviar frame procesado junto con FPS
                    await websocket.send_json({
                        "frame": frame_base64,
                        "fps": round(fps, 2)
                    })
                except Exception as e:
                    print(f"Error procesando frame: {str(e)}")
                    
                    # Si hay un error pero tenemos un frame anterior, enviar ese
                    if last_frame is not None:
                        await websocket.send_json({
                            "frame": last_frame,
                            "fps": round(fps, 2)
                        })
                    else:
                        await websocket.send_json({
                            "type": "error",
                            "message": f"Error procesando frame: {str(e)}"
                        })
                    continue
            else:
                print("Skipping frame due to latency")
                continue

    except WebSocketDisconnect:
        print("Cliente desconectado")
    except Exception as e:
        print(f"Error en WebSocket: {str(e)}")
        await websocket.close(code=1011)
    finally:
        print("Conexión WebSocket tiempo real cerrada")

@router.get("/videos/{filename}/boxes")
async def get_detection_boxes(filename: str):
    boxes_path = Path(PROCESSED_DIR) / "videos" / filename / "detection_boxes.json"
    
    if not boxes_path.exists():
        return JSONResponse(
            status_code=404,
            content={"error": "Detection boxes not found"}
        )
    
    with open(boxes_path, 'r') as f:
        boxes_data = json.load(f)
    
    return JSONResponse(content=boxes_data)

@router.get("/videos/{task_id}/box_ids")
async def get_box_ids(task_id: str):
    boxes_path = Path(PROCESSED_DIR) / "videos" / task_id / "detection_boxes.json"
    
    if not boxes_path.exists():
        return JSONResponse(
            status_code=404,
            content={"error": "Detection boxes not found"}
        )
    
    try:
        with open(boxes_path, 'r') as f:
            boxes_data = json.load(f)
        
        # Extract all box IDs from all frames
        box_ids = set()
        for detection in boxes_data.get('detections', []):
            for box in detection.get('boxes', []):
                if box.get('id') is not None:
                    box_ids.add(box['id'])
        print("[DEBUG] Box IDs", box_ids)
        # Convert set to sorted list
        return JSONResponse(content={"box_ids": sorted(list(box_ids))})
    
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": f"Error processing box IDs: {str(e)}"}
        )

async def render_video_with_box_id(task_id: str, box_ids: list[int], show_trajectory: bool = False):
    """Re-render video showing only the specified box IDs"""
    try:
        # Load boxes data
        boxes_path = Path(PROCESSED_DIR) / "videos" / task_id / "detection_boxes.json"
        if not boxes_path.exists():
            raise FileNotFoundError("Boxes data not found")
        
        with open(boxes_path, 'r') as f:
            boxes_data = json.load(f)
        
        # Find original video
        upload_dir = Path(UPLOAD_DIR) / "videos"
        original_video = None
        for file in upload_dir.glob(f"{task_id}.*"):
            if file.suffix.lower() in ['.mp4', '.avi', '.mov']:
                original_video = file
                break
        
        if not original_video:
            raise FileNotFoundError("Original video not found")
        
        # Create output directory
        output_path = Path(PROCESSED_DIR) / "videos" / task_id / "filtered"
        output_path.mkdir(parents=True, exist_ok=True)
        # Create filename with all box IDs
        box_ids_str = "_".join(map(str, sorted(box_ids)))
        output_file = output_path / f"boxes_{box_ids_str}_video.mp4"
        
        # Open video
        cap = cv2.VideoCapture(str(original_video))
        if not cap.isOpened():
            raise Exception("Could not open video")
        
        # Calcular trayectorias si se solicita
        trajectories = {}
        if show_trajectory:
            trajectories = calculate_trajectories(boxes_data)

        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Create video writer with H.264 codec
        fourcc = cv2.VideoWriter_fourcc(*'avc1')  # Use H.264 codec
        out = cv2.VideoWriter(str(output_file), fourcc, fps, (width, height))
        
        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Dibujar trayectorias primero (para que queden detrás de las cajas)
            if show_trajectory:
                for box_id in box_ids:
                    if box_id in trajectories:
                        path = trajectories[box_id]
                        # Obtener color consistente para este ID
                        color = get_color_for_id(box_id)
                        # Convertir color HEX a BGR (OpenCV usa BGR)
                        bgr_color = (
                            int(color[5:7], 16),  # B
                            int(color[3:5], 16),  # G
                            int(color[1:3], 16)   # R
                        )
                        # Dibujar línea conectando los puntos
                        for i in range(1, len(path)):
                            if path[i][2] <= frame_count + 1:  # +1 porque frame_count es 0-based
                                cv2.line(frame, 
                                        (int(path[i-1][0]), int(path[i-1][1])),
                                        (int(path[i][0]), int(path[i][1])),
                                        bgr_color, 2)  # Usar color consistente

            # Find boxes for this frame
            frame_boxes = None
            for detection in boxes_data['detections']:
                if detection['frame_number'] == frame_count + 1:  # +1 because frame_count is 0-based
                    frame_boxes = detection['boxes']
                    break
            
            if frame_boxes:
                # Filter boxes for the specified IDs
                filtered_boxes = [box for box in frame_boxes if box.get('id') in box_ids]
                
                # Draw only the filtered boxes
                for box in filtered_boxes:
                    box_id = box['id']
                    color = get_color_for_id(box_id)
                    bgr_color = (
                        int(color[5:7], 16),  # B
                        int(color[3:5], 16),  # G
                        int(color[1:3], 16)   # R
                    )
                    
                    x1, y1, x2, y2 = map(int, box['xyxy'])
                    cv2.rectangle(frame, (x1, y1), (x2, y2), bgr_color, 2)
                    center_x = (x1 + x2) // 2
                    center_y = (y1 + y2) // 2
                    label = f"ID {box['id']} ({box['conf']*100:.1f}%) ({center_x},{center_y})"
                    # Get text size for background
                    (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                    # Draw background rectangle con el mismo color
                    cv2.rectangle(frame, (x1, y1 - text_height - 10), (x1 + text_width, y1), bgr_color, -1)
                    # Draw the original label with original color
                    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            out.write(frame)
            frame_count += 1
        
        # Cleanup
        cap.release()
        out.release()
        
        return str(output_file)
    
    except Exception as e:
        print(f"Error in render_video_with_box_id: {str(e)}")
        raise

@router.get("/videos/{task_id}/trajectories", response_model=TrajectoryResponse)
async def get_trajectories_data(task_id: str):
    """
    Endpoint para obtener datos de trayectorias en formato JSON
    """
    try:
        # 1. Verificar si existe el análisis
        boxes_path = Path(PROCESSED_DIR) / "videos" / task_id / "detection_boxes.json"
        
        if not boxes_path.exists():
            raise HTTPException(
                status_code=404, 
                detail="No se encontraron datos de detección para esta tarea"
            )

        # 2. Cargar los datos existentes
        with open(boxes_path, 'r') as f:
            boxes_data = json.load(f)

        # 3. Procesar para el formato de trayectorias
        detections = []
        for frame_data in boxes_data.get('detections', []):
            frame_entry = {
                "frame": frame_data['frame_number'],
                "boxes": []
            }
            
            for box in frame_data.get('boxes', []):
                if box.get('id') is not None:
                    # Calcular centro de la caja
                    x1, y1, x2, y2 = box['xyxy']
                    center_x = (x1 + x2) / 2
                    center_y = (y1 + y2) / 2
                    
                    frame_entry['boxes'].append({
                        "id": box['id'],
                        "x": center_x,
                        "y": center_y
                    })
            
            if frame_entry['boxes']:  # Solo añadir frames con cajas
                detections.append(frame_entry)

        return {"detections": detections}

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error al obtener trayectorias: {str(e)}"
        )

@router.post("/videos/{task_id}/boxes")
async def get_video_with_boxes(task_id: str, request: Request, box_ids: list[int] = Body(...), show_trajectory: bool = Body(False)):
    """Endpoint to get video showing only specific box IDs"""
    try:
        output_file = await render_video_with_box_id(task_id, box_ids, show_trajectory)
        
        # Get file size for range requests
        file_size = Path(output_file).stat().st_size
        
        # Common headers for both range and non-range requests
        common_headers = {
            "Accept-Ranges": "bytes",
            "Content-Disposition": f"inline; filename=boxes_{'_'.join(map(str, sorted(box_ids)))}_video.mp4",
            "Cache-Control": "no-cache",
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Headers": "Range, Content-Type, Accept",
            "Access-Control-Expose-Headers": "Content-Range, Content-Length, Accept-Ranges",
            "Content-Type": "video/mp4; codecs=\"avc1.42E01E, mp4a.40.2\""
        }
        
        # Check for range header
        range_header = request.headers.get('range')
        if range_header:
            try:
                # Parse range header
                range_type, range_value = range_header.split('=')
                if range_type != 'bytes':
                    return JSONResponse(
                        status_code=400,
                        content={"error": "Invalid range type"}
                    )
                
                # Parse range values
                start, end = range_value.split('-')
                start = int(start) if start else 0
                end = int(end) if end else file_size - 1
                
                # Validate range
                if start >= file_size or end >= file_size or start > end:
                    return JSONResponse(
                        status_code=416,
                        content={"error": "Requested range not satisfiable"}
                    )
                
                # Calculate content length
                content_length = end - start + 1
                
                async def range_response():
                    with open(output_file, 'rb') as file:
                        file.seek(start)
                        remaining = content_length
                        while remaining > 0:
                            chunk_size = min(8192, remaining)
                            chunk = file.read(chunk_size)
                            if not chunk:
                                break
                            yield chunk
                            remaining -= len(chunk)
                
                return StreamingResponse(
                    range_response(),
                    media_type="video/mp4",
                    headers={
                        **common_headers,
                        "Content-Range": f"bytes {start}-{end}/{file_size}",
                        "Content-Length": str(content_length),
                    },
                    status_code=206
                )
                
            except Exception as e:
                return JSONResponse(
                    status_code=400,
                    content={"error": f"Invalid range request: {str(e)}"}
                )
        else:
            # No range header, return full file
            async def full_response():
                with open(output_file, 'rb') as file:
                    while chunk := file.read(8192):
                        yield chunk
            
            return StreamingResponse(
                full_response(),
                media_type="video/mp4",
                headers={
                    **common_headers,
                    "Content-Length": str(file_size),
                }
            )
    
    except FileNotFoundError as e:
        return JSONResponse(
            status_code=404,
            content={"error": str(e)}
        )
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": f"Error processing video: {str(e)}"}
        )

