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
import moviepy as moviepy
from time import time
import asyncio
import json
from typing import Dict
from fastapi.middleware.cors import CORSMiddleware
import yt_dlp
from fastapi import Query, Body

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

async def process_video_real_time(file_path: str, task_id: str, tecnologia: str, modelo: str):
    temp_output_dir = None
    output_writer = None
    
    try:
        # Inicializar modelo
        try:
            if tecnologia == "yolo":
                model = YOLOModel(Path(YOLO_MODEL_PATH) / modelo)
                model.start_metrics()
            else:
                model = MediaPipeObjectDetector(str(Path(MEDIAPIPE_MODEL_PATH) / modelo))
                model.start_metrics()
        except Exception as e:
            print(f"Error inicializando el modelo: {str(e)}")
            await manager.send_message(task_id, {
                "type": "error",
                "message": "Error inicializando el modelo"
            })
            return

        # Configurar captura de video
        cap = cv2.VideoCapture(file_path)
        if not cap.isOpened():
            await manager.send_message(task_id, {
                "type": "error",
                "message": "No se pudo abrir el archivo de video"
            })
            return

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        """ # Crear directorio temporal para el video de salida
        temp_output_dir = Path(PROCESSED_DIR) / f"temp_{task_id}"
        temp_output_dir.mkdir(parents=True, exist_ok=True) """
        
        output_path = Path(PROCESSED_DIR) / "videos" / task_id
        output_path.mkdir(parents=True, exist_ok=True)
        output_file = Path(PROCESSED_DIR) / "videos" / task_id / "processed_output_frames.mp4"
        
        # Configurar escritor de video
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        output_writer = cv2.VideoWriter(
            str(output_file), 
            fourcc, 
            fps, 
            (width, height)
        )

        frame_count = 0
        last_progress_update = 0
        buffer_size = 10  # Número de frames a almacenar en buffer
        frame_buffer = []
        
        # Tiempo para control de envío de frames (evitar saturación)
        last_frame_time = time()
        min_frame_interval = 1 / 30  # Máximo 30 fps para WebSocket
        
        while cap.isOpened():
            # Verificar si la tarea fue cancelada
            if task_id not in active_tasks:
                await manager.send_message(task_id, {"type": "cancelled"})
                break

            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            progress = (frame_count / total_frames) * 100
            
            try:
                # Procesar frame
                if tecnologia == "yolo":
                    processed_frame = model.process_image(frame, frame_count, total_frames)
                else:
                    timestamp_ms = int(frame_count * (1000 / fps))
                    processed_frame = model.process_image(frame, timestamp_ms, frame_count, total_frames)
                
                # Guardar frame procesado para el video final
                if output_writer is not None:
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
                    if abs(progress - last_progress_update) >= 1:
                        await manager.send_message(task_id, {
                            "type": "frame",
                            "frame": frame_base64,
                            "progress": progress,
                            "frame_count": frame_count,
                            "total_frames": total_frames
                        })
                        last_progress_update = progress
                    
                    last_frame_time = current_time
                
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
        cap.release()
        if output_writer is not None:
            output_writer.release()

        # Guardar métricas finales
        final_metrics = model.get_current_metrics() if model else {}
        metrics_path = output_path / "metrics.json"
        with open(metrics_path, 'w') as f:
            json.dump(final_metrics, f)
        
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
    input_path = Path(UPLOAD_DIR) / "videos" / f"{task_id}{Path(file.filename).suffix}"
    
    # Asegurar que el directorio existe
    input_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(input_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    # Crear y almacenar la tarea
    task = asyncio.create_task(process_video_real_time(
        str(input_path), task_id, tecnologia, modelo
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
async def get_video(filename: str):
    video_path = Path(PROCESSED_DIR)/ 'videos' / filename /  'processed_output.mp4'
    if not video_path.exists():
        return {"error": "Video no encontrado"}
    
    def iterfile():
        with open(video_path, mode="rb") as file_like:
            yield from file_like
    
    return StreamingResponse(
        iterfile(), 
        media_type="video/mp4",
        headers={
            "Content-Disposition": f"inline; filename={os.path.basename(filename)}",
            "Accept-Ranges": "bytes",
            "Cache-Control": "no-cache"
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
    task = asyncio.create_task(process_video_real_time(stream_url, task_id, tecnologia, modelo))
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
                        processed = model.process_image(image)
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