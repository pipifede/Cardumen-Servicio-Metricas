from fastapi import APIRouter, File, UploadFile, WebSocket, WebSocketException, status, WebSocketDisconnect, Request 
from fastapi.responses import JSONResponse, FileResponse, StreamingResponse
from pathlib import Path
from app.tecnologias.yolo import YOLOModel
from app.config import UPLOAD_DIR, PROCESSED_DIR, YOLO_MODEL_PATH, MODELOSYOLO
import os
import shutil
import uuid
import cv2
import numpy as np
from io import BytesIO
import base64
import glob
import mimetypes
from io import BytesIO
import base64
import glob
import mimetypes
import moviepy
from time import time


router = APIRouter()

def checkModeloYolo(modelo : str):
    if modelo in MODELOSYOLO:
        return True
    return False


def convert_avi_to_mp4(avi_file_path):
    if not os.path.exists(avi_file_path):
        raise FileNotFoundError(avi_file_path)
    t0 = time()
    clip = moviepy.VideoFileClip(avi_file_path)
    path, file_name = os.path.split(avi_file_path)
    output_name = os.path.join(path, 'tmp_' + os.path.splitext(file_name)[0] + '.mp4')
    clip.write_videofile(output_name)
    return output_name
    print('Finished conversion in %is' % (time() - t0))


@router.post("/upload/video")
async def upload_video(request: Request, file: UploadFile = File(...), tecnologia: str = "yolo", modelo: str = MODELOSYOLO[0]):
    # Validar la tecnología (solo YOLO soportada por ahora)
    if tecnologia.lower() != "yolo":
        return {"error": "Tecnología no soportada"}

    if not checkModeloYolo(modelo):
        return {"error": "Modelo no soportado"}

    # Validar el tipo de archivo
    if not file.filename.endswith(('.jpg', '.jpeg', '.png', '.mp4')):
        return {"error": "Tipo de archivo no soportado"}
    
    # Generar un UUID para el archivo
    import uuid
    uuid_str = str(uuid.uuid4())
    uuid_with_extension = f"{uuid_str}{Path(file.filename).suffix}"

    # Guardar el archivo subido
    input_path = Path(UPLOAD_DIR) / "videos" / uuid_with_extension
    input_path.parent.mkdir(parents=True, exist_ok=True)  # Crear directorios si no existen

    with open(input_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Procesar el archivo con la tecnología pasada por parámetro
    output_path = Path(PROCESSED_DIR) / "videos" / uuid_str
    output_path.parent.mkdir(parents=True, exist_ok=True)  # Crear directorios si no existen

    if tecnologia.lower() == "yolo":
        model = YOLOModel(Path(YOLO_MODEL_PATH) / "yolo11n.pt")
        resultado = model.process_video(str(input_path), str(output_path))


    output_path = Path(PROCESSED_DIR) / "videos" / uuid_str / "predict"
    output_files = list(output_path.glob("*"))
    output_file = output_files[0]
    file_path = convert_avi_to_mp4(output_file)
    media_type, _ = mimetypes.guess_type(str(file_path))
    if media_type is None:
        media_type = "application/octet-stream"
    # Devolver el archivo procesado
    file_size = os.path.getsize(file_path)
    range_header = request.headers.get('range')

    def iterfile(start=0, end=None):
        with open(file_path, "rb") as f:
            f.seek(start)
            remaining = end - start + 1 if end is not None else os.path.getsize(file_path) - start

            while remaining > 0:
                chunk_size = min(1024 * 1024, remaining)
                data = f.read(chunk_size)
                if not data:
                    break
                yield data
                remaining -= len(data)

    if range_header:
        # Parsear el encabezado 'Range'
        range_value = range_header.strip().lower().replace("bytes=", "")
        start_str, end_str = range_value.split("-")
        start = int(start_str)
        end = int(end_str) if end_str else file_size - 1

        content_length = end - start + 1

        return StreamingResponse(
            iterfile(start, end),
            status_code=206,
            media_type="video/avi",
            headers={
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Expose-Headers": "Content-Range, Accept-Ranges",
                "Content-Range": f"bytes {start}-{end}/{file_size}",
                "Accept-Ranges": "bytes",
                "Content-Length": str(content_length),
            }
        )
    else:
        return StreamingResponse(
            iterfile(),
            status_code=200,
            media_type="video/avi",
            headers={
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Expose-Headers": "Content-Range, Accept-Ranges",
                "Accept-Ranges": "bytes",
                "Content-Length": str(file_size),
            }
        )
    """ return FileResponse(output_file, media_type=media_type, filename=f"{uuid_str}.avi") """

@router.websocket("/ws/image")
async def websocket_image(websocket: WebSocket, tecnologia: str = "yolo", modelo: str = MODELOSYOLO[0]):
    if not checkModeloYolo(modelo):
        raise WebSocketException(reason="MODELO INVALIDO")
    await websocket.accept()
    print("[WebSocket] Cliente conectado")

    
    # Cargar el modelo YOLO
    if tecnologia.lower() == "yolo":
        model = YOLOModel(YOLO_MODEL_PATH / modelo)
        try:
            while True:
                # Recibir datos del cliente
                data = await websocket.receive_text()

                # Decodificar la imagen en Base64
                image_data = base64.b64decode(data)
                nparr = np.frombuffer(image_data, np.uint8)
                image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

                # Procesar la imagen con YOLO
                processed_image = model.process_image(image)

                # Codificar la imagen procesada en Base64
                _, buffer = cv2.imencode('.jpg', processed_image)
                processed_image_base64 = base64.b64encode(buffer).decode('utf-8')

                # Enviar la imagen procesada al cliente
                await websocket.send_text(processed_image_base64)
        except WebSocketDisconnect:
            print("[WebSocket] Cliente desconectado")