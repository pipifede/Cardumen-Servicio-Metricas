from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse
import shutil
import os
from pathlib import Path

# Simulación del servicio de reconocimiento (YOLO)
def process_with_yolo(input_path: str, output_path: str):
    # Aquí iría la lógica de YOLO para procesar el archivo
    # Por ahora, simplemente copiamos el archivo como simulación
    shutil.copy(input_path, output_path)

app = FastAPI()

UPLOAD_DIR = "uploads"
PROCESSED_DIR = "processed"

# Crear directorios si no existen
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)

@app.post("/upload/")
async def upload_file(file: UploadFile = File(...), tecnologia: str = "yolo"):
    # Validar la tecnología (solo YOLO soportada por ahora)
    if tecnologia.lower() != "yolo":
        return {"error": "Tecnología no soportada"}

    # Validar el tipo de archivo
    if not file.filename.endswith(('.jpg', '.jpeg', '.png', '.mp4')):
        return {"error": "Tipo de archivo no soportado"}
    
    # Guardar el archivo subido
    input_path = Path(UPLOAD_DIR) / file.filename
    with open(input_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Procesar el archivo con la tecnología pasada por parametro
    output_path = Path(PROCESSED_DIR) / file.filename
    if tecnologia.lower() == "yolo":
        process_with_yolo(str(input_path), str(output_path))


    # Devolver el archivo procesado
    return FileResponse(output_path, media_type="application/octet-stream", filename=file.filename)

@app.get("/")
def read_root():
    return {"message": "API de procesamiento de imágenes y videos con YOLO"}