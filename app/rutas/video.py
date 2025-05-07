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
import moviepy
from time import time

router = APIRouter()

CHUNK_SIZE = 1024 * 1024  # 1MB por chunk


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

@router.get("/video/test")
def stream_video(request: Request):
    avi_path = 'app\static\\test.avi'

    if not os.path.exists(avi_path):
        return JSONResponse({"error": "File not found", "path": avi_path})

    # 🔁 Eliminar archivos MP4 viejos (test_*.mp4)
    old_mp4_files = glob.glob('app/static/tmp_*.mp4')
    for old_file in old_mp4_files:
        try:
            os.remove(old_file)
        except Exception as e:
            print(f"No se pudo eliminar {old_file}: {e}")

    # Convertir el archivo AVI a MP4
    

    file_path = convert_avi_to_mp4(avi_path)
    file_size = os.path.getsize(file_path)
    range_header = request.headers.get("range")

    if range_header:
        try:
            range_value = range_header.strip().split("=")[-1]
            range_start, range_end = range_value.split("-")
            start = int(range_start)
            end = int(range_end) if range_end else file_size - 1
            if end >= file_size:
                end = file_size - 1
            if start > end:
                return JSONResponse({"error": "Invalid range"})
        except ValueError:
            return JSONResponse({"error": "Invalid range format"})
    else:
        start = 0
        end = file_size - 1

    def iter_file():
        with open(file_path, "rb") as f:
            f.seek(start)
            remaining = end - start + 1
            while remaining > 0:
                chunk_size = min(CHUNK_SIZE, remaining)
                data = f.read(chunk_size)
                if not data:
                    break
                yield data
                remaining -= len(data)

    mime_type, _ = mimetypes.guess_type(file_path)
    mime_type = mime_type or "application/octet-stream"

    headers = {
        "Content-Range": f"bytes {start}-{end}/{file_size}",
        "Accept-Ranges": "bytes",
        "Content-Length": str(end - start + 1),
        "Content-Type": mime_type,
        "Access-Control-Allow-Origin": "*",
        "Access-Control-Expose-Headers": "Content-Range, Accept-Ranges",
    }

    return StreamingResponse(iter_file(), status_code=206, headers=headers)