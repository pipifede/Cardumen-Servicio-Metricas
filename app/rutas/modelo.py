from fastapi import APIRouter
from fastapi.responses import JSONResponse
from app.config import MODELOSYOLO, MODELOS_MEDIAPIPE


router = APIRouter()

@router.get("/modelos/yolo")
async def get_Yolo_models():
    return JSONResponse({'modelos' : MODELOSYOLO})

@router.get("/modelos/mediapipe")
async def get_mediapipe_models():
    return JSONResponse({'modelos': MODELOS_MEDIAPIPE})





