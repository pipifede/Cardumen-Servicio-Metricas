from fastapi import APIRouter
from fastapi.responses import JSONResponse
from app.config import MODELOSYOLO

router = APIRouter()

@router.get("/modelos/yolo")
async def get_Yolo_models():
    return JSONResponse({'modelos' : MODELOSYOLO})




