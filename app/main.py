from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.rutas import reconocimiento, modelo, video
from app.config import UPLOAD_DIR, PROCESSED_DIR  # Importa desde config.py


app = FastAPI()



# CORS middleware to allow requests from different origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include the recognition routes
app.include_router(reconocimiento.router)
app.include_router(modelo.router)
app.include_router(video.router)

@app.get("/")
def read_root():
    return {"message": "Welcome to the YOLO FastAPI Image Recognition App!"}