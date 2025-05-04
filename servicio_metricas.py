from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from pydantic import BaseModel
from typing import List
import uvicorn

# 1. Definición de esquema Pydantic para validación automática
class BBox(BaseModel):
    class_name: str
    confidence: float
    bbox: List[float]  # [x, y, w, h]

class DetectionMessage(BaseModel):
    frame_id: int
    timestamp: str
    detections: List[BBox]

app = FastAPI()

# 2. Endpoint REST para recibir POST /detect
@app.post("/detect")
async def receive_detection(msg: DetectionMessage):
    print(f"[REST] Frame {msg.frame_id} @ {msg.timestamp}")
    for det in msg.detections:
        print(f" - {det.class_name} ({det.confidence:.2f}) bbox={det.bbox}")
    return {"status": "ok"}

# 3. Endpoint WebSocket para recibir en tiempo real
@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    print("[WS] Cliente conectado")
    try:
        while True:
            data = await ws.receive_json()
            """ msg = DetectionMessage.parse_obj(data) """
            print(f"[WS] Frame {data}")
            """ for det in msg.detections:
                print(f" - {det.class_name} ({det.confidence:.2f}) bbox={det.bbox}") """
    except WebSocketDisconnect:
        print("[WS] Cliente desconectado")

if __name__ == "__main__":
    uvicorn.run("servicio_metricas:app", host="0.0.0.0", port=8000)