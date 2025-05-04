# Cardumen-Servicio-Metricas



## Utilizacion (Yolo | MediaPipe)
import asyncio
import websockets
import json

async def send_ws(uri, data):
    async with websockets.connect(uri) as ws:
        await ws.send(json.dumps(data))


asyncio.run(send_ws("ws://127.0.0.1:8000/ws", data))


## Execute
py -m venv venv
.\venv\Scripts\activate.ps1
pip install -r requirements.txt
python servicio_metricas.py | uvicorn servicio_metricas:app --reload
