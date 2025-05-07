# Cardumen-Servicio-Metricas



## Utilizacion (Yolo | MediaPipe)
import asyncio
import websockets
import json

async def send_ws(uri, data):
    async with websockets.connect(uri) as ws:
        await ws.send(json.dumps(data))


asyncio.run(send_ws("ws://127.0.0.1:8000/ws", data))


## Version Python
    Python 3.11.0
## Install
py -m venv venv

.\venv\Scripts\activate.ps1

pip install -r requirements.txt

## Execute API
.\venv\Scripts\activate.ps1

uvicorn app.main:app --reload    


## Execute test.html (Otra terminal)
.\venv\Scripts\activate.ps1

python -m http.server -b 127.0.0.1 8001 

