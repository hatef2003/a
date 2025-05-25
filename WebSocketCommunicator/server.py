# server.py
import asyncio
import websockets
import json
from PIL import Image
import io
import base64
async def handle_connection(websocket):
        # Respond with a simple message
    with open("C:\\Users\\Hot-f\\Desktop\\photo_2025-05-13_17-43-47.jpg", "rb") as f:
        image_data = f.read()
    
    i = Image.open(io.BytesIO(image_data))
    # i.show()
    image_base64 = base64.b64encode(image_data).decode('utf-8')
    response = {"type": "response", "payload":image_base64}
    await websocket.send(json.dumps(response))

async def main():
    # Start the server on 0.0.0.0:8125
    async with websockets.serve(handle_connection, "0.0.0.0", 8125):
        print("WebSocket server started on ws://0.0.0.0:8125")
        await asyncio.Future()  # run forever


asyncio.run(main())
