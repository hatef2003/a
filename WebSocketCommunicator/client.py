import asyncio
import json
from WebSocketCommunicator import WebSocketWrapper
from PIL import Image
import io
import base64
async def main():
    ws = WebSocketWrapper("192.168.43.8", 8125)
    await ws.connect()

    # Send JSON
    # await ws.send(json.dumps({"type": "greeting", "content": "Hello Server!"}))
    i = 0
    while (1):
        print(i)
        response = await ws.receive_json()

        image = response["payload"]
        image_bytes = base64.b64decode(image)
        # image= image.encode("utf-8")
        image = Image.open(io.BytesIO(image_bytes))

        image.show("./a.jpg")
        i+=1

    # await ws.close()

asyncio.run(main())
