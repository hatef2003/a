import asyncio
import json
from WebSocketCommunicator import WebSocketWrapper
from PIL import Image
import io
import base64
async def main():
    ws = WebSocketWrapper("192.168.43.249", 8125)
    await ws.connect()

    # Send JSON
    # await ws.send(json.dumps({"type": "greeting", "content": "Hello Server!"}))
    i = 0
    while (1):
        print(i)
        response = await ws.receive()

        # image = response["payload"]
        # print(len(image))
        # image_bytes = base64.b64decode(image)
        # # image= image.encode("utf-8")
        # image = Image.open(io.BytesIO(image_bytes))

        # image.show("./a.jpg")
        i+=1
        await ws.send("AAAAA")

    # await ws.close()

asyncio.run(main())
