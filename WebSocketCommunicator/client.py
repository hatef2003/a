import asyncio
import json
from WebSocketCommunicator import WebSocketWrapper

async def main():
    ws = WebSocketWrapper("localhost", 8125)
    await ws.connect()

    # Send JSON
    await ws.send(json.dumps({"type": "greeting", "content": "Hello Server!"}))
    
    response = await ws.receive_json()

    print("Received JSON response:", response['content'])

    await ws.close()

asyncio.run(main())
