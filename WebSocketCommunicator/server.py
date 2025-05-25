# server.py
import asyncio
import websockets
import json

async def handle_connection(websocket):
    async for message in websocket:
        print(f"Received: {message}")
        data = json.loads(message)
        
        # Respond with a simple message
        response = {"type": "response", "content": "Hello, Client!"}
        await websocket.send(json.dumps(response))

async def main():
    # Start the server on localhost:8125
    async with websockets.serve(handle_connection, "localhost", 8125):
        print("WebSocket server started on ws://localhost:8125")
        await asyncio.Future()  # run forever

asyncio.run(main())
