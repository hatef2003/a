# websocket_wrapper.py

import asyncio
import websockets
import json

class WebSocketWrapper:
    def __init__(self, host: str, port: int):
        self.uri = f"ws://{host}:{port}"
        self.connection = None

    async def connect(self):
        self.connection = await websockets.connect(self.uri)
        print(f"Connected to {self.uri}")

    async def send(self, message: str):
        if self.connection is None:
            raise RuntimeError("WebSocket is not connected.")
        await self.connection.send(message)
        print(f"Sent: {message}")

    async def receive(self) -> str:
        if self.connection is None:
            raise RuntimeError("WebSocket is not connected.")
        message = await self.connection.recv()
        print(f"Received: {message}")
        return message

    async def receive_json(self) -> dict:
        if self.connection is None:
            raise RuntimeError("WebSocket is not connected.")
        message = await self.receive()  
        try:
            json_message = json.loads(message)
            print(f"Parsed JSON: {json_message}")
            return json_message
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse message as JSON: {message}") from e

    async def close(self):
        if self.connection:
            await self.connection.close()
            print("Connection closed.")
