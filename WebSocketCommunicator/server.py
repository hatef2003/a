import asyncio
import websockets
import json
from PIL import Image
import io
import base64
import time  # Added for timing

async def handle_connection(websocket):
    # Load and encode image once
    with open("C:\\Users\\sarag\\Desktop\\Hatef\\a\\pets.jpg", "rb") as f:
        image_data = f.read()
    
    image_base64 = base64.b64encode(image_data).decode('utf-8')
    response = {"type": "response", "payload": image_base64}

    try:
        for i in range(1000):
            # Record time before sending
            start_time = time.time()

            # Send image
            await websocket.send("json.dumps(response)")
            end_time = time.time()

            try:
                # Wait for response from client
                message = await websocket.recv()

                # Record time after receiving

                elapsed_time = end_time - start_time
                print(f"Received from client: {message}")
                print(f"Round-trip time: {elapsed_time:.6f} seconds")

            except websockets.exceptions.ConnectionClosed:
                print("Client disconnected during message exchange.")
                break

        # Keep connection alive afterwards
        while True:
            await asyncio.sleep(1)
    except websockets.exceptions.ConnectionClosed:
        print("Client disconnected.")

async def main():
    async with websockets.serve(handle_connection, "0.0.0.0", 8125):
        print("WebSocket server started on ws://0.0.0.0:8125")
        await asyncio.Future()  # run forever

asyncio.run(main())
