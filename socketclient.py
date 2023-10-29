import asyncio
from websockets.sync.client import connect

def hello():
    with connect("ws://localhost:8765") as websocket:
        for i in range(1, 10001):  # Change the range to run 10,000 times
            websocket.send("Request [" + str(i) + "] Hello world!")
            message = websocket.recv()
            print(f"Received: {message}")

hello()