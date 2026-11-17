from typing import List, Dict
from fastapi import WebSocket
import asyncio
import json

class ConnectionManager:
    def __init__(self):
        # List to store all active websocket connections
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        print(f"Client connected. Total clients: {len(self.active_connections)}")

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
            print(f"Client disconnected. Total clients: {len(self.active_connections)}")

    async def send_personal_message(self, message: str, websocket: WebSocket):
        await websocket.send_text(message)

    async def broadcast(self, data: dict):
        """Broadcasts a JSON message to all connected clients"""
        if not self.active_connections:
            return
            
        message = json.dumps(data)
        disconnected = []
        
        for connection in self.active_connections:
            try:
                await connection.send_text(message)
            except Exception as e:
                # If connection is broken, mark for removal
                disconnected.append(connection)
                
        # Cleanup broken connections
        for conn in disconnected:
            self.disconnect(conn)

manager = ConnectionManager()
