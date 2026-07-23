"""
ws_manager.py — WebSocket Connection Manager
=============================================
Maintains a registry of connected browser clients and broadcasts
JSON events to all of them in real-time.

Events broadcasted:
    - face_event   : face recognition result (authorized / stranger / spoof)
    - vehicle_event: vehicle recognition result (known / unknown)
    - alert        : security alert (repeat visitor, forced entry, etc.)
    - system       : system health tick (CPU, memory, FPS)

Author: Hussain Raza (SP23-BAI-057)
"""

from __future__ import annotations

import asyncio
import json
from datetime import datetime
from typing import Any, Dict, List

from fastapi import WebSocket
from loguru import logger


class ConnectionManager:
    """
    Manages all active WebSocket connections from dashboard browsers.

    Usage::

        manager = ConnectionManager()

        # In WebSocket endpoint:
        await manager.connect(websocket)
        try:
            while True:
                await websocket.receive_text()  # keep-alive ping
        except WebSocketDisconnect:
            manager.disconnect(websocket)

        # From any async context:
        await manager.broadcast_event("face_event", {"person": "Hussain", ...})
    """

    def __init__(self) -> None:
        self.active: List[WebSocket] = []

    async def connect(self, websocket: WebSocket) -> None:
        await websocket.accept()
        self.active.append(websocket)
        logger.info(f"[WS] Client connected — {len(self.active)} active")

    def disconnect(self, websocket: WebSocket) -> None:
        if websocket in self.active:
            self.active.remove(websocket)
        logger.info(f"[WS] Client disconnected — {len(self.active)} active")

    async def broadcast(self, message: Dict[str, Any]) -> None:
        """Send a JSON payload to every connected client."""
        if not self.active:
            return
        data = json.dumps(message, default=str)
        dead: List[WebSocket] = []
        for ws in self.active:
            try:
                await ws.send_text(data)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.disconnect(ws)

    async def broadcast_event(
        self,
        event_type: str,
        payload: Dict[str, Any],
    ) -> None:
        """
        Broadcast a typed event envelope.

        Message shape::

            {
                "type"     : "face_event" | "vehicle_event" | "alert" | "system",
                "timestamp": "2025-01-01T12:00:00",
                "data"     : { ... payload ... }
            }
        """
        msg = {
            "type":      event_type,
            "timestamp": datetime.now().isoformat(),
            "data":      payload,
        }
        logger.info(f"[WS] Broadcasting {event_type} to {len(self.active)} clients")
        await self.broadcast(msg)

    async def send_personal(
        self,
        websocket: WebSocket,
        message: Dict[str, Any],
    ) -> None:
        """Send a message to one specific client."""
        try:
            await websocket.send_text(json.dumps(message, default=str))
        except Exception:
            self.disconnect(websocket)


# Module-level singleton — imported by main.py
ws_manager = ConnectionManager()
