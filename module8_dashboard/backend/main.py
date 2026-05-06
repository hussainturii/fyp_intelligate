"""
main.py — IntelliGate Dashboard Backend
========================================
FastAPI application that serves the REST API and WebSocket endpoint
for the IntelliGate Web Dashboard (Module 8).

Endpoints:
  GET  /api/stats                  — Dashboard overview counters
  GET  /api/events                 — Paginated event log
  POST /api/events                 — Log a new event (called by Modules 1 & 2)
  GET  /api/alerts                 — List open/resolved alerts
  POST /api/alerts                 — Create alert
  PUT  /api/alerts/{id}/resolve    — Mark alert resolved
  GET  /api/faces                  — List registered faces
  POST /api/faces/register         — Register a new face (multipart)
  DEL  /api/faces/{name}           — Delete a face profile
  GET  /api/vehicles               — List registered vehicles
  POST /api/vehicles/register      — Register a new vehicle
  DEL  /api/vehicles/{plate}       — Delete a vehicle record
  GET  /api/cameras                — List camera status
  POST /api/cameras/{id}/start     — Start a camera stream
  POST /api/cameras/{id}/stop      — Stop a camera stream
  GET  /api/stream/{id}            — MJPEG live stream
  GET  /api/health                 — System CPU/RAM/GPU health
  WS   /ws                         — WebSocket for live push events

Run with::

    cd intelligate
    uvicorn module8_dashboard.backend.main:app --reload --port 8000

Author: Hussain Raza (SP23-BAI-057)
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional

import psutil
import yaml
from fastapi import (
    Body,
    FastAPI,
    HTTPException,
    Query,
    UploadFile,
    WebSocket,
    WebSocketDisconnect,
    File,
    Form,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from loguru import logger
from pydantic import BaseModel

# ── Project root on sys.path ──────────────────────────────────────────────────
_ROOT = Path(__file__).resolve().parents[2]  # intelligate/
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# ── Internal modules ──────────────────────────────────────────────────────────
from module8_dashboard.backend.event_store  import EventStore
from module8_dashboard.backend.ws_manager   import ws_manager, ConnectionManager
from module8_dashboard.backend.camera_manager import camera_manager

# ── Config ────────────────────────────────────────────────────────────────────
_CONFIG_PATH = _ROOT / "config" / "config.yaml"
with open(_CONFIG_PATH, encoding="utf-8") as f:
    _CFG = yaml.safe_load(f)

_DASHBOARD_CFG = _CFG.get("dashboard", {})
_ALLOWED_ORIGINS = _DASHBOARD_CFG.get("cors_origins", ["http://localhost:5173", "http://localhost:3000"])
_DB_PATH = _ROOT / _DASHBOARD_CFG.get("event_db", "logs/events.db")
_CAM_IDX = _CFG.get("hardware", {}).get("camera_index", 0)
_CAM_W   = _CFG.get("hardware", {}).get("camera_width",  1280)
_CAM_H   = _CFG.get("hardware", {}).get("camera_height", 720)

# ── Lazy imports for Module 1 & 2 databases ───────────────────────────────────
def _get_face_db():
    """Load face database lazily (Module 1)."""
    try:
        from module1_face_recognition.database.face_db import FaceDatabase
        db_path = _ROOT / _CFG["face_recognition"]["database"]["path"]
        return FaceDatabase(str(db_path))
    except Exception as e:
        logger.warning(f"[Dashboard] Cannot load FaceDatabase: {e}")
        return None

def _get_vehicle_db():
    """Load vehicle database lazily (Module 2)."""
    try:
        from module2_vehicle.database.vehicle_db import VehicleDatabase
        return VehicleDatabase()
    except Exception as e:
        logger.warning(f"[Dashboard] Cannot load VehicleDatabase: {e}")
        return None


# ─────────────────────────────────────────────────────────────────────────────
# App lifecycle
# ─────────────────────────────────────────────────────────────────────────────

event_store: EventStore

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Start camera and health-tick background task on startup."""
    global event_store
    event_store = EventStore(str(_DB_PATH))

    # Start default camera
    ok = camera_manager.add_camera(_CAM_IDX, _CAM_W, _CAM_H)
    if not ok:
        logger.warning("[Dashboard] Default camera not available — stream will be offline")

    # Background system health broadcaster
    health_task = asyncio.create_task(_health_broadcast_loop())

    logger.success("[Dashboard] IntelliGate Dashboard backend running")
    yield

    # Cleanup
    health_task.cancel()
    camera_manager.stop_all()
    logger.info("[Dashboard] Backend shutdown")


async def _health_broadcast_loop() -> None:
    """Broadcast CPU/RAM stats to WebSocket clients every 5 seconds."""
    while True:
        try:
            payload = _get_health_payload()
            event_store.log_health(
                cpu_pct=payload["cpu"],
                mem_pct=payload["memory"],
                gpu_pct=payload.get("gpu", 0),
                fps=payload.get("fps", 0),
                cam_online=payload.get("camera_online", False),
            )
            await ws_manager.broadcast_event("system", payload)
        except Exception as e:
            logger.debug(f"[Health] broadcast error: {e}")
        await asyncio.sleep(5)


def _get_health_payload() -> Dict[str, Any]:
    cpu = psutil.cpu_percent(interval=None)
    mem = psutil.virtual_memory().percent
    gpu = 0.0
    try:
        import pynvml
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        util   = pynvml.nvmlDeviceGetUtilizationRates(handle)
        gpu    = float(util.gpu)
    except Exception:
        pass

    cam_list = camera_manager.list_cameras()
    cam_online = bool(cam_list)
    fps = cam_list.get(_CAM_IDX, {}).get("fps", 0.0) if cam_list else 0.0

    return {
        "cpu":           round(cpu, 1),
        "memory":        round(mem, 1),
        "gpu":           round(gpu, 1),
        "fps":           round(fps, 1),
        "camera_online": cam_online,
    }


# ─────────────────────────────────────────────────────────────────────────────
# FastAPI App
# ─────────────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="IntelliGate Dashboard API",
    version="1.0",
    description="REST + WebSocket backend for the IntelliGate Security Dashboard",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],           # Allow all origins so the MJPEG <img> tag works
    allow_credentials=False,       # Must be False when allow_origins=["*"]
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─────────────────────────────────────────────────────────────────────────────
# Pydantic Models
# ─────────────────────────────────────────────────────────────────────────────

class EventIn(BaseModel):
    module:     str
    event_type: str
    identity:   Optional[str]  = None
    confidence: Optional[float] = None
    method:     Optional[str]  = None
    camera_id:  int            = 0
    meta:       Optional[Dict] = None


class AlertIn(BaseModel):
    severity:    str
    category:    str
    title:       str
    description: Optional[str] = None
    event_id:    Optional[int] = None


# ─────────────────────────────────────────────────────────────────────────────
# ── Stats ────────────────────────────────────────────────────────────────────

@app.get("/api/stats")
async def get_stats() -> Dict[str, Any]:
    """Overview counters for the dashboard home page."""
    return event_store.get_stats()


# ─────────────────────────────────────────────────────────────────────────────
# ── Events ───────────────────────────────────────────────────────────────────

@app.get("/api/events")
async def get_events(
    module:      Optional[str] = Query(None),
    event_type:  Optional[str] = Query(None),
    since_hours: Optional[int] = Query(None),
    limit:       int           = Query(100, ge=1, le=500),
    offset:      int           = Query(0, ge=0),
) -> List[Dict]:
    return event_store.get_events(
        module=module,
        event_type=event_type,
        since_hours=since_hours,
        limit=limit,
        offset=offset,
    )


@app.post("/api/events")
async def log_event(event: EventIn) -> Dict[str, Any]:
    """
    Called by Module 1 / Module 2 pipelines to log a recognition event
    and push it to connected dashboard clients via WebSocket.
    """
    meta_str = json.dumps(event.meta) if event.meta else None
    eid = event_store.log_event(
        module=event.module,
        event_type=event.event_type,
        identity=event.identity,
        confidence=event.confidence,
        method=event.method,
        camera_id=event.camera_id,
        meta=meta_str,
    )

    # Push to WebSocket clients
    ws_type = "face_event" if event.module == "face" else "vehicle_event"
    await ws_manager.broadcast_event(ws_type, {
        "event_id":   eid,
        "event_type": event.event_type,
        "identity":   event.identity,
        "confidence": event.confidence,
        "method":     event.method,
        "camera_id":  event.camera_id,
    })

    return {"event_id": eid, "status": "logged"}


# ─────────────────────────────────────────────────────────────────────────────
# ── Alerts ───────────────────────────────────────────────────────────────────

@app.get("/api/alerts")
async def get_alerts(
    resolved: Optional[bool] = Query(False),
    limit:    int             = Query(50, ge=1, le=200),
) -> List[Dict]:
    return event_store.get_alerts(resolved=resolved, limit=limit)


@app.post("/api/alerts")
async def create_alert(alert: AlertIn) -> Dict[str, Any]:
    aid = event_store.log_alert(
        severity=alert.severity,
        category=alert.category,
        title=alert.title,
        description=alert.description,
        event_id=alert.event_id,
    )
    await ws_manager.broadcast_event("alert", {
        "alert_id":    aid,
        "severity":    alert.severity,
        "category":    alert.category,
        "title":       alert.title,
        "description": alert.description,
    })
    return {"alert_id": aid, "status": "created"}


@app.put("/api/alerts/{alert_id}/resolve")
async def resolve_alert(alert_id: int) -> Dict[str, Any]:
    event_store.resolve_alert(alert_id)
    return {"alert_id": alert_id, "status": "resolved"}


# ─────────────────────────────────────────────────────────────────────────────
# ── Face Database ─────────────────────────────────────────────────────────────

@app.get("/api/faces")
async def list_faces() -> Dict[str, Any]:
    """List all registered persons in the face database."""
    db = _get_face_db()
    if db is None:
        raise HTTPException(503, "Face database unavailable")
    try:
        # FaceDatabase.list_persons() returns dict of name → metadata
        persons = db.list_persons() if hasattr(db, "list_persons") else {}
        return {"count": len(persons), "persons": persons}
    except Exception as e:
        raise HTTPException(500, str(e))


@app.delete("/api/faces/{name}")
async def delete_face(name: str) -> Dict[str, Any]:
    db = _get_face_db()
    if db is None:
        raise HTTPException(503, "Face database unavailable")
    try:
        if hasattr(db, "delete_person"):
            db.delete_person(name)
        return {"name": name, "status": "deleted"}
    except Exception as e:
        raise HTTPException(500, str(e))


# ─────────────────────────────────────────────────────────────────────────────
# ── Vehicle Database ──────────────────────────────────────────────────────────

@app.get("/api/vehicles")
async def list_vehicles() -> Dict[str, Any]:
    """List all registered vehicles."""
    db = _get_vehicle_db()
    if db is None:
        raise HTTPException(503, "Vehicle database unavailable")
    try:
        vehicles = (db.list_vehicles_dict() if hasattr(db, "list_vehicles_dict")
                    else {})
        return {"count": len(vehicles), "vehicles": vehicles}
    except Exception as e:
        raise HTTPException(500, str(e))


@app.delete("/api/vehicles/{plate}")
async def delete_vehicle(plate: str) -> Dict[str, Any]:
    db = _get_vehicle_db()
    if db is None:
        raise HTTPException(503, "Vehicle database unavailable")
    try:
        if hasattr(db, "delete_vehicle"):
            db.delete_vehicle(plate)
        return {"plate": plate, "status": "deleted"}
    except Exception as e:
        raise HTTPException(500, str(e))


# ─────────────────────────────────────────────────────────────────────────────
# ── Cameras ───────────────────────────────────────────────────────────────────

@app.get("/api/cameras")
async def list_cameras() -> Dict[str, Any]:
    return camera_manager.list_cameras()


@app.post("/api/cameras/{cam_id}/start")
async def start_camera(cam_id: int) -> Dict[str, Any]:
    ok = camera_manager.add_camera(cam_id, _CAM_W, _CAM_H)
    if not ok:
        raise HTTPException(503, f"Cannot open camera {cam_id}")
    return {"cam_id": cam_id, "status": "started"}


@app.post("/api/cameras/{cam_id}/stop")
async def stop_camera(cam_id: int) -> Dict[str, Any]:
    camera_manager.remove_camera(cam_id)
    return {"cam_id": cam_id, "status": "stopped"}


@app.get("/api/stream/{cam_id}")
async def video_stream(cam_id: int):
    """
    MJPEG live video stream.
    Open in an <img> tag: <img src="http://localhost:8000/api/stream/0" />
    """
    if cam_id not in camera_manager._cameras:
        ok = camera_manager.add_camera(cam_id, _CAM_W, _CAM_H)
        if not ok:
            raise HTTPException(503, f"Camera {cam_id} unavailable")

    return StreamingResponse(
        camera_manager.mjpeg_generator(cam_id),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


# ─────────────────────────────────────────────────────────────────────────────
# ── Health ────────────────────────────────────────────────────────────────────

@app.get("/api/health")
async def get_health() -> Dict[str, Any]:
    return {
        **_get_health_payload(),
        "history": event_store.get_recent_health(60),
    }


# ─────────────────────────────────────────────────────────────────────────────
# ── WebSocket ─────────────────────────────────────────────────────────────────

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """
    WebSocket endpoint for real-time event push.

    On connect, sends a snapshot of current stats.
    Stays open indefinitely — clients receive events as they happen.
    """
    await ws_manager.connect(websocket)
    try:
        # Send initial state on connect
        await ws_manager.send_personal(websocket, {
            "type":  "init",
            "data":  event_store.get_stats(),
        })
        # Keep-alive: read pings from client (any text → pong)
        while True:
            msg = await websocket.receive_text()
            if msg == "ping":
                await ws_manager.send_personal(websocket, {"type": "pong"})
    except WebSocketDisconnect:
        ws_manager.disconnect(websocket)
    except Exception as e:
        logger.debug(f"[WS] Connection error: {e}")
        ws_manager.disconnect(websocket)


# ─────────────────────────────────────────────────────────────────────────────
# ── Serve React frontend (production) ─────────────────────────────────────────

_FRONTEND_DIST = Path(__file__).parent.parent / "frontend" / "dist"
if _FRONTEND_DIST.exists():
    from fastapi.responses import FileResponse

    @app.get("/{full_path:path}", include_in_schema=False)
    async def serve_spa(full_path: str):
        """Serve React SPA for any unmatched path (client-side routing)."""
        file = _FRONTEND_DIST / full_path
        if file.exists() and file.is_file():
            return FileResponse(str(file))
        return FileResponse(str(_FRONTEND_DIST / "index.html"))

    app.mount("/assets", StaticFiles(directory=str(_FRONTEND_DIST / "assets")), name="assets")