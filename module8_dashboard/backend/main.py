"""CPU-only software dashboard: replay, policy, virtual devices and evidence."""
from __future__ import annotations

import asyncio
import json
import os
import threading
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

import cv2
import numpy as np
import psutil
import yaml
from fastapi import FastAPI, File, Form, HTTPException, Query, Request, Response, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .auth import Auth
from .camera_manager import camera_manager
from .event_store import EventStore
from .software_runtime import SoftwareRuntime
from .vision import FaceEngine, ReplayPipeline, INFERENCE_LOCK, appearance
from .ws_manager import ws_manager
from .test_workbench import TestWorkbench

ROOT = Path(__file__).resolve().parents[2]
CFG = yaml.safe_load((ROOT / 'config/config.yaml').read_text(encoding='utf-8'))
DB_PATH = Path(os.environ.get('IG_EVENT_DB', str(ROOT / 'logs/events.db')))
UPLOAD_DIR = DB_PATH.parent / 'video_uploads'
ORIGINS = CFG['dashboard']['cors_origins'] + ['http://127.0.0.1:5173', 'http://127.0.0.1:8000']
LAN_ORIGIN_REGEX = r"^https?://(localhost|127\.0\.0\.1|192\.168\.\d+\.\d+|10\.\d+\.\d+\.\d+|172\.(1[6-9]|2\d|3[01])\.\d+\.\d+)(:\d+)?$"


def is_allowed_origin(origin: str | None) -> bool:
    if not origin:
        return True
    if origin in ORIGINS:
        return True
    try:
        from urllib.parse import urlparse
        import ipaddress
        parsed = urlparse(origin)
        host = parsed.hostname
        if host in {'localhost', '127.0.0.1', '::1'}:
            return True
        ip = ipaddress.ip_address(host)
        return ip.is_private
    except Exception:
        return False

event_store = runtime = auth = None
face_engine = vehicle_db = None
main_loop = None
pipelines = {}
workbench = None
pipeline_lock = threading.RLock()
SOURCE_MODES = Literal['raw', 'face', 'vehicle', 'scene', 'low_light', 'thermal']


def get_faces():
    global face_engine
    with INFERENCE_LOCK:
        if face_engine is None:
            face_engine = FaceEngine(event_store)
    return face_engine


def get_vehicles():
    global vehicle_db
    with INFERENCE_LOCK:
        if vehicle_db is None:
            from module2_vehicle.database.vehicle_db import VehicleDatabase
            vehicle_db = VehicleDatabase(str(DB_PATH.parent / 'cpu_vehicles.h5'), str(DB_PATH.parent / 'cpu_unknown_vehicles.h5'))
    return vehicle_db


def on_pipeline_event(event):
    event_meta = event.setdefault('meta', {})
    event_meta['provenance'] = 'perception_internal'
    event_meta['trusted_perception'] = True
    result = runtime.observe(event, trusted=True) if event['module'] in {'face', 'vehicle'} else {
        'event_id': runtime.emit(event['module'], event['event_type'], event.get('identity'), **(event.get('meta') or {}))}
    if event['module'] in {'queue', 'visitors'} and event['event_type'] in {'congestion', 'loitering'}:
        event_store.log_alert('MEDIUM', event['event_type'], event['event_type'].replace('_', ' ').title(), event_id=result['event_id'])
    payload = {**event, **result}
    if main_loop and main_loop.is_running():
        asyncio.run_coroutine_threadsafe(ws_manager.broadcast_event('face_event' if event['module'] == 'face' else 'vehicle_event' if event['module'] == 'vehicle' else 'event', payload), main_loop)


@asynccontextmanager
async def lifespan(app):
    global event_store, runtime, auth, main_loop, face_engine, vehicle_db, workbench
    event_store = EventStore(str(DB_PATH))
    runtime, auth = SoftwareRuntime(event_store), Auth(event_store)
    workbench = TestWorkbench(ROOT,DB_PATH)
    main_loop = asyncio.get_running_loop()
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    task = asyncio.create_task(background())
    yield
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass
    await asyncio.to_thread(camera_manager.stop_all)
    pipelines.clear()
    face_engine = vehicle_db = None


async def background():
    last_broadcast = 0
    last_report = None
    with event_store._conn() as db:
        last_alert_id = db.execute('SELECT COALESCE(MAX(id),0) FROM alerts').fetchone()[0]
    while True:
        await asyncio.to_thread(runtime.tick)
        with event_store._conn() as db:
            new_alerts = db.execute('SELECT * FROM alerts WHERE id>? ORDER BY id LIMIT 200', (last_alert_id,)).fetchall()
        for row in new_alerts:
            await ws_manager.broadcast_event('alert', {**dict(row), 'alert_id': row['id']})
            last_alert_id = row['id']
        now = time.time()
        day = time.strftime('%Y-%m-%d', time.gmtime(now))
        if last_report != day:
            runtime.save('latest_report', runtime.report())
            last_report = day
        if now - last_broadcast >= 5:
            payload = health()
            event_store.log_health(payload['cpu'], payload['memory'], fps=payload['fps'], cam_online=payload['camera_online'])
            await ws_manager.broadcast_event('system', payload)
            await ws_manager.broadcast_event('gate', runtime.gate())
            last_broadcast = now
        await asyncio.sleep(0.1)


app = FastAPI(title='IntelliGate software prototype', version='2.0', lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=ORIGINS, allow_origin_regex=LAN_ORIGIN_REGEX, allow_credentials=True, allow_methods=['*'], allow_headers=['*'])


@app.middleware('http')
async def authorization(request: Request, call_next):
    path = request.url.path
    if path.startswith('/api/'):
        if request.method not in {'GET', 'HEAD', 'OPTIONS'}:
            origin = request.headers.get('origin')
            if origin and not is_allowed_origin(origin):
                return JSONResponse({'detail': 'Origin is not allowed'}, status_code=403)
        if path not in {'/api/auth/status', '/api/auth/setup', '/api/auth/login'} and request.method != 'OPTIONS':
            user = auth.user(request.cookies.get('ig_session')) if auth else None
            if not user:
                return JSONResponse({'detail': 'Sign in required'}, status_code=401)
            request.state.user = user
            if request.method not in {'GET', 'HEAD'}:
                admin_only = '/register' in path or path == '/api/auth/users' or request.method == 'DELETE'
                if (admin_only and user['role'] != 'admin') or (user['role'] == 'viewer' and path != '/api/auth/logout'):
                    return JSONResponse({'detail': 'Your role cannot perform this action'}, status_code=403)
    return await call_next(request)


class Credentials(BaseModel):
    username: str = Field(min_length=1, max_length=64)
    password: str = Field(min_length=1, max_length=128)


class UserIn(Credentials):
    role: Literal['admin', 'security_staff', 'viewer']


@app.get('/api/auth/status')
def auth_status(request: Request):
    return {'setup_required': not auth.ready(), 'user': auth.user(request.cookies.get('ig_session'))}


@app.post('/api/auth/setup')
def setup(body: Credentials, request: Request):
    if request.client.host not in {'127.0.0.1', '::1', 'localhost', 'testclient'}:
        raise HTTPException(403, 'Initial setup is local-only')
    return auth.create(body.username, body.password, 'admin', bootstrap=True)


@app.post('/api/auth/login')
def login(body: Credentials, request: Request, response: Response):
    token = auth.login(body.username, body.password, request.client.host)
    response.set_cookie('ig_session', token, httponly=True, samesite='strict', secure=request.url.scheme == 'https', max_age=8*3600)
    return {'user': auth.user(token)}


@app.post('/api/auth/logout')
def logout(request: Request, response: Response):
    auth.logout(request.cookies.get('ig_session'))
    response.delete_cookie('ig_session')
    return {'status': 'signed_out'}


@app.post('/api/auth/users')
def add_user(body: UserIn):
    return auth.create(body.username, body.password, body.role)


@app.get('/api/stats')
def stats():
    result = event_store.get_stats()
    with event_store._conn() as db:
        table = db.execute("SELECT 1 FROM sqlite_master WHERE name='cpu_faces'").fetchone()
        result['registered_faces'] = db.execute('SELECT COUNT(*) FROM cpu_faces').fetchone()[0] if table else 0
    result['registered_vehicles'] = len(get_vehicles()._vehicles)
    return result


class EventIn(BaseModel):
    module: Literal['face', 'vehicle', 'queue', 'safety', 'visitors']
    event_type: str = Field(min_length=1, max_length=64)
    identity: str | None = Field(None, max_length=128)
    confidence: float | None = Field(None, ge=0, le=1)
    method: str | None = None
    camera_id: int = Field(0, ge=0)
    meta: dict = Field(default_factory=dict)


@app.get('/api/events')
def events(module: str | None = None, event_type: str | None = None, since_hours: int | None = Query(None, ge=1, le=8760), limit: int = Query(100, ge=1, le=500), offset: int = Query(0, ge=0)):
    return event_store.get_events(module, event_type, since_hours, limit, offset)


@app.post('/api/events')
async def add_event(body: EventIn, request: Request):
    user = getattr(request.state, 'user', None)
    if not user or user.get('role') not in {'admin', 'security_staff'}:
        raise HTTPException(403, 'Operator role required to submit events')

    # Explicit Server-Side Provenance Boundary:
    # All external events over HTTP are stamped with operator provenance metadata.
    # They can never impersonate internal perception pipelines or assert validated access.
    event = body.model_dump()
    event['meta'] = dict(event.get('meta') or {})
    event['meta']['provenance'] = 'operator_annotation'
    event['meta']['submitted_by'] = user['username']
    event['meta']['client_ip'] = request.client.host if request.client else 'unknown'
    event['meta']['trusted_perception'] = False
    event['meta']['validated_liveness'] = False
    event['meta']['ocr_confidence'] = 0
    if event.get('method') in {'exact_plate', 'face_recognition', 'reid'}:
        event['method'] = f"operator_reported_{event['method']}"

    result = runtime.observe(event, trusted=False)
    await ws_manager.broadcast_event('event', {**event, **result})
    return result


@app.get('/api/system/models')
def system_models(request: Request):
    """Supply-chain inventory and verification status of active model weights."""
    user = getattr(request.state, 'user', None)
    if not user or user.get('role') not in {'admin', 'security_staff'}:
        raise HTTPException(403, 'Operator role required to view model supply-chain details')
    from .model_integrity import get_model_inventory
    return {'models': get_model_inventory()}


@app.get('/api/alerts')
def alerts(resolved: bool | None = False, limit: int = Query(50, ge=1, le=200)):
    return event_store.get_alerts(resolved, limit)


class AlertIn(BaseModel):
    severity: Literal['LOW', 'MEDIUM', 'HIGH', 'CRITICAL']
    category: str = Field(min_length=1, max_length=64)
    title: str = Field(min_length=1, max_length=200)
    description: str | None = Field(None, max_length=2000)
    event_id: int | None = None


@app.post('/api/alerts')
async def add_alert(body: AlertIn):
    if body.event_id is not None:
        with event_store._conn() as db:
            if not db.execute('SELECT 1 FROM events WHERE id=?', (body.event_id,)).fetchone():
                raise HTTPException(422, 'Evidence event not found')
    identity = event_store.log_alert(**body.model_dump())
    await ws_manager.broadcast_event('alert', {'alert_id': identity, **body.model_dump()})
    return {'status': 'created', 'alert_id': identity}


@app.put('/api/alerts/{alert_id}/resolve')
def resolve_alert(alert_id: int):
    if not event_store.resolve_alert(alert_id):
        raise HTTPException(404, 'Alert not found')
    return {'status': 'resolved', 'alert_id': alert_id}


@app.get('/api/faces')
def list_faces():
    try:
        persons = get_faces().list()
        return {'count': len(persons), 'persons': persons}
    except RuntimeError as exc:
        raise HTTPException(503, str(exc))


async def image_upload(file):
    content = await file.read(10*1024*1024 + 1)
    if len(content) > 10*1024*1024:
        raise HTTPException(413, 'Image limit is 10 MB')
    frame = cv2.imdecode(np.frombuffer(content, dtype=np.uint8), cv2.IMREAD_COLOR)
    if frame is None:
        raise HTTPException(422, 'File is not a readable image')
    if max(frame.shape[:2]) > 1600:
        scale = 1600 / max(frame.shape[:2])
        frame = cv2.resize(frame, None, fx=scale, fy=scale)
    return frame


@app.post('/api/faces/register')
async def register_face(request: Request, name: str = Form(..., min_length=1, max_length=100), role: str = Form('resident'), image: UploadFile = File(...)):
    frame = await image_upload(image)
    try:
        identity = await asyncio.to_thread(get_faces().register, name.strip(), role, frame)
    except (ValueError, RuntimeError) as exc:
        raise HTTPException(422, str(exc))
    runtime.emit('identity', 'face_registered', name, actor=request.state.user['username'])
    return {'status': 'registered', 'user_id': identity, 'model': 'SFace CPU'}


@app.delete('/api/faces/{name}')
def delete_face(name: str):
    with INFERENCE_LOCK:
        if not get_faces().delete(name):
            raise HTTPException(404, 'Person not found')
    runtime.emit('identity', 'face_removed', name)
    return {'status': 'deleted'}


@app.get('/api/vehicles')
def list_vehicles():
    with INFERENCE_LOCK:
        vehicles = get_vehicles().list_vehicles_dict()
    return {'count': len(vehicles), 'vehicles': vehicles}


@app.post('/api/vehicles/register')
async def register_vehicle(request: Request, plate: str = Form(..., min_length=1, max_length=20), owner: str = Form(..., min_length=1, max_length=100), image: UploadFile | None = File(None)):
    frame = await image_upload(image) if image and image.filename else None
    plate = ''.join(c for c in plate.upper() if c.isalnum())
    if not plate:
        raise HTTPException(422, 'Plate must contain letters or digits')
    with INFERENCE_LOCK:
        db = get_vehicles()
        if plate in db._plate_map:
            raise HTTPException(409, 'Plate already registered')
        emb = None
        notes = 'Plate registration without vehicle image'
        if frame is not None:
            try:
                from module2_vehicle.vehicle_reid import VehicleReID
                emb = VehicleReID().extract_embedding(frame)
                notes = 'OSNet-AIN 512-D deep vehicle metric embedding'
            except Exception:
                emb = appearance(frame)
                notes = 'CPU colour histogram baseline fallback'
        vid = db.register(plate, owner.strip(), embedding=emb, notes=notes)
        for pipeline in pipelines.values():
            pipeline.invalidate_vehicle_gallery()
    runtime.emit('identity', 'vehicle_registered', plate, actor=request.state.user['username'])
    return {'status': 'registered', 'vehicle_id': vid}


@app.delete('/api/vehicles/{plate}')
def delete_vehicle(plate: str):
    with INFERENCE_LOCK:
        if not get_vehicles().delete_vehicle(plate):
            raise HTTPException(404, 'Vehicle not found')
        for pipeline in pipelines.values():
            pipeline.invalidate_vehicle_gallery()
    runtime.emit('identity', 'vehicle_removed', plate)
    return {'status': 'deleted'}


def configure_source(source_id, mode, path=None, network_url=None, network_token=None):
    with pipeline_lock:
        # Single active inference source is deliberate on this 8 GB CPU host.
        previous_sources = list(pipelines)
        camera_manager.stop_all()
        for previous in previous_sources:
            runtime.obstruction(False, f'replay-{previous}')
        pipelines.clear()
        pipeline = ReplayPipeline(mode, runtime, on_pipeline_event, get_faces() if mode == 'face' else None,
                                  get_vehicles() if mode == 'vehicle' else None,
                                  runtime.load('scene_config', {}))
        pipeline.source_id = source_id
        pipeline.replayed = path is not None
        if not camera_manager.add_camera(source_id, 960, 540, video_path=str(path) if path else None,
                                         pipeline_func=pipeline.process_frame,
                                         network_url=network_url, network_token=network_token):
            raise ValueError('Cannot open camera/video: check camera index, permissions, network connection and other camera apps')
        pipelines[source_id] = pipeline
        label = network_url if network_url else Path(path).name if path else f'Camera {source_id}'
        runtime.emit('sources', 'remote_camera_started' if network_url else 'replay_started' if path else 'camera_started',
                     label, mode=mode, source_id=source_id)
        return {'cam_id': source_id, 'mode': mode, 'status': 'updated', 'replayed': path is not None, 'remote': network_url is not None, 'filename': label}


@app.post('/api/cameras/{cam_id}/upload-video')
async def upload_video(cam_id: int, video: UploadFile = File(...), mode: SOURCE_MODES = Form('raw')):
    if not 0 <= cam_id <= 7:
        raise HTTPException(422, 'Source index must be 0-7')
    suffix = Path(video.filename or '').suffix.lower()
    if suffix not in {'.mp4', '.avi', '.mov', '.mkv', '.webm'}:
        raise HTTPException(422, 'Upload MP4, AVI, MOV, MKV or WebM')
    path = UPLOAD_DIR / f'{uuid.uuid4()}{suffix}'
    size = 0
    try:
        with path.open('wb') as dest:
            while chunk := await video.read(1024*1024):
                size += len(chunk)
                if size > 250*1024*1024:
                    raise HTTPException(413, 'Video limit is 250 MB')
                dest.write(chunk)
        return await asyncio.to_thread(configure_source, cam_id, mode, path)
    except HTTPException:
        path.unlink(missing_ok=True)
        raise
    except Exception as exc:
        path.unlink(missing_ok=True)
        raise HTTPException(422, f'Pipeline unavailable: {exc}')


@app.get('/api/cameras')
def list_cameras():
    return camera_manager.list_cameras()


DEMO_SOURCES = {
    'local_vehicle': (ROOT.parent / 'nre/video test.mp4', 'Local vehicle clip (unannotated)'),
    'llvip_visible': (ROOT / 'logs/demo_data/llvip_visible.mp4', 'LLVIP visible illustration (unannotated)'),
    'llvip_infrared': (ROOT / 'logs/demo_data/llvip_infrared.mp4', 'LLVIP genuine infrared illustration (unannotated)'),
    'face_smoke': (ROOT / 'logs/demo_data/face_smoke.mp4', 'Static OpenCV face smoke test (not liveness data)'),
}


@app.get('/api/demo/sources')
def demo_sources():
    return [{'id': key, 'label': label} for key, (path, label) in DEMO_SOURCES.items() if path.exists()]


class DemoIn(BaseModel):
    source: Literal['local_vehicle', 'llvip_visible', 'llvip_infrared', 'face_smoke']
    mode: SOURCE_MODES = 'raw'


@app.post('/api/demo/replay')
async def demo_replay(body: DemoIn):
    if body.source.startswith('llvip') and body.mode in {'face', 'vehicle'}:
        raise HTTPException(422, 'LLVIP inputs are for scene/low-light detection only, not identity recognition')
    path, _ = DEMO_SOURCES[body.source]
    if not path.exists():
        raise HTTPException(404, 'Run python tools/prepare_demo_data.py first')
    try:
        return await asyncio.to_thread(configure_source, 0, body.mode, path)
    except Exception as exc:
        raise HTTPException(422, str(exc))


@app.post('/api/cameras/{cam_id}/start')
async def start_source(cam_id: int, mode: SOURCE_MODES = Query('face')):
    if not 0 <= cam_id <= 7:
        raise HTTPException(422,'Camera index must be 0-7')
    if mode == 'thermal':
        raise HTTPException(422,'Use genuine infrared recordings for thermal mode')
    try:
        return await asyncio.to_thread(configure_source,cam_id,mode)
    except Exception as exc:
        raise HTTPException(422,str(exc))


class RemoteCameraIn(BaseModel):
    url: str = Field(min_length=1, max_length=500)
    token: str = Field('', max_length=200)
    mode: SOURCE_MODES = 'face'


@app.post('/api/cameras/{cam_id}/network')
async def start_remote_source(cam_id: int, body: RemoteCameraIn, request: Request):
    if not 0 <= cam_id <= 7:
        raise HTTPException(422, 'Camera index must be 0-7')
    if body.mode == 'thermal':
        raise HTTPException(422, 'Use genuine infrared recordings for thermal mode')

    user = getattr(request.state, 'user', None)
    if not user or user.get('role') not in {'admin', 'security_staff'}:
        raise HTTPException(403, 'Operator role required to configure camera sources')

    from .camera_manager import validate_camera_url
    ok, err_msg, valid_url = validate_camera_url(body.url)
    if not ok:
        raise HTTPException(422, f'Remote camera URL rejected: {err_msg}')

    try:
        return await asyncio.to_thread(configure_source, cam_id, body.mode, None, valid_url, body.token.strip())
    except Exception as exc:
        raise HTTPException(422, str(exc))


class CameraEnrolment(BaseModel):
    name: str = Field(min_length=1,max_length=100)
    role: Literal['resident','employee','security_staff'] = 'resident'
    camera_id: int = Field(0,ge=0,le=7)


@app.post('/api/faces/register-camera')
async def register_camera_face(body: CameraEnrolment, request: Request):
    snapshot = camera_manager.get_snapshot(body.camera_id)
    if snapshot is None:
        raise HTTPException(422,'Open a camera/source first and wait for a clear frame')
    try:
        identity = await asyncio.to_thread(get_faces().register,body.name.strip(),body.role,snapshot)
    except (ValueError,RuntimeError) as exc:
        raise HTTPException(422,str(exc))
    runtime.emit('identity','face_registered',body.name,actor=request.state.user['username'])
    return {'status':'registered','user_id':identity,'model':'SFace CPU'}


@app.post('/api/cameras/{cam_id}/stop')
async def stop_source(cam_id: int):
    await asyncio.to_thread(camera_manager.remove_camera, cam_id)
    runtime.obstruction(False, f'replay-{cam_id}')
    pipelines.pop(cam_id, None)
    return {'status': 'stopped'}


@app.get('/api/stream/{cam_id}')
def stream(cam_id: int):
    if cam_id not in camera_manager._cameras:
        raise HTTPException(404, 'Upload a video first')
    return StreamingResponse(camera_manager.mjpeg_generator(cam_id), media_type='multipart/x-mixed-replace; boundary=frame')


class GateCommand(BaseModel):
    command: Literal['OPEN', 'CLOSE', 'STOP', 'LOCKDOWN', 'RESET']
    command_id: str = Field(default_factory=lambda: str(uuid.uuid4()), max_length=100)
    expires: float | None = None


@app.get('/api/gate')
def gate():
    return runtime.tick()


@app.post('/api/gate/command')
async def gate_command(body: GateCommand, request: Request):
    try:
        result = runtime.command(body.command, request.state.user['username'], body.command_id, body.expires)
    except ValueError as exc:
        raise HTTPException(409, str(exc))
    await ws_manager.broadcast_event('gate', result)
    return result


class ObstructionIn(BaseModel):
    present: bool


@app.post('/api/gate/obstruction')
def obstruction(body: ObstructionIn):
    return runtime.obstruction(body.present)


@app.get('/api/reviews')
def reviews():
    return runtime.reviews()


class ReviewIn(BaseModel):
    approve: bool


@app.post('/api/reviews/{request_id}')
def review(request_id: str, body: ReviewIn, request: Request):
    try:
        return runtime.review(request_id, body.approve, request.state.user['username'])
    except ValueError as exc:
        raise HTTPException(409, str(exc))


@app.get('/api/visitors')
def visitors():
    return runtime.visitors()


@app.get('/api/reports')
def reports(hours: int = Query(24, ge=1, le=8760)):
    return runtime.report(hours)


@app.get('/api/search')
def search(q: str = Query(..., min_length=1, max_length=200)):
    try:
        return runtime.search(q)
    except ValueError as exc:
        raise HTTPException(422, str(exc))


class SceneConfig(BaseModel):
    queue_roi: list[float] = Field(default=[0,0,1,1], min_length=4, max_length=4)
    gate_roi: list[float] = Field(default=[0.4,0.65,0.6,1], min_length=4, max_length=4)
    queue_threshold: int = Field(5, ge=1, le=100)
    loiter_seconds: int = Field(30, ge=1, le=3600)


@app.post('/api/scene/config')
def scene_config(body: SceneConfig):
    for roi in [body.queue_roi, body.gate_roi]:
        if not all(0 <= v <= 1 for v in roi) or not roi[0] < roi[2] or not roi[1] < roi[3]:
            raise HTTPException(422, 'ROIs require normalized x1,y1,x2,y2 with positive area')
    runtime.save('scene_config', body.model_dump())
    for pipeline in list(pipelines.values()):
        pipeline.config = body.model_dump()
    return body.model_dump()


@app.get('/api/modules')
def modules():
    return {'simulated': True, 'device': 'cpu', 'modules': [
        {'id': 1, 'name': 'Face recognition', 'status': 'Live/replay multi-face tracking; cached SFace identities; liveness unvalidated; review required'},
        {'id': 2, 'name': 'Vehicles', 'status': 'YOLO vehicles / EasyOCR; OpenCV contour plate fallback; colour histogram appearance baseline'},
        {'id': 3, 'name': 'Night / infrared', 'status': 'CLAHE RGB enhancement; genuine infrared input required; benchmark pending'},
        {'id': 4, 'name': 'Stranger memory', 'status': 'Encrypted face candidates; distinct session visits; rule-based dwell'},
        {'id': 5, 'name': 'Queue / density', 'status': 'ROI counts and IoU tracking baseline'},
        {'id': 6, 'name': 'Safety', 'status': 'ROI obstruction, virtual STOP and operator lockdown'},
        {'id': 7, 'name': 'Reporting', 'status': 'Evidence templates and keyword search; VLM/SSM deferred'},
        {'id': 8, 'name': 'Dashboard', 'status': 'Live webcam/replay, enrolment, approvals, alerts and reports'},
        {'id': 9, 'name': 'Devices', 'status': 'Virtual gate only; physical hardware excluded'},
        {'id': 10, 'name': 'Backend', 'status': 'Cookie authentication, roles, SQLite/HDF5 persistence'},
    ], 'scene': runtime.load('scene', {}), 'scene_config': runtime.load('scene_config', SceneConfig().model_dump()),
       'pipelines': {key: value.metrics for key, value in pipelines.items()}}


@app.post('/api/testing/checks',status_code=202)
def testing_checks():
    try:
        return {'id':workbench.checks()}
    except ValueError as exc:
        raise HTTPException(409,str(exc))


class CameraTestIn(BaseModel):
    camera_id:int=Field(0,ge=0,le=7)
    expected_identity:str|None=Field(None,max_length=100)
    expected_count:int|None=Field(None,ge=0,le=100)


@app.post('/api/testing/camera',status_code=202)
def testing_camera(body:CameraTestIn):
    frame=camera_manager.get_snapshot(body.camera_id)
    if frame is None:
        raise HTTPException(422,'Open a camera/source first and wait for a frame')
    ok,encoded=cv2.imencode('.png',frame)
    if not ok:
        raise HTTPException(422,'Cannot encode the current camera frame')
    try:
        run_id=workbench.reserve('camera_frame_test')
    except ValueError as exc:
        raise HTTPException(409,str(exc))
    path=workbench.folder(run_id)/'input.png'
    try:
        path.write_bytes(encoded.tobytes())
        workbench.media(run_id,path,'face',1,body.expected_identity.strip() if body.expected_identity else None,body.expected_count)
        return {'id':run_id}
    except Exception:
        path.unlink(missing_ok=True)
        with workbench.lock:
            workbench.active_id=None
        raise


@app.post('/api/testing/runs',status_code=202)
async def testing_media(file:UploadFile=File(...),mode:Literal['face','vehicle','scene','low_light','thermal']=Form('face'),
                        max_frames:int=Form(30,ge=1,le=300),expected_identity:str|None=Form(None,max_length=100),
                        expected_count:int|None=Form(None,ge=0,le=100)):
    if expected_identity and mode!='face':
        raise HTTPException(422,'Expected identity is supported in face mode only')
    if expected_count is not None and mode=='low_light':
        raise HTTPException(422,'Low-light enhancement does not estimate object counts')
    suffix=Path(file.filename or '').suffix.lower()
    if suffix not in {'.jpg','.jpeg','.png','.webp','.bmp','.mp4','.avi','.mov','.mkv','.webm'}:
        raise HTTPException(422,'Upload a supported image or video')
    try:
        run_id=workbench.reserve('media_test')
    except ValueError as exc:
        raise HTTPException(409,str(exc))
    path=workbench.folder(run_id)/f'input{suffix}'
    try:
        size=0
        with path.open('wb') as dest:
            while chunk:=await file.read(1024*1024):
                size+=len(chunk)
                if size>250*1024*1024:
                    raise HTTPException(413,'Input limit is 250 MB')
                dest.write(chunk)
        if not size:
            raise HTTPException(422,'Uploaded file is empty')
        workbench.media(run_id,path,mode,max_frames,expected_identity.strip() if expected_identity else None,expected_count)
        return {'id':run_id}
    except Exception:
        path.unlink(missing_ok=True)
        with workbench.lock:
            workbench.active_id=None
        workbench.write(run_id,{'id':run_id,'kind':'media_test','status':'error','error':'Upload did not complete'})
        raise


@app.get('/api/testing/runs')
def testing_history():
    paths=sorted(workbench.directory.glob('*/report.json'),key=lambda path:path.stat().st_mtime,reverse=True)[:30]
    result=[]
    for path in paths:
        try:
            data=json.loads(path.read_text(encoding='utf-8'))
            result.append({key:data.get(key) for key in ['id','kind','status','mode','verdict','created_at']})
        except (OSError,ValueError):
            continue
    return result


@app.get('/api/testing/runs/{run_id}')
def testing_result(run_id:str):
    try:
        return workbench.get(run_id)
    except (ValueError,FileNotFoundError):
        raise HTTPException(404,'Test run not found')


@app.get('/api/testing/runs/{run_id}/preview')
def testing_preview(run_id:str):
    try:
        path=workbench.folder(run_id)/'preview.jpg'
    except ValueError:
        raise HTTPException(404,'Test run not found')
    if not path.exists():
        raise HTTPException(404,'Preview not available')
    return FileResponse(path,media_type='image/jpeg')


def health():
    sources = camera_manager.list_cameras()
    return {'cpu': psutil.cpu_percent(), 'memory': psutil.virtual_memory().percent, 'gpu': 0,
            'fps': sum(source['fps'] for source in sources.values()), 'camera_online': any(source['running'] for source in sources.values()),
            'device': 'cpu', 'physical_gate_excluded': True, 'live_camera_enabled': True, 'simulated': True, 'sources': sources}


@app.get('/api/health')
def get_health():
    return {**health(), 'history': event_store.get_recent_health()}


@app.websocket('/ws')
async def websocket(websocket: WebSocket):
    origin = websocket.headers.get('origin')
    if (origin and not is_allowed_origin(origin)) or not auth.user(websocket.cookies.get('ig_session')):
        await websocket.close(code=1008)
        return
    await ws_manager.connect(websocket)
    try:
        await ws_manager.send_personal(websocket, {'type': 'init', 'data': event_store.get_stats()})
        while True:
            message = await websocket.receive_text()
            if not auth.user(websocket.cookies.get('ig_session')):
                await websocket.close(code=1008)
                break
            if message == 'ping':
                await ws_manager.send_personal(websocket, {'type': 'pong'})
    except WebSocketDisconnect:
        pass
    finally:
        ws_manager.disconnect(websocket)


DIST = ROOT / 'module8_dashboard/frontend/dist'
if DIST.exists():
    app.mount('/assets', StaticFiles(directory=str(DIST / 'assets')), name='assets')

    @app.get('/{full_path:path}', include_in_schema=False)
    def spa(full_path: str):
        if full_path.startswith('api/'):
            raise HTTPException(404, 'Endpoint not found')
        target = (DIST / full_path).resolve()
        if target.is_relative_to(DIST.resolve()) and target.is_file():
            return FileResponse(target)
        return FileResponse(DIST / 'index.html')
