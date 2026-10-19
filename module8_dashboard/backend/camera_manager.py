"""Live/webcam and replay capture with a bounded latest-frame inference queue."""
from __future__ import annotations

import asyncio
import math
import threading
import time
from typing import Optional

import cv2
import numpy as np
from loguru import logger


def validate_camera_url(
    url: str,
    allowed_schemes: tuple[str, ...] = ('http', 'https'),
    allowed_ports: tuple[int, ...] | None = None,
    allow_private_only: bool = True,
) -> tuple[bool, str, str]:
    """
    Validate remote camera URL against SSRF and network isolation boundaries.
    Enforces scheme, port whitelist, domain resolution, private RFC1918 range,
    and strictly forbids loopback (127.0.0.1/localhost) and link-local (169.254.169.254).
    """
    import os
    import socket
    import ipaddress
    from urllib.parse import urlparse

    if not url or not isinstance(url, str):
        return False, 'URL must be a non-empty string', ''

    url = url.strip()
    try:
        parsed = urlparse(url)
    except Exception as exc:
        return False, f'Malformed URL: {exc}', ''

    if parsed.scheme.lower() not in allowed_schemes:
        return False, f"Scheme '{parsed.scheme}' not allowed. Must be one of: {', '.join(allowed_schemes)}", ''

    hostname = parsed.hostname
    if not hostname:
        return False, 'Missing hostname in camera URL', ''

    port = parsed.port or (80 if parsed.scheme.lower() == 'http' else 443)
    if allowed_ports is not None:
        if port not in allowed_ports:
            return False, f'Port {port} is not in the allowed camera ports list', ''
    else:
        if port not in (80, 443) and not (1024 <= port <= 65535):
            return False, f'Port {port} is restricted. Only standard web ports (80, 443) and ports 1024-65535 are permitted.', ''

    allow_loopback = os.environ.get('INTELLIGATE_ALLOW_LOOPBACK_CAM', '0') == '1'

    try:
        addr_info = socket.getaddrinfo(hostname, port, family=socket.AF_INET, type=socket.SOCK_STREAM)
        if not addr_info:
            return False, f"Could not resolve host '{hostname}'", ''
        resolved_ip_str = addr_info[0][4][0]
        ip = ipaddress.ip_address(resolved_ip_str)
    except Exception as exc:
        return False, f"DNS resolution failed for '{hostname}': {exc}", ''

    if ip.is_loopback:
        if not allow_loopback:
            return False, 'Loopback addresses (localhost/127.0.0.1) are forbidden for remote cameras', ''

    if ip.is_link_local:
        return False, 'Link-local addresses (e.g. cloud metadata 169.254.169.254) are forbidden', ''

    if ip.is_multicast or ip.is_reserved:
        return False, 'Multicast and reserved IP addresses are forbidden', ''

    if allow_private_only and not allow_loopback:
        if not ip.is_private:
            return False, f'Public IP addresses ({resolved_ip_str}) are forbidden. Remote camera must be on private LAN (192.168.x.x, 10.x.x.x, 172.16-31.x.x)', ''

    return True, '', url


class CameraStream:
    def __init__(self, cam_index, width=960, height=540, video_path=None, pipeline_func=None,
                 network_url=None, network_token=None):
        self.cam_index, self.width, self.height = cam_index, width, height
        self.video_path = video_path
        self.network_url = network_url
        self.network_token = network_token
        self._cap = None
        self._frame = self._snapshot = self._latest = None
        self._running = False
        self._thread = self._inference_thread = None
        self._lock = threading.Lock()
        self._condition = threading.Condition(self._lock)
        self._pipeline_func = pipeline_func
        self._sequence = 0
        self._fps = self._inference_fps = 0.0
        self.last_error = None
        self.source_time = 0.0
        self.actual_resolution = None
        self.processed_frames = 0
        self.last_inference_at = None
        self.last_inference_started = None

    def set_pipeline(self, pipeline_func):
        with self._condition:
            self._pipeline_func = pipeline_func
            self._condition.notify_all()

    def start(self):
        if self.network_url:
            self._running = True
            self._thread = threading.Thread(target=self._network_capture_loop, daemon=True, name=f'netcapture-{self.cam_index}')
            self._inference_thread = threading.Thread(target=self._inference_loop, daemon=True, name=f'inference-{self.cam_index}')
            self._thread.start()
            self._inference_thread.start()
            return True

        source = self.video_path if self.video_path else self.cam_index
        # DirectShow usually avoids the long MSMF webcam startup on Windows.
        import os
        if not self.video_path and os.name == 'nt':
            self._cap = cv2.VideoCapture(source, cv2.CAP_DSHOW)
            if not self._cap.isOpened():
                self._cap.release()
                self._cap = cv2.VideoCapture(source)
        else:
            self._cap = cv2.VideoCapture(source)
        if not self._cap.isOpened():
            self._cap.release()
            self.last_error = 'Cannot open video or camera; check index, permissions and other camera apps'
            return False
        if not self.video_path:
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            self._cap.set(cv2.CAP_PROP_FPS, 30)
            self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self._running = True
        self._thread = threading.Thread(target=self._capture_loop, daemon=True, name=f'capture-{self.cam_index}')
        self._inference_thread = threading.Thread(target=self._inference_loop, daemon=True, name=f'inference-{self.cam_index}')
        self._thread.start()
        self._inference_thread.start()
        return True

    def stop(self):
        with self._condition:
            self._running = False
            owner = getattr(self._pipeline_func, '__self__', None)
            if owner is not None:
                owner.active = False
            self._condition.notify_all()
        if self._thread:
            self._thread.join(timeout=3)
        if self._inference_thread:
            self._inference_thread.join(timeout=5)
        # The capture thread owns/release its capture handle; no concurrent reads.

    @property
    def is_running(self):
        return self._running

    @property
    def fps(self):
        return self._fps

    def _capture_loop(self):
        source_fps = self._cap.get(cv2.CAP_PROP_FPS)
        if not math.isfinite(source_fps) or not 1 <= source_fps <= 120:
            source_fps = 24
        started, interval_start, count, failures = time.monotonic(), time.monotonic(), 0, 0
        try:
            while self._running:
                frame_started = time.monotonic()
                ok, frame = self._cap.read()
                if not ok:
                    if self.video_path:
                        break
                    failures += 1
                    if failures >= 30:
                        self.last_error = 'Live camera stopped returning frames'
                        break
                    time.sleep(.05)
                    continue
                failures = 0
                scale = min(1, self.width / frame.shape[1])
                if scale < 1:
                    frame = cv2.resize(frame, None, fx=scale, fy=scale)
                self.actual_resolution = f'{frame.shape[1]}x{frame.shape[0]}'
                self.source_time = self._cap.get(cv2.CAP_PROP_POS_MSEC)/1000 if self.video_path else time.monotonic()-started
                with self._condition:
                    self._snapshot = frame.copy()
                    self._latest = frame.copy()  # One pending frame; older frames are replaced.
                    self._sequence += 1
                    pipeline = self._pipeline_func
                    self._condition.notify_all()
                preview = frame.copy()
                owner = getattr(pipeline, '__self__', None)
                if owner is not None and getattr(owner, 'mode', None) == 'low_light':
                    from .vision import enhance
                    preview = enhance(preview)
                elif owner is not None and hasattr(owner, 'overlay_frame'):
                    preview = owner.overlay_frame(preview)
                prefix = 'REPLAY' if self.video_path else 'LIVE'
                cv2.putText(preview, f'{prefix} {self.cam_index} | {self.source_time:.1f}s',
                            (10,24), cv2.FONT_HERSHEY_SIMPLEX,.6,(255,255,255),2)
                if self.last_error:
                    cv2.putText(preview,'PIPELINE ERROR - see source status',(10,50),0,.5,(0,0,255),1)
                encoded, jpeg = cv2.imencode('.jpg',preview,[cv2.IMWRITE_JPEG_QUALITY,80])
                if encoded:
                    with self._lock:
                        self._frame = jpeg.tobytes()
                count += 1
                elapsed = time.monotonic()-interval_start
                if elapsed >= 1:
                    self._fps = count/elapsed
                    interval_start, count = time.monotonic(), 0
                # Pace files; live capture is paced by the device itself.
                if self.video_path:
                    time.sleep(max(0, 1/source_fps-(time.monotonic()-frame_started)))
        except Exception as exc:
            self.last_error = str(exc)
            logger.exception('Capture failed')
        finally:
            if self._cap:
                self._cap.release()
            with self._condition:
                self._running = False
                self._condition.notify_all()

    def _network_capture_loop(self):
        import urllib.request
        import urllib.error
        import time

        started = interval_start = time.monotonic()
        count = 0
        headers = {'User-Agent': 'IntelliGate-Camera-Ingestion/2.0'}
        if self.network_token:
            headers['X-Camera-Token'] = self.network_token
            headers['Authorization'] = f'Bearer {self.network_token}'

        req = urllib.request.Request(self.network_url, headers=headers)
        bytes_buf = bytearray()
        MAX_FRAME_SIZE = 5 * 1024 * 1024  # 5 MB frame limit

        while self._running:
            try:
                with urllib.request.urlopen(req, timeout=5.0) as resp:
                    self.last_error = None
                    while self._running:
                        chunk = resp.read(16384)
                        if not chunk:
                            break
                        bytes_buf.extend(chunk)

                        if len(bytes_buf) > MAX_FRAME_SIZE:
                            bytes_buf = bytes_buf[-1024 * 1024:]

                        a = bytes_buf.find(b'\xff\xd8')
                        b = bytes_buf.find(b'\xff\xd9')
                        if a != -1 and b != -1 and b > a:
                            jpg_bytes = bytes(bytes_buf[a:b + 2])
                            del bytes_buf[:b + 2]

                            frame = cv2.imdecode(np.frombuffer(jpg_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
                            if frame is None:
                                continue

                            scale = min(1, self.width / frame.shape[1])
                            if scale < 1:
                                frame = cv2.resize(frame, None, fx=scale, fy=scale)
                            self.actual_resolution = f'{frame.shape[1]}x{frame.shape[0]}'
                            self.source_time = time.monotonic() - started

                            with self._condition:
                                self._snapshot = frame.copy()
                                self._latest = frame.copy()
                                self._sequence += 1
                                pipeline = self._pipeline_func
                                self._condition.notify_all()

                            preview = frame.copy()
                            owner = getattr(pipeline, '__self__', None)
                            if owner is not None and getattr(owner, 'mode', None) == 'low_light':
                                from .vision import enhance
                                preview = enhance(preview)
                            elif owner is not None and hasattr(owner, 'overlay_frame'):
                                preview = owner.overlay_frame(preview)

                            cv2.putText(preview, f'REMOTE {self.cam_index} | {self.source_time:.1f}s',
                                        (10, 24), cv2.FONT_HERSHEY_SIMPLEX, .6, (255, 255, 255), 2)
                            if self.last_error:
                                cv2.putText(preview, 'STREAM NOTICE - reconnecting...', (10, 50), 0, .5, (0, 0, 255), 1)

                            encoded, jpeg = cv2.imencode('.jpg', preview, [cv2.IMWRITE_JPEG_QUALITY, 80])
                            if encoded:
                                with self._lock:
                                    self._frame = jpeg.tobytes()

                            count += 1
                            elapsed = time.monotonic() - interval_start
                            if elapsed >= 1:
                                self._fps = count / elapsed
                                interval_start, count = time.monotonic(), 0
            except Exception as exc:
                if not self._running:
                    break
                self.last_error = f'Remote stream disconnected ({exc})'
                time.sleep(1.5)

        with self._condition:
            self._running = False
            self._condition.notify_all()

    def _inference_loop(self):
        last_sequence, next_inference = -1, 0.0
        interval_start, count = time.monotonic(), 0
        while True:
            with self._condition:
                ready=self._condition.wait_for(lambda:not self._running or (self._latest is not None and self._sequence != last_sequence), timeout=.5)
                if not self._running:
                    return
                # A cold webcam can take seconds to produce its first frame.
                # A condition timeout is not a frame-ready notification.
                if not ready or self._latest is None or self._sequence == last_sequence:
                    continue
                if time.monotonic() < next_inference:
                    self._condition.wait(timeout=next_inference-time.monotonic())
                    continue
                frame = self._latest.copy()
                last_sequence = self._sequence
                pipeline = self._pipeline_func
            if pipeline:
                inference_started = time.monotonic()
                self.last_inference_started = inference_started
                try:
                    pipeline(frame)
                    self.last_error = None
                    count += 1
                    self.processed_frames += 1
                    self.last_inference_at = time.monotonic()
                except Exception as exc:
                    self.last_error = str(exc)
                    logger.error(f'Inference failed: {exc}')
                next_inference = inference_started+.2
            else:
                next_inference = time.monotonic()+.2
            elapsed = time.monotonic()-interval_start
            if elapsed >= 1:
                self._inference_fps = count/elapsed
                interval_start, count = time.monotonic(), 0

    def get_frame(self):
        with self._lock:
            return self._frame

    def get_snapshot(self):
        with self._lock:
            return self._snapshot.copy() if self._snapshot is not None else None


class CameraManager:
    def __init__(self):
        self._cameras = {}

    def add_camera(self, cam_index, width=960, height=540, video_path=None, pipeline_func=None,
                   network_url=None, network_token=None):
        if cam_index in self._cameras:
            return self._cameras[cam_index].is_running
        camera = CameraStream(cam_index, width, height, video_path, pipeline_func,
                              network_url=network_url, network_token=network_token)
        if camera.start():
            self._cameras[cam_index] = camera
            return True
        return False

    def set_pipeline(self, cam_index, pipeline_func):
        if cam_index in self._cameras:
            self._cameras[cam_index].set_pipeline(pipeline_func)

    def remove_camera(self, cam_index):
        camera = self._cameras.pop(cam_index,None)
        if camera:
            camera.stop()

    def get_frame(self, cam_index):
        camera = self._cameras.get(cam_index)
        return camera.get_frame() if camera else None

    def get_snapshot(self, cam_index):
        camera = self._cameras.get(cam_index)
        return camera.get_snapshot() if camera else None

    def list_cameras(self):
        return {index:{'index':index,'running':camera.is_running,'fps':round(camera.fps,1),
                       'resolution':camera.actual_resolution,'replayed':bool(camera.video_path),
                       'remote':bool(camera.network_url),
                       'source_time_seconds':round(camera.source_time,2),'inference_sampling_fps':5,
                       'inference_fps':round(camera._inference_fps,1),'last_error':camera.last_error,
                       'processing_mode':getattr(getattr(camera._pipeline_func,'__self__',None),'mode','custom'),
                       'worker_alive':bool(camera._inference_thread and camera._inference_thread.is_alive()),
                       'processed_frames':camera.processed_frames,
                       'last_inference_age_seconds':round(time.monotonic()-camera.last_inference_at,1) if camera.last_inference_at else None,
                       'processing_status':('stopped' if not camera.is_running else 'error' if camera.last_error or not camera._inference_thread.is_alive() else 'waiting_for_first_frame' if camera._latest is None else 'starting_inference' if camera.last_inference_at is None else 'stalled' if time.monotonic()-camera.last_inference_at>10 else 'running')}
                for index,camera in list(self._cameras.items())}

    async def mjpeg_generator(self, cam_index):
        while cam_index in self._cameras:
            frame = self.get_frame(cam_index)
            if frame:
                yield b'--frame\r\nContent-Type: image/jpeg\r\n\r\n'+frame+b'\r\n'
            await asyncio.sleep(1/30)

    def stop_all(self):
        for index in list(self._cameras):
            self.remove_camera(index)


camera_manager = CameraManager()
