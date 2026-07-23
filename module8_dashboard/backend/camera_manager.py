"""
camera_manager.py — MJPEG Camera Stream Manager
================================================
Manages one or more OpenCV capture sources and exposes them
as async MJPEG streams that FastAPI can serve.

Each camera runs in a background thread that continuously reads frames.
The latest frame is stored in memory and served to all stream clients
without them needing to synchronize with each other.

Usage (in FastAPI route)::

    @app.get("/api/stream/{cam_id}")
    async def stream(cam_id: int):
        return StreamingResponse(
            camera_manager.mjpeg_generator(cam_id),
            media_type="multipart/x-mixed-replace; boundary=frame",
        )

Author: Hussain Raza (SP23-BAI-057)
"""

from __future__ import annotations

import asyncio
import threading
import time
from typing import AsyncGenerator, Dict, Optional

import cv2
import numpy as np
from loguru import logger


class CameraStream:
    """
    One camera: runs a background thread that reads frames continuously.
    Latest JPEG-encoded frame is always available via `self.frame`.
    """

    def __init__(self, cam_index: int, width: int = 1280, height: int = 720, video_path: Optional[str] = None) -> None:
        self.cam_index = cam_index
        self.width     = width
        self.height    = height
        self.video_path = video_path

        self._cap:     Optional[cv2.VideoCapture] = None
        self._frame:   Optional[bytes] = None       # latest JPEG bytes
        self._running: bool = False
        self._thread:  Optional[threading.Thread] = None
        self._lock     = threading.Lock()
        self._fps:     float = 0.0
        self._pipeline_func = None

    def set_pipeline(self, pipeline_func) -> None:
        """Set a processing pipeline function that takes a frame and returns a dict with 'annotated_frame'."""
        with self._lock:
            self._pipeline_func = pipeline_func

    # ── Control ───────────────────────────────────────────────────────────────

    def start(self) -> bool:
        """Open camera and start capture thread. Returns True on success."""
        source = self.video_path if self.video_path else self.cam_index
        self._cap = cv2.VideoCapture(source)
        if not self._cap.isOpened():
            logger.error(f"[Camera {self.cam_index}] Cannot open source: {source}")
            return False

        if not self.video_path:
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH,  self.width)
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            self._cap.set(cv2.CAP_PROP_FPS, 30)

        self._running = True
        self._thread  = threading.Thread(
            target=self._capture_loop, daemon=True, name=f"cam-{self.cam_index}"
        )
        self._thread.start()
        logger.success(f"[Camera {self.cam_index}] Started (source: {source})")
        return True

    def stop(self) -> None:
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)
        if self._cap:
            self._cap.release()
        logger.info(f"[Camera {self.cam_index}] Stopped")

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def fps(self) -> float:
        return self._fps

    # ── Capture loop (background thread) ─────────────────────────────────────

    def _capture_loop(self) -> None:
        frame_count = 0
        t0 = time.time()

        while self._running:
            ok, frame = self._cap.read()
            if not ok:
                if self.video_path:
                    # Loop video file seamlessly
                    self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                else:
                    logger.warning(f"[Camera {self.cam_index}] Frame read failed, retrying...")
                    time.sleep(0.05)
                    continue

            # Apply ML Pipeline if one is set
            with self._lock:
                pipeline_func = self._pipeline_func
            
            if pipeline_func:
                try:
                    result = pipeline_func(frame)
                    frame = result.get("annotated_frame", frame)
                except Exception as e:
                    logger.error(f"[Camera {self.cam_index}] Pipeline Error: {e}")

            # Add timestamp overlay
            ts = time.strftime("%Y-%m-%d %H:%M:%S")
            cv2.putText(
                frame, f"CAM {self.cam_index} | {ts}",
                (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2
            )

            # Encode to JPEG
            _, jpeg = cv2.imencode(
                ".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 75]
            )

            with self._lock:
                self._frame = jpeg.tobytes()

            # FPS tracking
            frame_count += 1
            elapsed = time.time() - t0
            if elapsed >= 2.0:
                self._fps   = frame_count / elapsed
                frame_count = 0
                t0          = time.time()

    # ── Frame access ──────────────────────────────────────────────────────────

    def get_frame(self) -> Optional[bytes]:
        """Return the latest JPEG frame bytes, or None if not ready."""
        with self._lock:
            return self._frame

    def get_snapshot(self) -> Optional[np.ndarray]:
        """Return the latest frame as an OpenCV BGR array."""
        if self._cap and self._cap.isOpened():
            ok, frame = self._cap.read()
            if ok:
                return frame
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Manager
# ─────────────────────────────────────────────────────────────────────────────

class CameraManager:
    """
    Registry of all active camera streams.

    Usage::

        mgr = CameraManager()
        mgr.add_camera(0)          # start webcam 0
        frame = mgr.get_frame(0)   # bytes or None
    """

    def __init__(self) -> None:
        self._cameras: Dict[int, CameraStream] = {}

    def add_camera(
        self,
        cam_index: int,
        width: int = 1280,
        height: int = 720,
        video_path: Optional[str] = None
    ) -> bool:
        if cam_index in self._cameras:
            return True  # already running
        cam = CameraStream(cam_index, width, height, video_path)
        if cam.start():
            self._cameras[cam_index] = cam
            return True
        return False

    def set_pipeline(self, cam_index: int, pipeline_func) -> None:
        """Set the processing pipeline for a specific camera."""
        if cam_index in self._cameras:
            self._cameras[cam_index].set_pipeline(pipeline_func)

    def remove_camera(self, cam_index: int) -> None:
        if cam_index in self._cameras:
            self._cameras[cam_index].stop()
            del self._cameras[cam_index]

    def get_frame(self, cam_index: int) -> Optional[bytes]:
        cam = self._cameras.get(cam_index)
        return cam.get_frame() if cam else None

    def get_snapshot(self, cam_index: int) -> Optional[np.ndarray]:
        cam = self._cameras.get(cam_index)
        return cam.get_snapshot() if cam else None

    def list_cameras(self) -> Dict[int, Dict]:
        return {
            idx: {
                "index":      idx,
                "running":    cam.is_running,
                "fps":        round(cam.fps, 1),
                "resolution": f"{cam.width}x{cam.height}",
            }
            for idx, cam in self._cameras.items()
        }

    async def mjpeg_generator(self, cam_index: int) -> AsyncGenerator[bytes, None]:
        """
        Async generator that yields MJPEG frames for StreamingResponse.

        Each chunk is a multipart frame boundary + JPEG data.
        """
        if cam_index not in self._cameras:
            logger.warning(f"[CameraManager] Camera {cam_index} not found")
            return

        while True:
            frame = self.get_frame(cam_index)
            if frame:
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n"
                    + frame
                    + b"\r\n"
                )
            await asyncio.sleep(1 / 30)  # ~30 FPS max

    def stop_all(self) -> None:
        for cam in self._cameras.values():
            cam.stop()
        self._cameras.clear()


# Module-level singleton
camera_manager = CameraManager()
