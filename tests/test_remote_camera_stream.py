"""tests/test_remote_camera_stream.py — Comprehensive tests for LAN remote camera ingestion."""
from __future__ import annotations

import os
import socket
import time
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler
import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from module8_dashboard.backend.camera_manager import (
    CameraStream,
    CameraManager,
    validate_camera_url,
)
from module8_dashboard.backend.main import app


class DummyMJPEGHandler(BaseHTTPRequestHandler):
    token = "test-secret-token"
    allowed_ip = None
    frames_served = 0

    def log_message(self, format, *args):
        pass  # Suppress test output noise

    def do_GET(self):
        # 1. IP Whitelist check
        if self.allowed_ip and self.client_address[0] != self.allowed_ip:
            self.send_response(403)
            self.end_headers()
            return

        # 2. Token check
        auth_hdr = self.headers.get("X-Camera-Token") or self.headers.get("Authorization", "").replace("Bearer ", "")
        query = self.path.split("?")[-1] if "?" in self.path else ""
        has_valid_token = (auth_hdr == self.token) or (f"token={self.token}" in query)

        if not has_valid_token:
            self.send_response(401)
            self.send_header("WWW-Authenticate", "Bearer")
            self.end_headers()
            self.wfile.write(b'{"detail":"Unauthorized"}')
            return

        if "/stream" in self.path:
            self.send_response(200)
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.end_headers()

            # Generate synthetic test frame
            img = np.zeros((240, 320, 3), dtype=np.uint8)
            cv2.putText(img, "TEST FRAME", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
            _, jpeg = cv2.imencode(".jpg", img)
            frame_bytes = jpeg.tobytes()

            try:
                for _ in range(5):
                    self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n")
                    DummyMJPEGHandler.frames_served += 1
                    time.sleep(0.05)
            except (BrokenPipeError, ConnectionResetError):
                pass
            return

        self.send_response(404)
        self.end_headers()


# ── SSRF & URL Validation Tests ───────────────────────────────────────────────

def test_validate_camera_url_rejects_empty():
    ok, err, _ = validate_camera_url("")
    assert not ok
    assert "non-empty" in err


def test_validate_camera_url_rejects_unsupported_schemes():
    ok, err, _ = validate_camera_url("ftp://192.168.1.10:8080/stream")
    assert not ok
    assert "Scheme 'ftp' not allowed" in err

    ok, err, _ = validate_camera_url("file:///etc/passwd")
    assert not ok


def test_validate_camera_url_rejects_loopback():
    os.environ["INTELLIGATE_ALLOW_LOOPBACK_CAM"] = "0"
    ok, err, _ = validate_camera_url("http://127.0.0.1:8080/stream")
    assert not ok
    assert "Loopback addresses" in err

    ok, err, _ = validate_camera_url("http://localhost:8080/stream")
    assert not ok
    assert "Loopback addresses" in err


def test_validate_camera_url_rejects_cloud_metadata():
    os.environ["INTELLIGATE_ALLOW_LOOPBACK_CAM"] = "0"
    ok, err, _ = validate_camera_url("http://169.254.169.254/latest/meta-data")
    assert not ok
    assert "Link-local addresses" in err


def test_validate_camera_url_rejects_public_ips():
    os.environ["INTELLIGATE_ALLOW_LOOPBACK_CAM"] = "0"
    ok, err, _ = validate_camera_url("http://8.8.8.8:8080/stream")
    assert not ok
    assert "Public IP addresses" in err


def test_validate_camera_url_rejects_disallowed_ports():
    ok, err, _ = validate_camera_url("http://192.168.1.10:22/stream", allowed_ports=(8080, 8443))
    assert not ok
    assert "not in the allowed camera ports" in err


def test_validate_camera_url_accepts_private_lan_ips():
    os.environ["INTELLIGATE_ALLOW_LOOPBACK_CAM"] = "0"
    ok, err, valid_url = validate_camera_url("http://192.168.1.55:8080/stream")
    assert ok
    assert valid_url == "http://192.168.1.55:8080/stream"

    ok, _, _ = validate_camera_url("http://10.0.0.12:8080/stream")
    assert ok


# ── CameraStream Ingestion & Lifecycle Tests ──────────────────────────────────

@pytest.fixture
def mjpeg_server():
    DummyMJPEGHandler.frames_served = 0
    server = HTTPServer(("127.0.0.1", 0), DummyMJPEGHandler)
    port = server.server_port
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{port}/stream"
    server.shutdown()
    server.server_close()


def test_camera_stream_remote_ingestion(mjpeg_server):
    os.environ["INTELLIGATE_ALLOW_LOOPBACK_CAM"] = "1"
    stream = CameraStream(
        cam_index=0,
        width=640,
        height=360,
        network_url=mjpeg_server,
        network_token="test-secret-token",
    )
    started = stream.start()
    assert started, "CameraStream should start successfully"
    assert stream.is_running

    # Wait for frame ingestion
    for _ in range(30):
        if stream.get_frame() is not None:
            break
        time.sleep(0.1)

    frame = stream.get_frame()
    assert frame is not None, "CameraStream should capture at least one decoded frame"
    assert stream.get_snapshot() is not None
    assert stream.actual_resolution is not None

    stream.stop()
    assert not stream.is_running


def test_camera_stream_remote_reconnect_tolerance():
    os.environ["INTELLIGATE_ALLOW_LOOPBACK_CAM"] = "1"
    # Connect to an unreachable port to verify it does not crash or exit immediately
    stream = CameraStream(
        cam_index=1,
        width=640,
        height=360,
        network_url="http://127.0.0.1:65432/stream",
        network_token="tok",
    )
    started = stream.start()
    assert started
    time.sleep(0.3)
    # Stream should remain running while retrying in background
    assert stream.is_running
    stream.stop()
    assert not stream.is_running


# ── API Endpoint Authorization & Validation Tests ─────────────────────────────

def test_network_camera_endpoint_requires_auth(tmp_path, monkeypatch):
    monkeypatch.setenv('IG_EVENT_DB', str(tmp_path / 'cam_api.db'))
    import importlib
    import module8_dashboard.backend.main as main
    importlib.reload(main)
    with TestClient(main.app) as client:
        resp = client.post("/api/cameras/0/network", json={"url": "http://192.168.1.10:8080/stream", "mode": "face"})
        assert resp.status_code == 401


def test_network_camera_endpoint_rejects_ssrf(tmp_path, monkeypatch):
    monkeypatch.setenv('IG_EVENT_DB', str(tmp_path / 'cam_api.db'))
    import importlib
    import module8_dashboard.backend.main as main
    importlib.reload(main)
    with TestClient(main.app) as client:
        # Initial admin setup
        creds = {"username": "admin", "password": "test-password-123"}
        setup_res = client.post("/api/auth/setup", json=creds)
        assert setup_res.status_code == 200
        login_res = client.post("/api/auth/login", json=creds)
        assert login_res.status_code == 200

        # Attempt SSRF attack to cloud metadata
        os.environ["INTELLIGATE_ALLOW_LOOPBACK_CAM"] = "0"
        attack_resp = client.post(
            "/api/cameras/0/network",
            json={"url": "http://169.254.169.254/latest/meta-data", "mode": "face"},
        )
        assert attack_resp.status_code == 422
        assert "Remote camera URL rejected" in attack_resp.json()["detail"]


def test_network_camera_endpoint_rejects_invalid_index(tmp_path, monkeypatch):
    monkeypatch.setenv('IG_EVENT_DB', str(tmp_path / 'cam_api.db'))
    import importlib
    import module8_dashboard.backend.main as main
    importlib.reload(main)
    with TestClient(main.app) as client:
        creds = {"username": "admin", "password": "test-password-123"}
        client.post("/api/auth/setup", json=creds)
        client.post("/api/auth/login", json=creds)
        resp = client.post(
            "/api/cameras/99/network",
            json={"url": "http://192.168.1.10:8080/stream", "mode": "face"},
        )
        assert resp.status_code == 422

