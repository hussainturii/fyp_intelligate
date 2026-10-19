#!/usr/bin/env python3
"""
tools/laptop_cam_streamer.py — Lightweight Authenticated LAN Camera Streamer

Runs on a secondary device (e.g. laptop) to securely stream webcam frames
over the private local Wi-Fi network to the IntelliGate Main PC server.

Security features:
- Binds to LAN interface.
- Enforces Bearer / X-Camera-Token header authentication (with query-param fallback).
- Optional client IP whitelist (--allowed-ip) to only accept requests from the Main PC.
- No heavy frameworks required — standard library HTTP server + opencv-python.
- Graceful webcam release on Ctrl+C.
"""
from __future__ import annotations

import argparse
import ipaddress
import secrets
import socket
import sys
import threading
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

try:
    import cv2
except ImportError:
    print("\n[ERROR] opencv-python is not installed.")
    print("Please install it by running: pip install opencv-python\n")
    sys.exit(1)


class WebcamCapture:
    """Threaded camera capture to keep the latest frame fresh without buffering latency."""

    def __init__(self, device_index: int = 0, width: int = 960, height: int = 540, fps: int = 25):
        self.device_index = device_index
        self.target_width = width
        self.target_height = height
        self.target_fps = fps
        self._cap = None
        self._running = False
        self._frame_bytes = None
        self._lock = threading.Lock()
        self._thread = None
        self.actual_fps = 0.0

    def start(self) -> bool:
        if sys.platform == "win32":
            self._cap = cv2.VideoCapture(self.device_index, cv2.CAP_DSHOW)
            if not self._cap.isOpened():
                self._cap.release()
                self._cap = cv2.VideoCapture(self.device_index)
        else:
            self._cap = cv2.VideoCapture(self.device_index)

        if not self._cap.isOpened():
            return False

        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.target_width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.target_height)
        self._cap.set(cv2.CAP_PROP_FPS, self.target_fps)
        self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        self._running = True
        self._thread = threading.Thread(target=self._capture_worker, daemon=True, name="WebcamWorker")
        self._thread.start()
        return True

    def _capture_worker(self):
        frame_interval = 1.0 / max(1, self.target_fps)
        count = 0
        last_metric_time = time.monotonic()

        while self._running:
            start_t = time.monotonic()
            ok, frame = self._cap.read()
            if not ok:
                time.sleep(0.05)
                continue

            # Scale if larger than requested
            if frame.shape[1] > self.target_width or frame.shape[0] > self.target_height:
                scale = min(self.target_width / frame.shape[1], self.target_height / frame.shape[0])
                frame = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

            # Encode as JPEG
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 80]
            ok_enc, buf = cv2.imencode(".jpg", frame, encode_param)
            if ok_enc:
                with self._lock:
                    self._frame_bytes = buf.tobytes()

            count += 1
            now = time.monotonic()
            if now - last_metric_time >= 1.0:
                self.actual_fps = count / (now - last_metric_time)
                count = 0
                last_metric_time = now

            elapsed = time.monotonic() - start_t
            sleep_time = max(0.0, frame_interval - elapsed)
            if sleep_time > 0:
                time.sleep(sleep_time)

        if self._cap:
            self._cap.release()

    def get_latest_jpeg(self) -> bytes | None:
        with self._lock:
            return self._frame_bytes

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)
        if self._cap and self._cap.isOpened():
            self._cap.release()


class StreamHandler(BaseHTTPRequestHandler):
    camera: WebcamCapture = None
    token: str = ""
    allowed_ip: str | None = None

    def log_message(self, format, *args):
        # Concise logging
        sys.stderr.write(f"[{time.strftime('%H:%M:%S')}] {self.client_address[0]} - {args[0]} {args[1]}\n")

    def _is_authorized(self) -> bool:
        # 1. Check IP restriction if configured
        client_ip = self.client_address[0]
        if self.allowed_ip:
            if client_ip != self.allowed_ip and client_ip not in ("127.0.0.1", "::1"):
                return False

        # 2. Check X-Camera-Token header
        header_token = self.headers.get("X-Camera-Token")
        if header_token and secrets.compare_digest(header_token.strip(), self.token):
            return True

        # 3. Check Authorization: Bearer <token>
        auth_header = self.headers.get("Authorization", "")
        if auth_header.startswith("Bearer "):
            bearer_token = auth_header[7:].strip()
            if secrets.compare_digest(bearer_token, self.token):
                return True

        # 4. Fallback: check query parameter ?token=
        parsed = urlparse(self.path)
        query_params = parse_qs(parsed.query)
        if "token" in query_params:
            if secrets.compare_digest(query_params["token"][0].strip(), self.token):
                return True

        return False

    def do_GET(self):
        parsed = urlparse(self.path)

        if parsed.path == "/health":
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"status":"ok","service":"intelligate_laptop_cam"}')
            return

        if not self._is_authorized():
            self.send_response(HTTPStatus.UNAUTHORIZED)
            self.send_header("Content-Type", "application/json")
            self.send_header("WWW-Authenticate", 'Bearer realm="IntelliGate Camera"')
            self.end_headers()
            self.wfile.write(b'{"detail":"Unauthorized camera stream access. Provide valid token."}')
            return

        if parsed.path in ("/stream", "/video_feed", "/mjpeg"):
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
            self.send_header("Pragma", "no-cache")
            self.send_header("Expires", "0")
            self.send_header("Connection", "close")
            self.end_headers()

            boundary = b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
            last_sent = None

            try:
                while True:
                    frame = self.camera.get_latest_jpeg()
                    if frame and frame is not last_sent:
                        self.wfile.write(boundary + frame + b"\r\n")
                        last_sent = frame
                    time.sleep(0.033)  # ~30 FPS polling
            except (BrokenPipeError, ConnectionResetError):
                pass
            return

        if parsed.path == "/snapshot":
            frame = self.camera.get_latest_jpeg()
            if not frame:
                self.send_response(HTTPStatus.SERVICE_UNAVAILABLE)
                self.end_headers()
                return
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "image/jpeg")
            self.send_header("Content-Length", str(len(frame)))
            self.end_headers()
            self.wfile.write(frame)
            return

        self.send_response(HTTPStatus.NOT_FOUND)
        self.end_headers()


def get_local_ip() -> str:
    """Best effort detection of primary outbound LAN IP."""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        # Connecting to a public IP doesn't actually send packets but selects the active network interface
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
    except Exception:
        ip = "127.0.0.1"
    finally:
        s.close()
    return ip


def main():
    parser = argparse.ArgumentParser(description="IntelliGate Laptop Webcam LAN Streamer")
    parser.add_argument("--port", type=int, default=8080, help="Port to listen on (default: 8080)")
    parser.add_argument("--camera", type=int, default=0, help="Webcam device index (default: 0)")
    parser.add_argument("--token", type=str, default="", help="Custom secret token (auto-generated if omitted)")
    parser.add_argument("--allowed-ip", type=str, default=None, help="Main PC IP address to allow (optional)")
    parser.add_argument("--width", type=int, default=960, help="Capture width (default: 960)")
    parser.add_argument("--height", type=int, default=540, help="Capture height (default: 540)")
    parser.add_argument("--fps", type=int, default=25, help="Capture FPS (default: 25)")
    args = parser.parse_args()

    # Generate token if not specified
    token = args.token.strip() if args.token else secrets.token_urlsafe(16)

    # Validate allowed_ip if passed
    if args.allowed_ip:
        try:
            ipaddress.ip_address(args.allowed_ip)
        except ValueError:
            print(f"[ERROR] Invalid --allowed-ip: '{args.allowed_ip}' is not a valid IP address.")
            sys.exit(1)

    print("=" * 72)
    print("  INTELLIGATE — LAPTOP WEBCAM LAN STREAMER")
    print("=" * 72)

    # Start webcam capture
    print(f"[*] Initializing webcam index {args.camera} ({args.width}x{args.height} @ {args.fps} FPS)...")
    cam = WebcamCapture(device_index=args.camera, width=args.width, height=args.height, fps=args.fps)
    if not cam.start():
        print(f"[ERROR] Cannot open webcam device index {args.camera}.")
        print("Check if another application (Teams, Zoom, Camera app) is using the webcam.")
        sys.exit(1)

    # Wait for first frame
    for _ in range(50):
        if cam.get_latest_jpeg() is not None:
            break
        time.sleep(0.1)

    if cam.get_latest_jpeg() is None:
        print("[WARNING] Webcam opened, but first frame took longer than expected.")

    local_ip = get_local_ip()

    # Configure HTTP handler
    StreamHandler.camera = cam
    StreamHandler.token = token
    StreamHandler.allowed_ip = args.allowed_ip

    ThreadingHTTPServer.allow_reuse_address = True
    server = None
    active_port = args.port
    candidate_ports = [args.port]
    if args.port == 8080:
        candidate_ports.extend([8088, 8888, 5000, 9000, 9090, 8443])

    for p in candidate_ports:
        try:
            server = ThreadingHTTPServer(("0.0.0.0", p), StreamHandler)
            active_port = p
            break
        except (PermissionError, OSError) as bind_err:
            if p == candidate_ports[-1]:
                print(f"\n[ERROR] Could not bind to candidate ports ({candidate_ports}): {bind_err}")
                print("Windows Hyper-V or another service has reserved these ports.")
                print("Try specifying a custom port, for example: python laptop_cam_streamer.py --port 8088\n")
                cam.stop()
                sys.exit(1)
            continue

    print("\n[SUCCESS] Webcam streamer is RUNNING!")
    print(f"[*] Stream URL:  http://{local_ip}:{active_port}/stream")
    print(f"[*] Auth Token:  {token}")
    if args.allowed_ip:
        print(f"[*] IP Whitelist: Requests restricted to Main PC ({args.allowed_ip})")
    else:
        print(f"[*] IP Whitelist: Open to LAN with valid token authentication")
    print("\nIn IntelliGate Dashboard -> Cameras tab:")
    print(f"  URL:   http://{local_ip}:{active_port}/stream")
    print("Press Ctrl+C to safely stop webcam and close server.")
    print("=" * 72)

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[*] Stopping webcam and shutting down server...")
    finally:
        cam.stop()
        server.server_close()
        print("[*] Webcam released. Streamer stopped cleanly.\n")


if __name__ == "__main__":
    main()
