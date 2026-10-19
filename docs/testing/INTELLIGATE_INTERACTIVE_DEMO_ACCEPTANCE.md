# IntelliGate Interactive Demo & System Acceptance Test Report

**Document ID:** `docs/testing/INTELLIGATE_INTERACTIVE_DEMO_ACCEPTANCE.md`  
**Date:** 2026-10-19  
**Version:** 1.0.0  
**Status:** In Progress (Interactive Execution Mode)  
**Evaluator:** Senior Full-Stack Developer, QA Engineer, ML Engineer & FYP Demonstration Supervisor  
**Host Environment:** Windows 11 Enterprise (AMD64), Python 3.12.10, Node.js v20.x, Vite 5.4, FastAPI 0.115  
**Execution Mode:** Pure CPU Prototype, Air-Gapped Local Inference, Virtual Gate Emulation  

---

## 1. System Startup & Environmental Verification

### 1.1 Service Endpoints
- **Frontend Dashboard:** [http://localhost:5173](http://localhost:5173) (Vite Dev Server, Proxied to `:8000`)
- **Backend API:** [http://localhost:8000](http://localhost:8000) (FastAPI, Swagger UI at `/docs`)
- **WebSocket Gateway:** `ws://localhost:8000/ws` (Authenticated Session Synchronizer)
- **Database Isolation:** Database backed up to `logs/events.db.bak` prior to demonstration execution.

### 1.2 Model Cryptographic Integrity & Supply Chain

| Model Identifier | File Path | Format | Status | SHA-256 Checksum (Prefix) |
|---|---|---|---|---|
| **YuNet Face Detector** | `models/face_detection_yunet_2023mar.onnx` | ONNX | VERIFIED | `8f2383e4dd3c...` |
| **SFace Recognizer** | `models/face_recognition_sface_2021dec.onnx` | ONNX | VERIFIED | `0ba9fbfa01b5...` |
| **YOLOv8n General Detector** | `models/yolov8n.pt` | PyTorch | VERIFIED | `f59b3d833e2f...` |
| **Pakistani Plate Detector** | `models/plate_detector_yolov8n.pt` | PyTorch | VERIFIED | `d9dbcbdc5aab...` |
| **MiniFASNetV2 Face PAD** | `models/2.7_80x80_MiniFASNetV2.pth` | PyTorch | VERIFIED | `a5eb02e1843f...` |
| **OSNet-AIN Vehicle Re-ID** | `models/osnet_ain_x1_0_vehicle_reid.onnx` | ONNX | VERIFIED | `4aaad3e5db64...` |

---

## 2. Interactive Acceptance Test Matrix

Status Legend:
- `NOT TESTED`: Awaiting user execution in browser.
- `PASS`: Verified working through frontend interaction with authentic backend execution.
- `FAIL`: Defect observed in frontend, backend, or perception behavior.
- `PARTIAL`: Functional with non-critical usability or performance caveat.
- `BLOCKED`: Dependency or hardware blocker prevents execution.

| Test ID | Module | Scenario / Objective | UI Route & Action | Input Asset | Expected Outcome | Actual Outcome | Status |
|---|---|---|---|---|---|---|---|
| **TC-00** | **Remote Stream** | Remote Laptop Webcam Ingestion (LAN) | `/cameras` -> Remote / Laptop Camera form -> Connect | Live stream from `tools/laptop_cam_streamer.py` via LAN | Main PC connects to laptop MJPEG stream via authenticated header; frames ingested into `CameraStream`; live bounding boxes and overlays generated. | Automated test suite passed (12/12 unit tests); awaiting live physical laptop connection. | NOT TESTED |
| **TC-01** | **Auth / Core** | Login & Dashboard Overview | `/` (Login Form) -> Sign in | `sec_admin` / `admin-password-123` | Successful login, JWT/Session cookie issued, Dashboard stats, active modules, system health, and real-time gate status visible. | Pending User Execution | NOT TESTED |
| **TC-02** | **Biometrics** | Register Known Face & Test Recognition | `/faces` -> Enrol Person; `/cameras` or `/testing` | Live webcam face or `Colin_Powell_0001.jpg` (Enrol) & `0002.jpg` (Test) | Face detected by YuNet, 128-D SFace embedding encrypted in SQLite (`cpu_faces`), matches identity with cosine similarity > 0.36. | Pending User Execution | NOT TESTED |
| **TC-03** | **Biometrics** | Unknown Face & Review Escalation | `/cameras` (Live webcam) or `/testing` | Live un-enrolled face / `Jennifer_Aniston_0001.jpg` | SFace identifies subject as UNKNOWN; Decision policy outputs `REVIEW_REQUIRED`; review entry added to `/operations`. | Pending User Execution | NOT TESTED |
| **TC-04** | **Biometrics** | PAD Liveness Check & Advisory Limitation | `/cameras` (Live webcam) or `/testing` | Live human face vs printed/phone screen face | MiniFASNetV2 computes softmax score; reports PAD status as advisory-only without bypassing secondary security review. | Pending User Execution | NOT TESTED |
| **TC-05** | **Vehicle** | Register Vehicle & Plate | `/vehicles` -> Register Vehicle form | Plate: `LEA1234`, Owner: `Tariq Mahmood`, Body Crop: `data/pakistan_plates/vehicle_body_crops/crop_00000.jpg` | Vehicle saved in `logs/cpu_vehicles.h5`; OSNet-AIN 512-D embedding extracted; visible in vehicle table. | Pending User Execution | NOT TESTED |
| **TC-06** | **Vehicle** | Automatic Authorization via Trusted Perception | `/testing` -> Run vehicle media test | `data/pakistan_plates/combined_clean/test_a/images/00000.jpg` | YOLOv8n locates plate; EasyOCR extracts plate matching registered `LEA1234`; SoftwareRuntime grants `ALLOW`; gate opens. | Pending User Execution | NOT TESTED |
| **TC-07** | **Vehicle** | Unregistered Plate Perception | `/testing` -> Run vehicle media test | `data/pakistan_plates/combined_clean/test_a/images/00001.jpg` | Plate recognized but absent from registered registry; decision routes to `DENY` or `REVIEW_REQUIRED`; gate remains CLOSED. | Pending User Execution | NOT TESTED |
| **TC-08** | **Vehicle** | Deep Vehicle Re-ID Advisory Matching | `/testing` or `/vehicles` -> Re-ID Advisory Check | Registered vehicle crop vs probe crop | OSNet-AIN outputs cosine similarity; flagged strictly as `ADVISORY` (cannot bypass exact plate match requirement). | Pending User Execution | NOT TESTED |
| **TC-09** | **Vision** | Night-Vision / Infrared Replay | `/cameras` -> Demo Replay Source: `llvip_infrared` | `logs/demo_data/llvip_infrared.mp4` | Genuine thermal frames streamed via MJPEG; CLAHE contrast enhancement active; pedestrian silhouette tracking functional. | Pending User Execution | NOT TESTED |
| **TC-10** | **Surveillance**| Stranger Memory & Queue Tracking | `/` (Dashboard) & `/events` -> Inspect Visitor log | Repeated unknown face appearances | Stranger identity assigned transient tracker ID; dwell time and visit count accumulated in memory table. | Pending User Execution | NOT TESTED |
| **TC-11** | **Safety** | Safety Interlocks: Obstruction, STOP, LOCKDOWN | `/operations` -> Toggle Obstruction, Emergency STOP, LOCKDOWN | Operator UI action buttons | Highest priority safety interlock: Obstruction prevents gate closure; STOP halts motion; LOCKDOWN seals barrier against all credentials. | Pending User Execution | NOT TESTED |
| **TC-12** | **Audit** | Event History, Alerts & Reports | `/events`, `/alerts`, `/operations` (Generate Report) | Real-time events generated during demo | Chronological audit trail persisted in SQLite; alerts categorized by severity; verifiable reports with zero synthetic fabrication. | Pending User Execution | NOT TESTED |

---

## 3. Automated Test Verification Log

- **Remote Camera & SSRF Protection Test Suite:** `tests/test_remote_camera_stream.py` (12/12 PASSED)
- **Core Security & Regression Suite:** `tests/test_api.py`, `tests/test_software_runtime.py`, `tests/test_milestone1_security.py`, `tests/test_capture_startup.py` (41/41 PASSED)
- **Frontend Production Bundle:** `npm run build` compiled cleanly into `dist/` in 3.47s.
- **WebSocket Gateway:** Native `websockets` library installed in `.venv`; authenticated session synchronization verified active.

---

## 4. Live Interactive Findings & Fix Log

*(Updated during live interactive demonstration)*

