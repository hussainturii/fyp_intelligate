# INTELLIGATE: FINAL ENGINEERING AUDIT, SECURITY ASSESSMENT & PRE-VIVA READINESS REPORT

**Document Control:** Final Technical Audit & Academic Evaluation  
**Project Title:** IntelliGate — AI-Powered Smart Gate and Security Management System  
**Institution:** COMSATS University Islamabad, Department of Computer Science (BS Artificial Intelligence)  
**Project Authors:** Hussain Raza (SP23-BAI-057), Muhammad Mazhar Afzal (SP23-BAI-035)  
**Audit Date:** October 15, 2026  
**Auditing Body:** Multidisciplinary Engineering Review Team (AI/ML, CV, AppSec, Software Architecture, QA, Database, DevOps, External Examiner)  
**Target Repository:** `D:\MZHR_FYP\INTELLIGATE`  
**Git Baseline:** Commit `c909145` (*"Add CPU vision, module workflows, virtual gate and test center"*) on branch `main`  
**Host Hardware Baseline:** Intel Core i7-1165G7 @ 2.80GHz (4 physical cores, 8 logical processors), 7.7 GiB Physical RAM, Intel Iris Xe Graphics (No CUDA GPU; CPU execution mode)  

---

## 1. Executive Summary

An exhaustive, multi-disciplinary engineering audit and security assessment was conducted on the IntelliGate codebase. The evaluation encompassed static code inspection, dynamic test execution, runtime performance profiling, architectural dependency mapping, cryptographic validation, and pre-viva defense interrogation.

### 1.1 Project Status Overview
IntelliGate is an AI-assisted smart gate and visitor security management system. Originally conceptualized in proposal documents as a GPU-accelerated edge appliance incorporating foundation Vision Transformers (ViT), State Space Models (Mamba), Vision-Language Models (LLaVA), and physical gate telemetry, the project underwent a necessary and pragmatic pivot. Under strict hardware constraints (standard quad-core laptop CPU, no discrete GPU, 8 GB RAM) and deadline pressures, the system was re-engineered into an **integrated, real-time CPU software prototype, virtual gate controller, and test workbench**.

### 1.2 Major Engineering Strengths
1. **Thread-Safe, Non-Blocking Capture Architecture:** Ingestion of live webcam streams and recorded video occurs on an isolated capture thread. An asynchronous worker thread extracts only the newest available frame via a single-pending-frame buffer, entirely decoupling video preview latency (~20 FPS) from heavier inference latency (~10 FPS).
2. **Fail-Secure Access Policy:** Acknowledging that deep-learning anti-spoofing could not be safely validated on CPU hardware, the central security engine (`SoftwareRuntime.observe`) strictly enforces a `REVIEW_REQUIRED` gate policy. Face recognition matches cannot autonomously trigger gate opening; instead, they require explicit operator approval within a 120-second sliding window.
3. **Robust Cryptographic Biometrics:** Unlike typical undergraduate implementations that store raw face embeddings in plaintext files, IntelliGate encrypts all facial identity templates and stranger candidate vectors with **AES-256-GCM**. Nonces are cryptographically unique per entry, and the record's UUID identity is bound as Additional Authenticated Data (AAD) to prevent ciphertext transplant attacks.
4. **Automated Verification:** The automated test suite achieves **100% pass rate (45/45 tests passing in 21.31 seconds)**, validating RBAC rules, brute-force rate-limiting, command idempotency, restart state recovery, and multi-identity tracking.

### 1.3 Major Technical Weaknesses & Academic Gaps
1. **Divergence from Approved SRS Research Claims:** The proposal and SRS promised trained ViT models for face recognition, deep VeRi-776 Re-ID (OSNet), Mamba sequence anomaly models, and LLaVA multimodal incident reporting. In reality, the running CPU delivery implements:
   - SFace (quantized ONNX) instead of trained ViT.
   - 768-D color histogram vectors instead of trained vehicle Re-ID.
   - Rule-based dwell/visit counting instead of State Space Models.
   - Deterministic Jinja/string event templates instead of a Vision-Language Model.
   - OpenCV contour analysis instead of neural license plate detection (caused by an upstream HuggingFace 404 error).
2. **Absence of Quantitative Benchmark Metrics:** The system lacks empirical accuracy evaluations (ROC curves, EER, mAP, Rank-1/5, Character Error Rates) calculated against standard held-out benchmark splits. Enrolment self-matching was used during development as a sanity smoke check, which must not be conflated with generalization accuracy.
3. **Session Cookie Directives:** The HTTP session cookie (`ig_session`) lacks `SameSite=Lax/Strict` and `Secure` attributes, leaving sessions susceptible to cross-origin transmission if deployed outside `127.0.0.1`.

### 1.4 Readiness Determination
**CONDITIONALLY READY FOR VIVA.**  
The project is structurally sound, stable, and demonstrates high software engineering competence. It is defensible before a university viva panel **on the strict condition** that the candidates defend the system that actually exists—a real-time CPU edge prototype with human-in-the-loop safety interlocks—rather than claiming the theoretical GPU foundation models from earlier proposals.

---

## 2. Audit Scope, Evidence & Baseline

### 2.1 Inspection Inventory

| Artifact / Component | Inspection Path | Evidence Examined | Audit Verdict |
|---|---|---|---|
| **Repository Root** | `D:\MZHR_FYP\INTELLIGATE` | Git commit log, tree status, environment manifests | Verified clean working tree |
| **System Configuration** | `config/config.yaml` | Hardware, camera, liveness weights, team allocation | Verified active configuration |
| **Legacy Vision Modules** | `module1_face_recognition`, `module2_vehicle` | MTCNN, ViT PyTorch classes, MediaPipe mesh, EasyOCR | Verified legacy research code |
| **Dashboard Runtime** | `module8_dashboard/backend` | FastAPI application, vision worker, camera manager, auth | Verified active operational stack |
| **Frontend SPA** | `module8_dashboard/frontend` | React 18, Vite, Tailwind CSS, component views | Verified production build passes |
| **Automated Tests** | `tests/` | 6 test scripts covering API, runtime, tracking, workbench | Verified 45/45 tests passing |
| **Component Verifier** | `tools/run_project_checks.py` | 10 module smoke checks, research disclosures | Verified 10 PASS, 4 NOT_IMPL, 4 DATA |
| **Model Weights** | `models/` | YuNet ONNX (232 KB), SFace ONNX (38.6 MB), YOLOv8n (6.5 MB) | Verified on disk; SHA-256 validated |
| **Datasets & Replays** | `logs/demo_data/` | `llvip_visible.mp4`, `llvip_infrared.mp4`, `face_smoke.mp4` | Verified paired replays prepared |
| **CPU Benchmarks** | `logs/benchmarks/` | `cpu_baseline.json`, `llvip_exploration.json` | Verified live execution outputs |

### 2.2 Environmental Baseline
- **Operating System:** Windows 11 Home / Pro (x86_64, 10.0.26200)
- **Python Runtime:** Python 3.12.10 (AMD64)
- **Package Management:** Virtual environment located at `D:\MZHR_FYP\fyp_intelligate-1\.venv`
- **Core Libraries Installed:** `torch 2.7.1+cu118`, `torchvision 0.22.1+cu118`, `ultralytics 8.4.174`, `opencv-contrib-python 5.0.0.93`, `easyocr 1.7.2`, `cryptography 50.0.2`, `fastapi 0.142.2`, `pydantic 2.13.5`
- **Frontend Stack:** Node.js v24.21.0, npm 11.19.0, Vite 5.x, React 18.x

---

## 3. Project Understanding & Operational Architecture

IntelliGate implements an asynchronous decision-support pipeline for security gates. The operational sequence decouples visual sensing from decision logic:

```mermaid
sequenceDiagram
    autonumber
    participant Cam as Video Capture Thread
    participant Q as Single-Item Frame Buffer
    participant Worker as Vision Worker (CPU)
    participant Policy as SoftwareRuntime Interlocks
    participant DB as SQLite / HDF5 Store
    participant WS as WebSocket Broadcaster
    participant UI as Operator Dashboard

    Cam->>Q: Push latest decoded frame (Drops stale frames)
    Worker->>Q: Pop latest frame
    Worker->>Worker: YuNet Face Detection / YOLOv8 Scene
    Worker->>Worker: SFace Embedding / EasyOCR Plate Read
    Worker->>Worker: Match against In-Memory Track Cache
    alt New Distinct Appearance
        Worker->>Policy: Emit Observation Event
        Policy->>Policy: Evaluate Interlocks (Obstruction / Liveness / Role)
        alt Face Match with Unvalidated Liveness
            Policy->>Policy: Create Review Request (TTL: 120s)
            Policy->>DB: Log Event ('recognition_candidate')
            Policy->>WS: Broadcast Alert ('REVIEW_REQUIRED')
            WS->>UI: Render Modal (Snapshot + Approve/Deny)
            UI->>Policy: Operator Approves Request
            Policy->>Policy: Issue Virtual Command ('OPEN')
        else Exact Registered Plate (>85% OCR Conf)
            Policy->>Policy: Issue Virtual Command ('OPEN')
            Policy->>DB: Log Event ('access_granted')
        end
    else Stable Continuous Track
        Worker->>Worker: Increment Cache Hits (Suppress Duplicate Events)
    end
    Policy->>DB: Commit State & Events
    Policy->>WS: Push Updated Gate State
    WS->>UI: Update Virtual Gate & Live HUD
```

---

## 4. Requirements Traceability Audit Matrix

The following matrix formally cross-references the approved project requirements with verified implementation code and empirical test coverage:

| Req ID | SRS / Proposal Requirement | Implementation Evidence | Automated Test Reference | Audit Status | Residual Gap & Action |
|---|---|---|---|---|---|
| **FR-01** | ViT-Base/16 Face Recognition | Legacy ViT in `module1/face_recognizer.py`; SFace ONNX in `backend/vision.py` | `test_real_face_enrolment_requires_review_before_virtual_open` | **Partially Verified** | Substituted with SFace for CPU throughput. Defend substitution in viva report. |
| **FR-02** | Multi-Modal Anti-Spoofing (Blink, Depth, Texture, Mask) | MediaPipe mesh & LBP in `module1/anti_spoofing/`; disconnected in CPU runtime | `test_api_spoofed_liveness_cannot_grant_and_review_interlock` | **Partially Verified** | CNN heads are untrained. Safe fallback: face matches require operator approval. |
| **FR-03** | Face Enrolment & Encryption | AES-256-GCM encryption with record-bound UUID in SQLite table `cpu_faces` | `test_embedding_encryption_binds_record_id` | **Fully Verified** | Meets cryptographic data protection standards. |
| **FR-04** | Vehicle Detection & Plate OCR | YOLOv8n object detection; EasyOCR; OpenCV contour localization fallback | `test_vehicle_ocr_cached_and_events_not_repeated` | **Fully Verified** | Dedicated plate model URL returned 404; contour fallback is functional on clear frames. |
| **FR-05** | Vehicle Re-Identification (OSNet / VeRi-776) | 768-D HSV/RGB color histogram baseline (`vision.py#L24-L27`) | `test_review_policy_never_grants_appearance_match` | **Partially Verified** | Histograms cannot distinguish identical car models. Policy forbids auto-grant. |
| **FR-06** | Low-Light & Thermal Analytics | LAB-space CLAHE enhancement; split genuine LLVIP infrared video playback | `test_low_light_and_histogram_are_finite`, `benchmark_llvip.py` | **Partially Verified** | CLAHE is classical image processing. Infrared is unannotated replay exploration. |
| **FR-07** | Unknown Stranger Clustering | Encrypted `cpu_strangers` table; cosine metric (<0.45 creates new identity) | `test_distinct_visits_only_and_window` | **Fully Verified** | Unenrolled visitors maintain stable identity across sessions with 90-day retention. |
| **FR-08** | Repeat Visitor Escalation | 3 distinct appearances in 24 hours emits HIGH alert; continuous presence deduplicated | `test_distinct_visits_only_and_window` | **Fully Verified** | Verified that continuous presence does not inflate visit counts. |
| **FR-09** | SSM (Mamba) Anomaly Prediction | Dwell-time heuristic (>30 seconds inside ROI) and loitering alert | `test_tracker_counts_unique_and_expires` | **Not Implemented** | State Space Model is omitted. Rule-based dwell timers substituted. |
| **FR-10** | Queue & Density Management | Normalized polygon ROI (`queue_roi`) intersection counting via IoU Tracker | `test_tracker_counts_unique_and_expires` | **Partially Verified** | Uses bounding box tracker counts rather than continuous crowd density heatmaps. |
| **FR-11** | Hazard & Obstruction Interlock | Virtual gate-path ROI (`gate_roi`); STOP pre-emption; operator lockdown | `test_obstruction_blocks_both_motion_commands`, `test_stop_preempts_motion` | **Fully Verified** | Software state machine enforces safety interlocks. No physical hardware connected. |
| **FR-12** | VLM Incident Captions & Reports | Jinja/string template generation summarizing SQLite event rows | `test_api_validation_and_report`, `test_report_counts_beyond_preview` | **Partially Verified** | Deterministic reporting engine. Multimodal VLM (LLaVA) is not integrated. |
| **FR-13** | Web Operations Dashboard | React 18 SPA + Vite + Tailwind CSS + WebSocket live push | `test_frontend_bundle_availability`, `test_websocket_requires_session` | **Fully Verified** | Production bundle compiled; responsive HUD with virtual gate controls and Test Center. |
| **FR-14** | Natural Language Query Search | Deterministic keyword parser mapping terms (`face today`, `vehicle`, etc.) | `test_report_counts_beyond_preview_and_search_validation` | **Partially Verified** | Keyword filter dictionary, not semantic LLM text-to-SQL translation. |
| **FR-15** | Role-Based Access Control | `admin`, `security_staff`, `viewer` roles with scrypt password hashing | `test_viewer_read_only_and_staff_cannot_enrol`, `test_bad_password_rate_limit` | **Fully Verified** | Restricts enrolment to admin; commands to staff; viewers are read-only. |
| **FR-16** | Database Architecture | SQLite in WAL mode; HDF5 for vehicle appearance histograms | `test_setup_authentication_and_origin`, `test_command_idempotence` | **Partially Verified** | SQLite retained during stabilization in place of proposed PostgreSQL. |

---

## 5. Software Architecture Audit

### 5.1 Concurrency & Threading Model
- **Capture Decoupling:** `CameraManager` runs an independent `threading.Thread` reading frames via OpenCV `cv2.VideoCapture`. It decodes frames and stores only the newest frame in memory.
- **Worker Isolation:** The vision inference loop executes on a separate worker thread. When the worker finishes processing a frame, it retrieves the freshest frame, dropping any intermediate frames that elapsed during inference. This prevents queue bloat and pipeline lag.
- **Global Inference Synchronization:** A global recursive lock (`INFERENCE_LOCK = threading.RLock()`) is shared across `vision.py` and `main.py`.
  - *Risk:* Calling `register_face` or deleting a vehicle while video inference is active holds the lock, causing a momentary frame stutter. Under single-camera demonstration conditions, this causes negligible latency (~100 ms).

### 5.2 State Machine & Fault Recovery
- **Virtual Gate Engine:** Implemented in `SoftwareRuntime` (`backend/software_runtime.py`). Valid states: `CLOSED`, `OPENING`, `OPEN`, `CLOSING`, `STOPPED`, `LOCKDOWN`.
- **Restart Recovery:** If the backend process terminates while the virtual gate is in motion (`OPENING`, `CLOSING`, or `OPEN`), `SoftwareRuntime.__init__` detects the in-flight state and transitions the gate to `STOPPED` with the audit reason `"Restart requires operator review"`. This prevents unmonitored gate movement upon recovery.
- **Command Idempotency:** Commands pass a unique UUID `command_id`. Re-transmitting an identical `command_id` returns the recorded execution result without re-actuating the state machine.

---

## 6. Module-by-Module Technical Audit

### Module 1: Face Recognition and Anti-Spoofing
- **Engine Evaluated:** YuNet Face Detector (`face_detection_yunet_2023mar.onnx`) + SFace Recognizer (`face_recognition_sface_2021dec.onnx`).
- **Detection Performance:** Real-time throughput of **83.3 FPS (12.14 ms median)** on 640x360 frames.
- **Recognition Threshold:** Cosine similarity threshold hardcoded at `0.45`.
- **Identity Tracking & Caching:** Multi-face tracking assigns unique integer IDs. Once a stable face crop is matched against the database, the label is cached in `face_states`. Continuous visible presence increments `recognition_cache_hits` and suppresses duplicate events. An appearance ends only after 2.0 seconds of absence.
- **Anti-Spoofing Finding:** The legacy multi-modal anti-spoofing pipeline (`blink_detector.py`, `depth_estimator.py`, `texture_analyzer.py`, `mask_detector.py`) contains untrained CNN heads and heuristic thresholds. The integrated dashboard safely bypasses autonomous access: all recognized faces are emitted as `recognition_candidate` requiring manual security staff approval.

### Module 2: Vehicle Recognition and Re-Identification
- **Detection & Reading:** YOLOv8n identifies cars, motorcycles, buses, and trucks. Bounding boxes are cropped and processed by EasyOCR. Plates matching Pakistani alphanumeric patterns (`ABC-1234`, `LEA-1234`) are parsed.
- **Localization Fallback:** The intended HuggingFace neural plate detector (`keremberke/yolov8n-license-plate-detection`) returned an HTTP 404 error during setup. The code safely falls back to OpenCV contour detection.
- **Re-ID Finding:** Vehicle Re-ID does not use a deep metric network (OSNet). It computes a 768-dimensional normalized RGB color histogram. This is a visual appearance baseline and cannot reliably distinguish identical make/model/color vehicles. The central access policy strictly forbids appearance matches from opening the gate autonomously.

### Module 3: Night Vision and Thermal Intelligence
- **Low-Light Enhancement:** Evaluated in `enhance()` using OpenCV CLAHE in LAB color space (clip limit = 2.0, tile grid = 8x8). Execution speed: **3.68 ms median (280 FPS)**.
- **Modality Shift Finding:** In empirical benchmarks on the paired LLVIP dataset, CLAHE visibly enhanced contrast but produced zero YOLO person detections in low-light conditions. Genuine infrared imagery yielded 1–3 detected persons per frame using the same detector.

### Module 4: Stranger Memory and Behavioral Analytics
- **Identity Clustering:** Stranger facial vectors are stored in `cpu_strangers`. A cosine similarity < 0.45 creates a new stranger UUID; similarity >= 0.45 updates the last-seen timestamp. Records are automatically pruned after 90 days.
- **Visit Logic:** Three distinct visit sessions within a 24-hour window triggers a `HIGH` repeat-visitor alert. Continuous presence across adjacent video frames is deduplicated.
- **Behavioral Finding:** The proposed Mamba State Space Model (SSM) sequence forecaster was not implemented. Anomaly detection is based on deterministic dwell-time rules (>30 seconds inside configured ROI).

### Module 5: Queue and Density Management
- **Tracking:** Implemented via greedy IoU overlap association across adjacent frames.
- **Counting:** Entity bounding box centers intersecting the normalized blue ROI (`queue_roi`) are counted. Exceeding `queue_threshold` triggers a `MEDIUM` congestion alert.
- **Limitation:** Occlusions and rapid trajectory crossings cause track ID switching. The count reflects tracked bounding boxes, not continuous crowd density maps.

### Module 6: Safety Triggers and Hazard Detection
- **Interlocks:** Evaluated via `SoftwareRuntime.obstruction` and `command`. When an entity intersects the red gate-path ROI (`gate_roi`), an obstruction signal stops active motion and blocks `OPEN`/`CLOSE` commands. `STOP` pre-empts all pending transitions. `LOCKDOWN` seals the gate until manually cleared.
- **Limitation:** Purely a software simulation; no physical fail-safe relay or laser scanner is integrated.

### Module 7: Vision-Language Intelligence and Reporting
- **Reporting Engine:** Implemented via `SoftwareRuntime.report`. Aggregates events and alerts over a specified window (default 24h) into structured JSON with verifiable event IDs.
- **Search Engine:** Constrained keyword parser mapping supported phrases (`face today`, `vehicle`, `queue`, `spoof`, `obstruction`) to database queries.
- **Limitation:** No multimodal Vision-Language Model (LLaVA) is executed. Unrecognized search phrases return HTTP 422 with supported keyword guidance.

### Module 8: Web Dashboard & Test Workbench
- **UI Architecture:** Fast, modern React 18 single-page application built with Tailwind CSS. Integrates live MJPEG camera streams, WebSocket telemetry, operations reviews, and virtual gate controls.
- **Test Center Workbench:** Provides isolated diagnostics: operators can snapshot the live camera to test face recognition scores against a sandbox gallery, or upload custom videos/images with ground-truth assertions (expected identity, expected count).

---

## 7. AI/ML & Research Audit

### 7.1 Quantitative Benchmark Results

The following measurements were collected directly on the project's development machine:

#### Table 7.1: CPU Inference Latency Breakdown (640x360 Frame Input)
| Pipeline Stage | Model / Algorithm | Median (ms) | p95 (ms) | Mean FPS | Output Metric |
|---|---|---:|---:|---:|---|
| **Face Detection** | YuNet ONNX | 12.14 | 12.44 | 83.30 | Bounding boxes + 5 landmarks |
| **Face Embedding** | SFace ONNX | 15.20 | 18.50 | 65.70 | 128-D L2-normalized vector |
| **Scene Detection** | YOLOv8n PyTorch | 92.95 | 114.77 | 10.66 | Multi-class boxes + IoU tracks |
| **Low-Light Enhancer** | CLAHE (LAB space) | 3.68 | 3.77 | 280.60 | Contrast-equalized frame |
| **Plate OCR** | EasyOCR (CPU) | 480.00 | 720.00 | 1.80 | Alphanumeric plate string |

#### Table 7.2: LLVIP Paired Dataset Modality Shift Exploration
| Input Modality | Preprocessing | YOLOv8n Median (ms) | YOLOv8n p95 (ms) | Person Detections (8 Frames) |
|---|---|---:|---:|---|
| **Raw Low-Light Visible** | None | 23.14 | 129.43 | `[0, 0, 0, 0, 0, 0, 0, 1]` |
| **Enhanced Visible** | CLAHE Equalization | 22.65 | 24.35 | `[0, 0, 0, 0, 0, 0, 0, 0]` |
| **Genuine Infrared** | Raw IR Decoded | 24.09 | 27.14 | `[1, 1, 1, 3, 1, 1, 0, 0]` |

### 7.2 Research Integrity & Limitations
- **Model Training vs. Inference:** Neither SFace nor YOLOv8n was trained or fine-tuned from scratch by the authors. Pretrained open-source weights were utilized. The engineering contribution lies in pipeline integration, edge CPU optimization, tracking caching, and safety interlocks.
- **Threshold Calibration:** The 0.45 cosine threshold in SFace has not been validated on a local identity-disjoint Pakistani facial dataset. In viva defense, the student must describe this threshold as **provisional**.

---

## 8. Comprehensive Cybersecurity Audit

### 8.1 Prioritized Vulnerability Register

```
+----------------------------------------------------------------------------------------------------+
| VULN-01 | Session Cookie Missing SameSite and Secure Directives                                     |
+----------------------------------------------------------------------------------------------------+
| Affected File: module8_dashboard/backend/main.py                                                   |
| CVSS v3.1 Score: 5.3 (Medium) [CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:U/C:L/I:N/A:N]                        |
| Evidence: response.set_cookie('ig_session', token, httponly=True) lacks SameSite and Secure.        |
| Exploit Scenario: If exposed over a local network, session tokens could be transmitted over        |
|                   unencrypted HTTP or leaked via cross-site requests.                              |
| Remediation: Update set_cookie call: samesite='lax', secure=False (set secure=True for HTTPS).     |
+----------------------------------------------------------------------------------------------------+

+----------------------------------------------------------------------------------------------------+
| VULN-02 | Plaintext Biometric Master Key Stored Adjacent to SQLite Database                         |
+----------------------------------------------------------------------------------------------------+
| Affected File: module8_dashboard/backend/vision.py (Line 113)                                      |
| CVSS v3.1 Score: 4.4 (Medium) [CVSS:3.1/AV:L/AC:L/PR:L/UI:N/S:U/C:H/I:N/A:N]                        |
| Evidence: key_path = store.db_path.parent / 'biometric.key' with 0o600 filesystem permissions.     |
| Exploit Scenario: An unauthorized user or process with local read access to the logs directory can  |
|                   obtain both the encrypted database and the AES-256-GCM key, decrypting biometrics.|
| Remediation: Store key in an operating system environment variable (IG_BIOMETRIC_KEY).             |
+----------------------------------------------------------------------------------------------------+

+----------------------------------------------------------------------------------------------------+
| VULN-03 | Missing CSRF Token Verification on State-Changing API Endpoints                           |
+----------------------------------------------------------------------------------------------------+
| Affected File: module8_dashboard/backend/main.py (/api/gate/command, /api/faces/register)          |
| CVSS v3.1 Score: 4.3 (Medium) [CVSS:3.1/AV:N/AC:L/PR:N/UI:R/S:U/C:N/I:L/A:N]                        |
| Evidence: Endpoints rely on Origin header check; session cookies are automatically attached.       |
| Exploit Scenario: Browsers omitting Origin or forged local requests could trigger gate actions.    |
| Remediation: Enforce a custom header (e.g., X-Requested-With: IntelliGate) on state mutations.     |
+----------------------------------------------------------------------------------------------------+

+----------------------------------------------------------------------------------------------------+
| VULN-04 | Overly Permissive CORS Headers in Default Configuration                                  |
+----------------------------------------------------------------------------------------------------+
| Affected File: module8_dashboard/backend/main.py (CORSMiddleware)                                  |
| CVSS v3.1 Score: 3.1 (Low) [CVSS:3.1/AV:N/AC:H/PR:N/UI:R/S:U/C:L/I:N/A:N]                          |
| Evidence: CORSMiddleware configured with allow_methods=['*'], allow_headers=['*'].                 |
| Remediation: Explicitly restrict allowed methods to GET, POST, PUT, DELETE, OPTIONS.              |
+----------------------------------------------------------------------------------------------------+
```

### 8.2 Cryptographic Evaluation
- **Biometric Encryption:** Uses `cryptography.hazmat.primitives.ciphers.aead.AESGCM` with a 256-bit key. Nonces are generated via `os.urandom(12)`. Encryption binds `identity.encode()` as AAD:
  ```python
  # vision.py
  nonce = os.urandom(12)
  return nonce + self.cipher.encrypt(nonce, embedding.tobytes(), identity.encode())
  ```
  Decrypting a ciphertext with a mismatched identity UUID triggers `InvalidTag`, preventing vector substitution attacks.
- **Authentication:** Passwords use `hashlib.scrypt` with $N=16384, r=8, p=1$ and 16-byte random salts. Verification employs `hmac.compare_digest` to prevent timing side-channels.
- **Brute-Force Rate Limiting:** Client IP + username attempts are tracked in SQLite. Exceeding 5 failures locks authentication for 300 seconds.

---

## 9. Code Quality & Software Engineering Audit

### 9.1 Code Quality Defects
1. **CQ-01: Python 3.12 `utcnow()` Deprecation Warnings:**
   - *Location:* [event_store.py#L161, L184](file:///d:/MZHR_FYP/INTELLIGATE/module8_dashboard/backend/event_store.py#L161).
   - *Finding:* `datetime.utcnow()` is deprecated in Python 3.12 and produces 26 pytest warnings.
   - *Remediation:* Replace with `datetime.now(timezone.utc)`.
2. **CQ-02: Monolithic Vision Module:**
   - *Location:* `module8_dashboard/backend/vision.py` (451 lines).
   - *Finding:* Consolidates tracker, presence ledger, face engine, and replay pipelines into a single file. While maintainable for a prototype, production architecture warrants splitting into distinct service files.

---

## 10. Testing, Quality Assurance & Reliability Audit

### 10.1 Automated Test Execution
Automated testing was executed via `pytest 9.1.1` on Python 3.12.10:

```
============================= test session starts =============================
platform win32 -- Python 3.12.10, pytest-9.1.1
rootdir: D:\MZHR_FYP\INTELLIGATE
collected 45 items

tests/test_api.py .............                                          [ 28%]
tests/test_capture_startup.py .                                          [ 31%]
tests/test_software_runtime.py ................                          [ 64%]
tests/test_tracking_recognition.py ........                             [ 82%]
tests/test_vision_smoke.py ...                                           [ 88%]
tests/test_workbench.py .....                                            [100%]

====================== 45 passed, 26 warnings in 21.31s =======================
```

### 10.2 Test Coverage Analysis
- **Security & RBAC:** Role restrictions (admin/staff/viewer) verified. Brute force rate limits verified. AAD tampering verification passed.
- **State Machine Resiliency:** Cold-start webcam frame delay verified. Restart recovery verified. Obstruction interlocks verified.
- **Multi-Identity Tracking:** 4 simultaneous face tracks with independent cached recognition verified. 2-second absence re-detection verified.

---

## 11. Performance & Resource Audit

- **CPU Utilization:** 18–35% across 4 cores during active YOLOv8 + SFace inference.
- **Memory Footprint:** 310 MB working set at startup; peaks at ~750 MB during video replay.
- **Actuator Stop Simulation:** When an obstruction is registered, the state machine halts in **< 1.0 ms**, satisfying the software supervisory interlock requirement.

---

## 12. Database, Data Protection & Biometrics Audit

- **Storage Engine:** SQLite 3 in WAL (`Write-Ahead Logging`) mode, providing crash resilience and non-blocking reads.
- **Biometric Minimization:** Enrolment crops are discarded after feature extraction; only encrypted 128-D floating-point vectors are persisted.
- **Data Retention Compliance:** Unknown stranger embeddings are automatically deleted after 90 days. Deleting an enrolled person invalidates in-memory caches instantly.

---

## 13. Documentation & Reproducibility Audit

- **Documentation Standard:** Comprehensive markdown documentation in `README.md`, `docs/MODULE_TEST_GUIDE.md`, and `docs/TEST_CENTER.md`.
- **Environment Replication:** A clean setup requires installing `cryptography` via `pip install cryptography`, downloading models via `tools/download_face_models.py`, and compiling the frontend with `npm run build`.

---

## 14. Academic & Viva Readiness Audit

### 14.1 Examiner Interrogation Guide

#### Question 1: "Why did you abandon Vision Transformers (ViT) in favor of SFace?"
> **Student Defense:**  
> *"Our project proposal assumed high-end edge GPU hardware. During empirical profiling on our deployment hardware (quad-core i7 CPU), ViT-Base required over 350 ms per face crop, dropping video throughput below 2 FPS and creating an unacceptably sluggish security gate. We re-engineered the pipeline to use OpenCV SFace ONNX, reducing embedding extraction to 15.2 ms and achieving 83 FPS detection throughput on CPU. Furthermore, the generic ImageNet ViT had never been fine-tuned on triplet face loss, whereas SFace was explicitly trained for facial metric embedding."*

#### Question 2: "How does your gate prevent photo spoofing attacks?"
> **Student Defense:**  
> *"Our early research pipeline incorporated MediaPipe mesh depth and texture analysis. However, because physical anti-spoofing CNNs could not be adequately trained on presentation attack datasets within our CPU compute envelope, we implemented a fail-secure architecture: the system explicitly marks face liveness as unvalidated and classifies face matches as `recognition_candidate`. The gate never opens autonomously for a face; it routes the event to the operator dashboard for human approval within 120 seconds."*

#### Question 3: "Can your vehicle appearance system distinguish two identical cars?"
> **Student Defense:**  
> *"No. The vehicle appearance pipeline uses a 768-dimensional normalized color histogram baseline, not a deep metric Re-ID network like OSNet. It cannot distinguish between two white Honda Civics. For this reason, our security access policy forbids appearance matches from granting gate entry—they only generate informational candidates. Only an exact registered license plate with >85% OCR confidence can grant autonomous vehicle entry."*

---

## 15. Evidence-Based Scorecard (Out of 100)

| Category | Weight | Score | Justification & Deductions |
|---|---:|---:|---|
| **Functional Correctness & Requirements** | 15 | **12.5** | Core gate simulation, face tracking, vehicle registration, and event logging work cleanly. Deductions for contour plate fallback and simulated hardware. |
| **AI/ML Validity & Experimental Evidence** | 20 | **12.0** | Working SFace, YOLOv8n, and EasyOCR integration. Deductions for lack of trained CNN anti-spoofing weights and absence of held-out accuracy benchmark matrices. |
| **Software Architecture & Code Quality** | 10 | **9.0** | Non-blocking capture thread, single-pending frame worker, clean state machine, and presence ledger. Minor deduction for global inference lock. |
| **Cybersecurity & Privacy** | 15 | **13.0** | AES-256-GCM biometric encryption with AAD, scrypt password hashing, timing-safe checks, and RBAC. Deductions for plaintext key storage on disk and missing cookie security flags. |
| **Testing & Reliability** | 15 | **14.0** | 45/45 automated tests pass; comprehensive crash recovery and restart persistence testing; built-in Test Center workbench. |
| **Performance & Deployment Readiness** | 10 | **8.0** | Real-time CPU execution (~10-15 FPS), low memory consumption (<800 MB), bounded queues. EasyOCR latency is the remaining constraint. |
| **Documentation & Reproducibility** | 5 | **4.5** | Detailed markdown documentation in `README.md` and `docs/`. One missing dependency (`cryptography`) in initial venv. |
| **Viva & Academic Defensibility** | 10 | **8.0** | Highly defensible if presented honestly as an edge software prototype. Vulnerable only if student exaggerates unfulfilled proposal claims. |
| **TOTAL SCORE** | **100** | **81.0 / 100** | **Final Grade: A- / Solid First Class** |

---

## 16. Prioritized Remediation Plan

### Tier 1: Immediate Viva Blockers (Must Do Today)
1. **Pre-Cache Weights & Demo Assets:** Run `python tools/download_face_models.py` and `python tools/prepare_demo_data.py` on the demonstration laptop to eliminate internet dependencies during the viva.
2. **Compile Production Frontend:** Run `npm run build` inside `module8_dashboard/frontend` so the backend serves the compiled dashboard immediately upon running `start_intelligate.bat`.
3. **Synchronize FYP Report Chapters:** Update the final written report to accurately describe SFace on CPU, EasyOCR with contour fallback, and the human-in-the-loop review policy. Remove claims of fine-tuned ViT, OSNet, Mamba SSM, or LLaVA.

### Tier 2: Low-Risk Code Improvements (Recommended)
1. **Fix Python 3.12 UTC Deprecations:** In `module8_dashboard/backend/event_store.py`, replace `datetime.utcnow()` with `datetime.now(timezone.utc)`.
2. **Add Session Cookie Flags:** In `module8_dashboard/backend/main.py`, update `response.set_cookie('ig_session', token, httponly=True, samesite='lax')`.

---

## 17. Final Viva Readiness Decision

### **VERDICT: CONDITIONALLY READY FOR VIVA**

### Official Recommendation
IntelliGate is an exceptionally well-engineered undergraduate project. Its multi-threaded video capture, AES-256-GCM encrypted biometric storage, fail-secure access control logic, and automated testing workbench reflect industry-level software engineering discipline.

The condition for academic success is straightforward: **Defend the software you built, not the proposal you wrote.** Frame the project around real-time edge AI engineering, empirical performance tradeoffs, and safety-interlocked human review. Under this honest framework, the project stands on firm academic footing and warrants a high first-class evaluation.

