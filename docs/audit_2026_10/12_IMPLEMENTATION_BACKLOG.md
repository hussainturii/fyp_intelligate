# INTELLIGATE: GRANULAR IMPLEMENTATION BACKLOG & TASK CATALOG

**Document ID:** AUD-2026-10-12  
**Target Repository:** `D:\MZHR_FYP\INTELLIGATE`  
**Total Backlog Volume:** 22 Granular Engineering Tasks (~155 Total Engineering Hours)  
**Audit Date:** October 18, 2026  

---

## 1. Backlog Overview & Estimation Sizing

Tasks are sized in engineering hours assuming execution by competent AI/software engineering undergraduates with pairs-programming assistance.

| Priority Tier | Task Count | Estimated Hours | Milestone Target |
|---|---|---|---|
| **Tier P0 (Critical Security & Platform)** | 6 Tasks | 30 Hours | End of Week 1 |
| **Tier P1 (Core AI & Scientific Validation)** | 6 Tasks | 45 Hours | End of Week 2 |
| **Tier P2 (Advanced AI & Multimodal)** | 5 Tasks | 42 Hours | End of Week 3 |
| **Tier P3 (Documentation & Viva Readiness)** | 5 Tasks | 38 Hours | End of Week 4 |
| **TOTALS** | **22 Tasks** | **155 Hours** | Complete FYP Excellence |

---

## 2. Granular Task Cards

### TASK-P0-01: Git History Biometric Scrub & Repo Hygiene
- **Priority:** P0 (Blocker)
- **Component:** Git Repository Root
- **Story:** As a project maintainer, I need to permanently purge all unencrypted biometric HDF5 files and `.pyc` caches from Git history so that biometric privacy regulations are fulfilled.
- **Acceptance Criteria:**
  1. `git log --all --full-history -- "**.h5"` returns zero commits.
  2. `git log --all --full-history -- "**.pyc"` returns zero commits.
  3. `.gitignore` contains rules for `*.h5`, `*.key`, `__pycache__/`, `*.task`.
  4. Repository size is reduced by at least 15 MB.
- **Test Command:** `python tools/check_repo_hygiene.py`
- **Hours:** 3h | **Dependencies:** None

---

### TASK-P0-02: Session Cookie Directives & CSRF Defense
- **Priority:** P0 (High)
- **Component:** `module8_dashboard/backend/main.py`
- **Story:** As a security administrator, I need session cookies to include `SameSite=Lax` and configurable `Secure` directives so that sessions cannot be intercepted or exploited via cross-site requests.
- **Acceptance Criteria:**
  1. `response.set_cookie()` sets `samesite="lax"` and `httponly=True`.
  2. Automated test verifies Set-Cookie header contains `SameSite=Lax`.
- **Test Command:** `pytest tests/test_api.py -k test_setup_authentication_and_origin`
- **Hours:** 2h | **Dependencies:** None

---

### TASK-P0-03: Local Verified YOLOv8-Nano License Plate Weights
- **Priority:** P0 (High)
- **Component:** `module2_vehicle/plate_detector.py`, `models/`
- **Story:** As a perception worker, I need local pre-verified YOLOv8-nano license plate detection weights so that plate localization does not depend on dead URLs or fragile contour heuristics.
- **Acceptance Criteria:**
  1. `license_plate_detector.pt` / `.onnx` placed in `models/` with SHA-256 registered in `download_manifest.json`.
  2. `PlateDetector` loads local model without network requests.
  3. Bounding box recall on test vehicles $\ge 90\%$.
- **Test Command:** `pytest tests/test_tracking_recognition.py -k test_vehicle_ocr_cached`
- **Hours:** 5h | **Dependencies:** None

---

### TASK-P0-04: PyTorch CUDA & ONNX TensorRT Hardware Enablement
- **Priority:** P0 (High)
- **Component:** `module8_dashboard/backend/vision.py`
- **Story:** As an AI engineer, I need inference pipelines to execute on the NVIDIA RTX 4060 Ti GPU so that full-pipeline throughput exceeds the 15 FPS SRS target.
- **Acceptance Criteria:**
  1. `vision.py` detects CUDA availability and routes YOLOv8 and ONNX models to `device='cuda:0'`.
  2. Full scene inference latency drops from 124 ms to $< 15\text{ ms}$ (>65 FPS).
  3. Graceful fallback to CPU if CUDA is unavailable.
- **Test Command:** `python tools/benchmark_cpu.py --video logs/demo_data/llvip_visible.mp4`
- **Hours:** 6h | **Dependencies:** Verified CUDA environment

---

### TASK-P0-05: SQLite Write-Ahead Logging (WAL) & Connection Pooling
- **Priority:** P0 (Medium)
- **Component:** `module8_dashboard/backend/event_store.py`
- **Story:** As a backend developer, I need SQLite to operate in WAL mode with connection pooling so that concurrent reads and writes do not trigger database lock errors.
- **Acceptance Criteria:**
  1. Database connection initializes `PRAGMA journal_mode=WAL; PRAGMA synchronous=NORMAL;`.
  2. 50 concurrent simulated writes complete without `sqlite3.OperationalError: database is locked`.
- **Test Command:** `pytest tests/test_software_runtime.py`
- **Hours:** 4h | **Dependencies:** None

---

### TASK-P0-06: Central Model Registry with Checksum Verification
- **Priority:** P0 (Medium)
- **Component:** `module8_dashboard/backend/model_registry.py` (new)
- **Story:** As a software architect, I need a centralized ModelRegistry class that verifies SHA-256 digests before loading weights so that model tampering or corruption is prevented.
- **Acceptance Criteria:**
  1. Registry loads models by name; verifies file hash against `download_manifest.json`.
  2. Corrupted or modified weights raise a fatal `ModelTamperingError`.
- **Test Command:** `pytest tests/test_workbench.py`
- **Hours:** 10h | **Dependencies:** P0-3, P0-4

---

### TASK-P1-01: MiniFASNetV2 Deep Face Anti-Spoofing ONNX Integration
- **Priority:** P1 (High)
- **Component:** `module8_dashboard/backend/vision.py`, `models/`
- **Story:** As an access controller, I need real deep face anti-spoofing to process face crops so that printed photos and video replays are autonomously rejected.
- **Acceptance Criteria:**
  1. MiniFASNetV2 ONNX model integrated into `FaceEngine.detect()`.
  2. Returns liveness probability $P(\text{real})$ in $< 5\text{ ms}$ on GPU.
  3. Sets `validated_liveness = True` when score exceeds calibrated threshold $\tau$.
- **Test Command:** `pytest tests/test_vision_smoke.py`
- **Hours:** 12h | **Dependencies:** P0-4, P0-6

---

### TASK-P1-02: Anti-Spoofing Threshold Calibration on CelebA-Spoof
- **Priority:** P1 (High)
- **Component:** `tools/calibrate_liveness.py` (new)
- **Story:** As an AI researcher, I need to empirically calibrate the liveness decision threshold on a validation split to achieve APCER $\le 3.5\%$ at BPCER $\le 2.0\%$.
- **Acceptance Criteria:**
  1. Evaluates 1,000 real and 1,000 attack images from CelebA-Spoof.
  2. Generates APCER vs. BPCER trade-off curve; selects optimal operating threshold $\tau$.
  3. Outputs calibration JSON report to `logs/benchmarks/liveness_calibration.json`.
- **Test Command:** `python tools/calibrate_liveness.py`
- **Hours:** 8h | **Dependencies:** TASK-P1-01

---

### TASK-P1-03: OSNet-IBN Deep Vehicle Re-Identification Integration
- **Priority:** P1 (High)
- **Component:** `module8_dashboard/backend/vision.py`, `models/`
- **Story:** As a security officer, I need deep metric appearance Re-ID to replace color histograms so that obscured vehicles are accurately matched.
- **Acceptance Criteria:**
  1. OSNet-IBN-x1_0 ONNX model integrated into `ReplayPipeline._vehicle()`.
  2. Extracts 512-D L2-normalized embedding in $< 6\text{ ms}$ on GPU.
  3. Replaces `appearance()` histogram; searches enrolled vehicle embeddings via cosine similarity.
- **Test Command:** `pytest tests/test_tracking_recognition.py -k test_vehicle`
- **Hours:** 12h | **Dependencies:** P0-4, P0-6

---

### TASK-P1-04: VeRi-776 Benchmark Evaluation & CMC Curve Generation
- **Priority:** P1 (High)
- **Component:** `tools/evaluate_veri776.py` (new)
- **Story:** As an AI researcher, I need to evaluate the vehicle Re-ID model on the official VeRi-776 test split to report Rank-1, Rank-5, and mAP metrics.
- **Acceptance Criteria:**
  1. Ingests VeRi-776 query and gallery sets.
  2. Computes Euclidean / Cosine distance matrix; outputs CMC curve and mAP.
  3. Achieves $\text{Rank-1} \ge 85.0\%$ and $\text{mAP} \ge 65.0\%$.
- **Test Command:** `python tools/evaluate_veri776.py`
- **Hours:** 8h | **Dependencies:** TASK-P1-03

---

### TASK-P1-05: Pakistani License Plate OCR Test Harness
- **Priority:** P1 (High)
- **Component:** `tools/evaluate_ocr.py` (new)
- **Story:** As a researcher, I need an annotated test set of 100 Pakistani license plates to empirically evaluate EasyOCR Character Error Rate and exact-match accuracy.
- **Acceptance Criteria:**
  1. Evaluates 100 annotated license plate crops across standard provincial formats.
  2. Computes Levenshtein Character Error Rate (CER) and exact plate accuracy.
  3. Outputs error breakdown report and confusion matrix to `logs/benchmarks/ocr_evaluation.json`.
- **Test Command:** `python tools/evaluate_ocr.py`
- **Hours:** 10h | **Dependencies:** P0-3

---

### TASK-P1-06: Face Verification ROC-AUC & EER Empirical Benchmark
- **Priority:** P1 (High)
- **Component:** `tools/evaluate_face_verification.py` (new)
- **Story:** As an AI researcher, I need to measure SFace on 1,000 genuine and impostor face pairs to compute the ROC curve, EER, and TAR@FAR.
- **Acceptance Criteria:**
  1. Evaluates pairs from LFW / CelebA-HQ test split.
  2. Plots ROC curve; calculates EER and TAR at $\text{FAR}=0.1\%$.
  3. Saves publication-ready plot to `docs/figures/face_roc_curve.png`.
- **Test Command:** `python tools/evaluate_face_verification.py`
- **Hours:** 5h | **Dependencies:** P0-4

---

### TASK-P2-01: Zero-DCE Learned Low-Light Enhancement Integration
- **Priority:** P2 (Medium)
- **Component:** `module8_dashboard/backend/vision.py`, `models/`
- **Story:** As a vision engineer, I need to integrate Zero-DCE deep curve estimation so that low-light video contrast is enhanced without amplifying sensor noise.
- **Acceptance Criteria:**
  1. Zero-DCE ONNX model integrated into `enhance()` method in `vision.py`.
  2. Forward pass executes in $< 6\text{ ms}$ on RTX 4060 Ti GPU.
  3. Demonstrates higher downstream YOLO pedestrian detection recall on LLVIP test frames than CLAHE.
- **Test Command:** `python tools/benchmark_llvip.py`
- **Hours:** 10h | **Dependencies:** P0-4

---

### TASK-P2-02: ByteTrack Multi-Object Association Tracker Integration
- **Priority:** P2 (Medium)
- **Component:** `module8_dashboard/backend/vision.py: Tracker`
- **Story:** As an operator, I need ByteTrack association to replace basic IoU tracking so that pedestrians and vehicles maintain stable track IDs through occlusions.
- **Acceptance Criteria:**
  1. ByteTrack algorithm integrated; associates low-score detection boxes with previous tracks.
  2. Track ID switches reduced by at least $50\%$ during synthetic occlusion test.
- **Test Command:** `pytest tests/test_tracking_recognition.py`
- **Hours:** 8h | **Dependencies:** None

---

### TASK-P2-03: Kinematic Trajectory Feature Extraction & Isolation Forest Anomaly
- **Priority:** P2 (Medium)
- **Component:** `module8_dashboard/backend/vision.py`, `software_runtime.py`
- **Story:** As a security analyst, I need an unsupervised Isolation Forest model fit on track kinematic features so that anomalous loitering is detected with machine learning.
- **Acceptance Criteria:**
  1. Computes 5-D kinematic vector $[v_x, v_y, \text{var}(v), \text{transit\_ratio}, \text{dwell}]$.
  2. Isolation Forest flags anomalous vectors; emits `loitering_anomaly` alert.
  3. Achieves $\text{F1} \ge 0.85$ on simulated anomaly dataset.
- **Test Command:** `pytest tests/test_software_runtime.py`
- **Hours:** 10h | **Dependencies:** TASK-P2-02

---

### TASK-P2-04: Local Multimodal VLM (Moondream2 / SmolVLM) Integration
- **Priority:** P2 (Medium)
- **Component:** `module8_dashboard/backend/vlm_engine.py` (new)
- **Story:** As a security supervisor, I need a local lightweight VLM running on GPU to generate concise visual descriptions of security alerts for evidence logs.
- **Acceptance Criteria:**
  1. Moondream2 (1.86B) or SmolVLM-500M loaded on RTX 4060 Ti GPU (~2.5 GB VRAM).
  2. Generates factual 2-sentence visual description of incident crop in $< 800\text{ ms}$.
  3. Read-only execution: VLM has zero connection to gate actuation endpoints.
- **Test Command:** `python tools/test_vlm_summary.py`
- **Hours:** 14h | **Dependencies:** P0-4 (RTX 4060 Ti GPU)

---

### TASK-P3-01: Formal SRS v4.0 Scope Amendment Documentation
- **Priority:** P3 (Documentation)
- **Component:** `docs/IntelliGate_Scope_Revision_v4.0.md`
- **Story:** As a degree candidate, I need a formal Scope Revision Note signed by my supervisors to officially supersede outdated physical hardware and foundation model claims.
- **Acceptance Criteria:**
  1. Scope revision document approved and signed.
  2. Thesis chapters aligned with authoritative replacement wording.
- **Hours:** 8h | **Dependencies:** None

---

### TASK-P3-02: 72-Hour Reliability & Resource Leak Soak Test
- **Priority:** P3 (Stability)
- **Component:** `tools/soak_test.py`
- **Story:** As an engineer, I need to execute a continuous 72-hour automated soak test to prove system stability and verify 99.5% operational uptime.
- **Acceptance Criteria:**
  1. Test executes for 72 hours continuously on video loop.
  2. Process RSS memory growth $< 50\text{ MB}$; zero unhandled exceptions.
  3. Generates verified uptime report in `logs/benchmarks/soak_test_report.json`.
- **Hours:** 6h setup | **Dependencies:** All active modules

---

### TASK-P3-03: Frontend Bundle Optimization & Code-Splitting
- **Priority:** P3 (Polish)
- **Component:** `module8_dashboard/frontend/`
- **Story:** As a frontend user, I need large components lazy-loaded so that the initial JS bundle is $< 300\text{ kB}$.
- **Acceptance Criteria:**
  1. `React.lazy()` applied to `TestCenter.jsx` and `Operations.jsx`.
  2. Production build bundle size reduces from 624 kB to $< 300\text{ kB}$ without warnings.
- **Test Command:** `npm run build` in `module8_dashboard/frontend`
- **Hours:** 6h | **Dependencies:** None

---

### TASK-P3-04: Offline Fallback Video Demonstrations
- **Priority:** P3 (Viva Readiness)
- **Component:** `docs/viva/videos/`
- **Story:** As a presenter, I need 1080p narrated offline backup screen recordings of all 8 core workflows to protect against hardware or camera failures during defense.
- **Acceptance Criteria:**
  1. 8 video clips recorded and verified playable on presentation laptop.
- **Hours:** 8h | **Dependencies:** P1-1, P1-3, P2-4

---

### TASK-P3-05: Final Thesis Defense Slide Deck & Viva Cheat Sheet
- **Priority:** P3 (Viva Readiness)
- **Component:** `docs/viva/`
- **Story:** As a candidate, I need a 25-slide defense deck and technical cheat sheet covering the 5 Toughest Viva Questions with empirical graphs.
- **Acceptance Criteria:**
  1. Presentation slides compiled with ROC curves, CMC plots, and FSM diagrams.
  2. 2-page candidate quick reference sheet printed.
- **Hours:** 10h | **Dependencies:** All evaluation tasks
