# INTELLIGATE: PRIORITIZED 4-WEEK ENGINEERING & AI EXCELLENCE ROADMAP

**Document ID:** AUD-2026-10-10  
**Target Repository:** `D:\MZHR_FYP\INTELLIGATE`  
**Available Hardware:** AMD Ryzen 7 5800X (8C/16T), 32 GB RAM, NVIDIA GeForce RTX 4060 Ti 16GB VRAM, CUDA 13.3 / PyTorch 2.7.1  
**Execution Window:** 2 to 4 Weeks (~120–160 Total Engineering Hours)  
**Audit Date:** October 18, 2026  

---

## 1. Roadmap Architecture & Priority Tiers

The roadmap is strictly partitioned into four dependency-aware priority tiers:
- **Tier P0 — Critical Security, Correctness & Stability (Week 1 / 30 Hours):** Fixes ethical/legal biometric exposure, hardens browser session transport, repairs broken plate localization, and enables CUDA acceleration on the RTX 4060 Ti.
- **Tier P1 — Essential AI Upgrades & Scientific Validation (Week 2 / 45 Hours):** Integrates deep anti-spoofing and deep vehicle Re-ID; executes empirical held-out benchmark evaluations to produce publishable academic metrics.
- **Tier P2 — Advanced Intelligence & Multimodal Analytics (Week 3 / 40 Hours):** Implements learned low-light enhancement, kinematic temporal anomaly detection, and local lightweight VLM incident summarization.
- **Tier P3 — Viva Polish, Documentation & Stress Testing (Week 4 / 25 Hours):** Finalizes SRS v4.0 Scope Amendment, conducts 72-hour reliability soak testing, records offline backup videos, and prepares defense slide deck.

---

## 2. Priority P0: Critical Security, Correctness & Platform Enablement (Week 1)

### P0-1: Purge Committed Biometric Exposure from Git History
- **Current Problem:** An unencrypted legacy HDF5 biometric embedding database (`module1_face_recognition/database/face_embeddings.h5`) was committed in initial commit `98df922`.
- **Why It Matters:** Committing raw biometric templates to Git history violates GDPR Article 9 and international biometric data protection regulations. If the repository is published or submitted, it exposes subjects to permanent biometric compromise.
- **Affected Files:** Git commit history, `.gitignore`, `module1_face_recognition/database/face_embeddings.h5`.
- **Proposed Architecture:** Use `git-filter-repo` to permanently erase all `.h5`, `.key`, and `.pyc` files from every commit across all branches, then force-push clean history.
- **Dependencies:** None.
- **Risk:** Low (requires offline backup before execution).
- **Estimated Hours:** 3 hours.
- **Academic Benefit:** Removes critical ethical/legal red flag during thesis evaluation.
- **Definition of Done:** `git log --all --full-history -- "**.h5"` returns zero matches; `.gitignore` contains explicit exclusion rules; clean clone builds without error.

---

### P0-2: Harden Session Security & Cookie Transport Directives
- **Current Problem:** `main.py:149` sets `ig_session` cookie without `samesite="lax"` or `secure=True`, leaving sessions susceptible to CSRF and unencrypted transport leakage.
- **Why It Matters:** Fails standard application security requirements (OWASP Top 10 A01:2021-Broken Access Control).
- **Affected Files:** `module8_dashboard/backend/main.py`, `module8_dashboard/backend/auth.py`.
- **Proposed Architecture:** Configure `samesite="lax"` and cryptographic HMAC session signature validation.
- **Dependencies:** None.
- **Risk:** Low.
- **Estimated Hours:** 2 hours.
- **Academic Benefit:** Demonstrates defense-in-depth engineering rigor during software architecture evaluation.
- **Definition of Done:** Automated test verifies `SameSite=Lax` attribute on Set-Cookie header; cross-origin forged POST requests are rejected.

---

### P0-3: Restore Neural License Plate Detection (YOLOv8-Nano Plate Model)
- **Current Problem:** Upstream HuggingFace model URL returned HTTP 404, forcing plate detection to always fall back to classical OpenCV contour detection (`vision.py:224`). Contour detection fails on oblique, low-contrast, or dirty plates.
- **Why It Matters:** High plate localization failure rate undermines downstream OCR and breaks automated vehicle entry.
- **Affected Files:** `module2_vehicle/plate_detector.py`, `module8_dashboard/backend/vision.py`, `models/`.
- **Proposed Architecture:** Acquire verified YOLOv8-nano license plate detection weights (`license_plate_detector.pt` / `.onnx`), place in `models/`, register SHA-256 hash in `models/download_manifest.json`, and instantiate `PlateDetector` with verified local weights.
- **Dependencies:** Ultralytics YOLOv8 runtime.
- **Risk:** Low.
- **Estimated Hours:** 5 hours.
- **Academic Benefit:** Eliminates brittle contour heuristic; provides solid foundation for vehicle access evaluation.
- **Definition of Done:** Plate localization achieves $\ge 90\%$ recall on test video clips; runs with local weights without network requests.

---

### P0-4: Enable PyTorch CUDA & ONNX TensorRT/CUDA Inference Engine
- **Current Problem:** System currently executes entirely on CPU, running YOLOv8n at only 7.92 FPS (failing the SRS 15 FPS target). The physical workstation contains a powerful **NVIDIA GeForce RTX 4060 Ti 16GB GPU** that is currently 0% utilized.
- **Why It Matters:** Unlocks high-throughput real-time multi-model execution (>60 FPS) and enables local deep model training/fine-tuning.
- **Affected Files:** `module8_dashboard/backend/vision.py`, `config/config.yaml`, `start_intelligate.bat`.
- **Proposed Architecture:** Update `vision.py` and `FaceEngine` to automatically detect CUDA availability (`torch.cuda.is_available()`); configure ONNX Runtime with `CUDAExecutionProvider` or OpenCV DNN with `DNN_BACKEND_CUDA`; pass `device='cuda:0'` to YOLOv8.
- **Dependencies:** Verified `PyTorch 2.7.1+cu118` in `D:\MZHR_FYP\fyp_intelligate-1\.venv`.
- **Risk:** Low (fallback to CPU is preserved if CUDA fails).
- **Estimated Hours:** 6 hours.
- **Academic Benefit:** Meets and exceeds SRS throughput target (NFR-P4); showcases modern hardware acceleration.
- **Definition of Done:** `vision.py` logs `[CUDA] Initialized NVIDIA GeForce RTX 4060 Ti`; YOLOv8n inference latency drops to $< 10\text{ ms}$ (>100 FPS).

---

## 3. Priority P1: Essential AI Upgrades & Scientific Validation (Week 2)

### P1-1: Integrate & Calibrate Deep Face Anti-Spoofing (MiniFASNetV2 ONNX)
- **Current Problem:** No active anti-spoofing model runs in the active runtime. All face matches force `REVIEW_REQUIRED`, preventing autonomous entry. Legacy heuristics fail against printed photographs.
- **Why It Matters:** Anti-spoofing is the primary security defense against biometric presentation attacks. Essential for an AI security project.
- **Affected Files:** `module8_dashboard/backend/vision.py`, `module1_face_recognition/anti_spoofing/`, `models/`.
- **Proposed Architecture:** Integrate **MiniFASNetV2** (depthwise separable CNN with Fourier frequency analysis) exported to ONNX (~1.5 MB). Process $80\times80$ face crop; compute liveness probability $P(\text{real})$. If $P(\text{real}) \ge \tau_{\text{calibrated}}$, set `validated_liveness = True`.
- **Dataset:** CelebA-Spoof / CASIA-SURF (1,000 genuine, 1,000 attack samples for calibration).
- **Evaluation Metrics:** APCER, BPCER, ACER, inference latency.
- **Dependencies:** P0-4 (CUDA engine).
- **Risk:** Medium (requires empirical threshold calibration).
- **Estimated Hours:** 16 hours.
- **Academic Benefit:** Directly addresses Viva Question 2; elevates Module 1 from heuristic to state-of-the-art deep learning.
- **Definition of Done:** MiniFASNet achieves $\text{ACER} \le 3.5\%$ on validation split; printed photos are autonomously denied and logged as `spoof_detected`.

---

### P1-2: Replace Color Histogram with Deep Vehicle Re-ID (OSNet-IBN ONNX)
- **Current Problem:** Vehicle appearance matching uses a 768-D concatenated RGB color histogram (`vision.py:24–27`), which cannot distinguish different vehicles of the same color/model.
- **Why It Matters:** Fulfills the core algorithmic promise of Module 2 (deep appearance Re-ID when plates are obscured).
- **Affected Files:** `module8_dashboard/backend/vision.py`, `module2_vehicle/vehicle_reid.py`, `models/`.
- **Proposed Architecture:** Export an **OSNet-IBN-x1_0** metric model (pretrained on VeRi-776) to ONNX (512-D embedding). Ingest 256×256 vehicle crop; extract L2-normalized embedding; compute cosine distance against registered vehicles.
- **Dataset:** VeRi-776 benchmark subset (200 query images, 1,000 gallery images).
- **Evaluation Metrics:** Rank-1, Rank-5, mean Average Precision (mAP).
- **Dependencies:** P0-4.
- **Risk:** Medium.
- **Estimated Hours:** 14 hours.
- **Academic Benefit:** Directly satisfies SRS FR-05 with publishable metric evaluation.
- **Definition of Done:** OSNet-IBN ONNX executes in $< 6\text{ ms}$ on GPU; achieves $\text{Rank-1} \ge 85.0\%$ on VeRi-776 evaluation subset.

---

### P1-3: Empirical Held-Out Accuracy Benchmarking & Evaluation Suite
- **Current Problem:** Zero quantitative benchmark curves or accuracy metrics exist in the repository.
- **Why It Matters:** Academic examiners grade BS AI projects on scientific empirical validation. Without quantitative metrics, an FYP cannot receive an 'A' grade.
- **Affected Files:** `tools/evaluate_benchmarks.py` (new), `docs/audit_2026_10/07_PERFORMANCE_BENCHMARKS.md`.
- **Proposed Architecture:** Implement a reproducible evaluation harness:
  1. Face Verification: Compute ROC curve, EER, and TAR@FAR ($0.1\%$, $1.0\%$) on 1,000 pairs from LFW / CelebA.
  2. Anti-Spoofing: Compute APCER vs. BPCER trade-off curve on CelebA-Spoof.
  3. Plate OCR: Evaluate Character Error Rate (CER) and exact-match rate on 100 annotated Pakistani plates.
  4. Vehicle Re-ID: Compute CMC curve and mAP on VeRi-776.
- **Dependencies:** P1-1, P1-2.
- **Risk:** Low (execution of structured scripts).
- **Estimated Hours:** 15 hours.
- **Academic Benefit:** Provides the empirical evaluation tables, graphs, and confusion matrices required for Chapter 5 of the thesis.
- **Definition of Done:** Standalone script generates full JSON benchmark reports and publication-ready matplotlib figures in `logs/benchmarks/`.

---

## 4. Priority P2: Advanced Intelligence & Multimodal Analytics (Week 3)

### P2-1: Learned Low-Light Enhancement Benchmark (Zero-DCE vs. CLAHE)
- **Current Problem:** Low-light processing relies exclusively on OpenCV CLAHE, which degraded YOLO detection recall during LLVIP exploration.
- **Why It Matters:** Solves real sensor noise amplification in night surveillance; provides a compelling scientific comparative study.
- **Affected Files:** `module8_dashboard/backend/vision.py`, `models/`, `tools/benchmark_llvip.py`.
- **Proposed Architecture:** Integrate **Zero-DCE** (Zero-Reference Deep Curve Estimation, 79K params, ONNX format). Process low-light frame in $< 5\text{ ms}$; pass enhanced frame to YOLOv8n.
- **Dataset:** LLVIP low-light visible test split (500 frames).
- **Evaluation Metrics:** Pedestrian detection mAP@0.5 delta, NIQE image naturalness score.
- **Dependencies:** P0-4.
- **Risk:** Low.
- **Estimated Hours:** 12 hours.
- **Academic Benefit:** Delivers a clear scientific research contribution: empirical proof that learned curve estimation outperforms classical equalization for downstream detection.
- **Definition of Done:** Zero-DCE achieves $> 5\%$ higher detection recall on LLVIP low-light frames than CLAHE.

---

### P2-2: Kinematic Feature Extraction & Temporal Anomaly Anomaly Detection
- **Current Problem:** Anomaly detection relies on a simple $>30$s dwell rule. State Space Models (SSM) from proposals are non-existent.
- **Why It Matters:** Upgrades Module 4 to an intelligent machine learning baseline without over-promising unprovable crime prediction.
- **Affected Files:** `module8_dashboard/backend/vision.py`, `module8_dashboard/backend/software_runtime.py`.
- **Proposed Architecture:** Extract 5-D trajectory kinematics per track: $[v_x, v_y, \text{speed\_var}, \text{dwell\_time}, \text{roi\_transit\_ratio}]$. Fit an unsupervised **Isolation Forest** on normal entrance transit vectors. Scores exceeding anomaly threshold fire `loitering_anomaly` events.
- **Dataset:** 100 simulated track vectors (75 normal passages, 25 prowling/loitering runs).
- **Evaluation Metrics:** Precision, Recall, F1-Score on anomaly detection.
- **Dependencies:** None.
- **Risk:** Low.
- **Estimated Hours:** 12 hours.
- **Academic Benefit:** Provides a legitimate, defensible unsupervised ML contribution that satisfies the behavioral analytics requirement.
- **Definition of Done:** Isolation Forest achieves $\text{F1} \ge 0.85$ on anomalous trajectory test set.

---

### P2-3: Local Lightweight VLM Incident Explainer (Moondream2 / SmolVLM on GPU)
- **Current Problem:** Security reports use deterministic string templates (`software_runtime.py:206`). No multimodal language intelligence exists.
- **Why It Matters:** Fulfills the multimodal Vision-Language intelligence requirement (Module 7) using the host's 16GB GPU.
- **Affected Files:** `module8_dashboard/backend/vlm_engine.py` (new), `module8_dashboard/backend/software_runtime.py`.
- **Proposed Architecture:** Load **Moondream2** (1.86B params, ~2.5 GB VRAM in FP16) or **SmolVLM-500M** on RTX 4060 Ti GPU. When a security alert occurs (e.g. unknown repeat visitor or loitering event), feed the incident crop + prompt: *"Describe the security event concisely: person clothing, stance, and activity"*.
- **Safety Boundary:** The VLM is strictly read-only and expository; it has zero access to gate command endpoints.
- **Dependencies:** P0-4 (RTX 4060 Ti GPU).
- **Risk:** Medium (requires prompt engineering to prevent hallucinations).
- **Estimated Hours:** 16 hours.
- **Academic Benefit:** High "wow factor" during live viva demonstrations; demonstrates mastery of modern local multimodal models.
- **Definition of Done:** VLM generates factual 2-sentence visual incident captions in $< 800\text{ ms}$; outputs displayed in dashboard Event Log evidence view.

---

## 5. Priority P3: Viva Readiness, Documentation & Polish (Week 4)

### P3-1: Formal SRS v4.0 Scope Amendment & Thesis Chapter Alignment
- **Current Problem:** Submitted proposal and SRS drafts contain contradictions (ViT vs. SFace, SSM vs. rules, physical hardware vs. software prototype).
- **Why It Matters:** Eliminates the risk of external examiners evaluating students against unfulfilled hardware promises.
- **Affected Files:** `docs/IntelliGate_Scope_Revision_v4.0.md` (new), Final Project Report.
- **Action:** Compile a formal 3-page *Scope Revision Note* signed by project authors and supervisors; incorporate authoritative replacement wording from `08_DOCUMENTATION_AND_SCOPE_CONTRADICTIONS.md`.
- **Estimated Hours:** 8 hours.
- **Definition of Done:** Document approved and signed; report chapters updated.

---

### P3-2: 72-Hour Reliability & Memory Leak Soak Testing
- **Current Problem:** SRS claims 99.5% uptime (NFR-P6) without any multi-hour endurance test evidence.
- **Why It Matters:** Demonstrates software engineering maturity and production stability.
- **Affected Files:** `tools/soak_test.py` (new), `logs/benchmarks/soak_test_report.json`.
- **Action:** Run continuous 72-hour headless video loop; log process memory (RSS), CPU utilization, VRAM leakage, and event store integrity every 15 minutes.
- **Estimated Hours:** 5 hours setup + 72 hours unattended execution.
- **Definition of Done:** Zero process crashes, memory leakage $< 50\text{ MB}$ over 72 hours; report saved to `logs/benchmarks/`.

---

### P3-3: Offline Fallback Video Demonstrations & Viva Slide Deck
- **Current Problem:** Live hardware demonstrations during viva examinations carry a high risk of unexpected failure (camera driver disconnect, lighting shift).
- **Why It Matters:** Guarantees a flawless demonstration regardless of physical examination room constraints.
- **Affected Files:** Presentation slides, pre-recorded scenario videos in `docs/viva/`.
- **Action:** Record 1080p narrated screen recordings of all 8 core workflows (Face approval, spoof rejection, vehicle entry, repeat stranger escalation, obstruction halt, low-light enhancement, VLM report, Test Center workbench); compile defense slides.
- **Estimated Hours:** 12 hours.
- **Definition of Done:** 8 high-definition video clips prepared; slide deck addresses all 5 Toughest Viva Questions with empirical charts.

---

## 6. Features Strictly Excluded (Negative Scope Register)

To ensure high technical depth and prevent scope dispersion over the 4-week window, the following items are **STRICTLY EXCLUDED**:

| Excluded Feature | Original Source | Explicit Rationale for Non-Implementation |
|---|---|---|
| **Physical Relay & GPIO Wiring** | SRS v1.0 §10.5 | Software-only pivot excludes hardware; custom wiring adds zero AI academic value. |
| **Physical FLIR LWIR Thermal Camera** | SRS v1.0 §10.6 | Hardware cost >$1,500; recorded LLVIP infrared replays fully satisfy research evaluation. |
| **Native iOS / Android Mobile Apps** | Proposal UI-3 | Web dashboard is fully responsive; mobile app adds frontend overhead without AI credit. |
| **Full Foundation ViT-Base Training** | Proposal FE-1 | Pretraining a 86M-parameter ViT requires millions of face images and weeks of cluster compute. |
| **Full Mamba / SSM Training** | SRS v1.0 FR-09 | Lacks required longitudinal pedestrian movement training datasets; Isolation Forest is far more defensible. |
| **Microservice Architecture Rewrite** | Kit Architecture | Splitting FastAPI into 11 Docker containers adds DevOps complexity without improving AI quality. |
