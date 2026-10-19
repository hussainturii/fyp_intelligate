# INTELLIGATE: MASTER FORENSIC ENGINEERING AUDIT, REQUIREMENTS VERIFICATION & AI EXCELLENCE ROADMAP

**Document ID:** AUD-2026-10-00  
**Target Repository:** `D:\MZHR_FYP\INTELLIGATE` (`origin: https://github.com/hussainturii/fyp_intelligate.git`)  
**Audit Date:** October 18, 2026  
**Auditing Panel:** Principal AI/ML Engineer, Senior Software Architect, Application Security Engineer, CV Research Supervisor, Independent Technical Auditor, Technical Project Lead  
**Institution:** COMSATS University Islamabad, Department of Computer Science  
**Degree Program:** Bachelor of Science in Artificial Intelligence (BS AI, Session 2023–2027)  
**Project Authors:** Hussain Raza (CIIT/SP23-BAI-057/ISB), Muhammad Mazhar Afzal (CIIT/SP23-BAI-035/ISB)  
**Project Supervisors:** Dr. Sameera Batool (Supervisor), Sir Noman Mehtab (Co-Supervisor)  
**Target Git Baseline:** Commit `ad76140` (*"Document IntelliGate security review"*) on branch `main` (ahead of `origin/main` at `c909145`)  

---

## 1. Executive Summary & Audit Mandate

This forensic engineering audit and scientific evaluation was conducted on the IntelliGate codebase between October 15 and October 18, 2026. The objective of this assessment is to establish a rigorous, empirical, and reproducible baseline of the project's technical reality, verify compliance against the original Software Requirements Specification (SRS v1.0/v3.0) and subsequent scope amendments, assess defensive cybersecurity risks, and formulate an executable 2–4 week AI excellence roadmap for viva defense and final project evaluation.

### 1.1 Hardware Environment Discovery

Prior audit documentation (dated October 15, 2026) operated under the assumption that IntelliGate was permanently constrained to a low-power quad-core laptop CPU (Intel Core i7-1165G7 with 7.7 GiB RAM and no discrete GPU). A critical finding of this repository audit is that **the current host workstation possesses high-performance, enterprise-grade deep learning hardware**:

- **Central Processing Unit (CPU):** AMD Ryzen 7 5800X 8-Core Processor (16 Logical Processors @ 3.80 GHz base / 4.70 GHz boost)
- **Host Memory (RAM):** 31.93 GB Total Physical RAM (~19.0 GB free)
- **Discrete GPU:** **NVIDIA GeForce RTX 4060 Ti 16GB** (16,380 MiB GDDR6 Dedicated VRAM, Driver 610.88, Ada Lovelace Architecture, Compute Capability 8.9)
- **CUDA Runtime Support:** CUDA 13.3 driver capability; fully operational `PyTorch 2.7.1+cu118` with CUDA device execution verified via `D:\MZHR_FYP\fyp_intelligate-1\.venv`
- **Storage Subsystem:** Drive D: (Project Volume) 502.2 GB Free / 1.37 TB Total; Drive C: (System Volume) 142.6 GB Free / 475.9 GB Total
- **Operating System:** Microsoft Windows 11 Pro / Home 64-bit (Build 10.0.26200)

**Impact on Project Capability:** The presence of an NVIDIA RTX 4060 Ti with 16GB of VRAM completely invalidates the previous premise that deep neural networks, vision transformers, fine-tuned vehicle Re-ID, neural anti-spoofing classifiers, and local multimodal vision-language models (VLMs) cannot be executed locally. The workstation has sufficient compute and memory capacity to train, fine-tune, and run low-latency inference on batch-size 32/64 vision models, quantized 7B/3B parameter VLMs (e.g., Moondream2, SmolVLM, Qwen2-VL), and metric learning embeddings.

---

## 2. Project Status Overview & Scope Verification

### 2.1 The Software-Only Scope Pivot
IntelliGate was originally proposed in late 2025 as an edge appliance combining physical gate integration (Raspberry Pi GPIO, 5V relays, magnetic sensors), dedicated dual-spectrum optical hardware (RGB + Long-Wave Infrared / thermal cameras), and complex foundation AI models (Vision Transformers, State Space Models, LLaVA).

On October 1, 2026, a strategic scope pivot was initiated to transition IntelliGate into a **software-only system** (documented in `totally software based.txt` and `docs/SOFTWARE_ONLY_DELIVERY_PLAN.md`):
- **Exclusion of Physical Actuators:** No physical gate motors, microcontrollers (Arduino/ESP32), or relay wiring are required. Gate mechanics are fully simulated via a software state machine.
- **Exclusion of Dedicated Thermal/IR Hardware:** Specialized FLIR cameras are replaced by curated/recorded multimodal datasets (such as LLVIP) and synthetic contrast evaluations.
- **Data Ingestion:** The live perception pipeline ingests standard USB/integrated webcams for operator demonstration, uploaded MP4 video clips, and streaming replay datasets.
- **Academic Defensibility Requirement:** A software-only pivot is academically defensible for a BS Artificial Intelligence project *only* if the time saved on physical hardware integration is reinvested into deeper algorithmic rigor, empirical model validation, statistical benchmarking, and defensive software architecture.

### 2.2 Current Implementation Baseline: The CPU Prototype
On October 15, 2026, an integrated CPU software prototype was committed (`c909145`). This operational system consists of:
1. **Unified Backend (`module8_dashboard/backend`):** A single-process FastAPI asynchronous server hosting REST endpoints, WebSocket event streaming, an MJPEG camera stream, and an SQLite database.
2. **Decoupled Perception Architecture (`vision.py`, `camera_manager.py`):** An independent capture thread decodes video frames at camera rate (~20–30 FPS) into a single-item drop-oldest buffer. An asynchronous inference worker runs on the latest available frame, decoupling preview responsiveness from heavier neural inference.
3. **Active Perception Models:**
   - Face Detection: OpenCV Zoo YuNet ONNX (640×360 input)
   - Face Recognition: OpenCV Zoo SFace ONNX (128-D L2-normalized embeddings)
   - Object/Scene Detection: Ultralytics YOLOv8n (COCO pretrained, 416×416 input)
   - Optical Character Recognition: EasyOCR (CRAFT detector + ResNet/LSTM recognizer)
   - License Plate Localisation: Classical OpenCV edge/contour filtering (fallback)
   - Vehicle Appearance: 768-D concatenated RGB color histograms (heuristic baseline)
   - Low-Light Enhancement: OpenCV LAB-space CLAHE (clipLimit=2, tileGridSize=8×8)
4. **Virtual Gate & Safety Policy (`software_runtime.py`):** A deterministic finite state machine (FSM) enforcing fail-secure states (`CLOSED`, `OPENING`, `OPEN`, `CLOSING`, `STOPPED`, `LOCKDOWN`), emergency stop pre-emption, obstruction interlocks, and a 120-second sliding review window.
5. **Modern Single-Page Application (`module8_dashboard/frontend`):** Built with React 18, Vite, and Tailwind CSS, providing real-time camera preview, gate controls, event logs, security alerts, and an interactive Test Center workbench.

---

## 3. High-Level Audit Findings

### 3.1 Strongest Engineering Contributions
1. **Perception-Authority Decoupling (Fail-Secure Gate Policy):** The architecture rigorously enforces that unvalidated AI inferences cannot actuate the virtual gate. Facial recognition matches against the enrolled database do not autonomously open the gate; instead, they generate a `recognition_candidate` event and mandate operator human-in-the-loop approval. Autonomous opening is strictly reserved for exact registered license plates exceeding an 85% OCR confidence threshold.
2. **Thread-Safe, Non-Blocking Capture Buffer:** Camera preview latency remains under 50 ms because the capture thread does not block on model inference. If an inference cycle takes 120 ms, intermediate video frames are safely dropped, preventing pipeline lag and frame queuing.
3. **Cryptographically Bound Biometric Storage:** Face embeddings are encrypted at rest with **AES-256-GCM**. Each entry generates a cryptographically random 12-byte nonce, and the record's unique UUID is bound as Additional Authenticated Data (AAD). This prevents ciphertext substitution and transplant attacks across database rows.
4. **Automated Verification Coverage:** The automated test suite achieves a **100% pass rate (45/45 tests passing in 23.00 seconds)**, verifying RBAC enforcement, session security, brute-force rate-limiting, command idempotency, restart state recovery, and multi-identity tracking.
5. **Architectural Candor in Current Prototype:** The backend (`test_workbench.py`, `DAY1_DAY2_STATUS.md`) explicitly flags unvalidated models with `NOT_IMPLEMENTED` and `NEEDS_DATA` labels rather than presenting mock scores as empirical results.

### 3.2 Critical Blockers & Academic Vulnerabilities
1. **Biometric Data Exposed in Git History:** An unencrypted legacy HDF5 biometric embedding database (`module1_face_recognition/database/face_embeddings.h5`) was committed to the repository in commit `98df922`. This constitutes a biometric data exposure risk under GDPR/biometric privacy standards and must be purged prior to public distribution.
2. **Divergence from Approved SRS AI Promises:** Earlier proposals and the signed SRS v1.0 promised fine-tuned Vision Transformers (ViT) for face and vehicle recognition, State Space Models (SSM/Mamba) for behavioral anomaly prediction, and local Vision-Language Models (LLaVA) for automated incident summaries. In the current runtime:
   - SFace ONNX is used instead of fine-tuned ViT.
   - Heuristic color histograms are used instead of deep vehicle Re-ID.
   - Rule-based dwell/visit counting (>30s dwell, >=3 visits in 24h) is used instead of SSMs.
   - Deterministic string templates are used instead of a multimodal VLM.
   - OpenCV contour analysis is used instead of neural license plate detection (caused by an upstream HuggingFace 404 URL).
3. **Absence of Quantitative Benchmark Validation:** The repository contains zero empirical evaluation metrics (ROC-AUC, EER, mAP, Rank-1/5, Character Error Rates) calculated over standard held-out test splits. Self-matching during enrollment was used as a sanity smoke check during development, which cannot be cited in an academic dissertation as model accuracy.
4. **Heuristic Anti-Spoofing Vulnerability:** The legacy anti-spoofing pipeline (`module1_face_recognition/anti_spoofing/`) relies on MediaPipe Face Mesh canonical 3D landmark Z-depth and LBP histogram entropy. Because MediaPipe fits a canonical 3D model to any 2D photograph, printed photos pass the depth check. Furthermore, the CNN texture classifier (`TextureCNN`) has no trained weights.
5. **Session Cookie Security Directives:** The HTTP session cookie (`ig_session`) lacks `SameSite=Lax/Strict` and `Secure` attributes, creating cross-site request forgery and cookie leakage risks if deployed outside `127.0.0.1`.

---

## 4. Academic Evaluation Summary (BS AI FYP Standards)

| Evaluation Dimension | Weight | Current Score (/10) | Potential Score (/10) | Evaluation Rationale |
|---|---|---|---|---|
| **Problem Relevance & Scope** | 10% | **8.5** | **9.5** | Highly relevant automated physical security domain; software-only scope is appropriate if AI depth is rigorous. |
| **AI/ML Technical Depth** | 20% | **4.0** | **9.0** | Current active path uses off-the-shelf ONNX models and color histograms. With RTX 4060 Ti, deep Re-ID and anti-spoofing can be trained. |
| **Originality & Novelty** | 10% | **4.5** | **8.5** | Novel multi-modal integration idea, but currently relies on classical heuristics (CLAHE, rules, templates). |
| **Research Methodology** | 10% | **3.5** | **9.0** | No ablation studies, no identity-disjoint splits, no baseline comparisons against published benchmarks. |
| **Model Evaluation Rigour** | 10% | **2.5** | **9.0** | Zero quantitative metric curves (ROC, DET, CMC, CER). Critical gap for an AI degree project. |
| **Software Architecture** | 10% | **8.5** | **9.5** | Outstanding asynchronous design, non-blocking video worker, clean FSM, and modular structure. |
| **System Integration** | 10% | **8.0** | **9.0** | End-to-end event pipeline from video ingestion to dashboard UI and virtual gate is fully operational. |
| **Security & Privacy** | 10% | **6.5** | **9.0** | AES-256-GCM encryption is excellent; weakened by committed legacy HDF5 in Git and missing cookie flags. |
| **Documentation Accuracy** | 5% | **5.0** | **9.0** | Significant drift between SRS v1.0 promises (ViT/SSM/LLaVA) and prototype reality; needs formal amendment. |
| **Viva Defensibility** | 5% | **5.0** | **9.5** | Highly vulnerable if students claim theoretical foundation models; completely defensible if defended honestly and upgraded. |
| **OVERALL PROJECT SCORE** | **100%** | **5.35 / 10** | **9.10 / 10** | **Current: Marginally Acceptable (Engineering High, AI Low). Potential: Outstanding First-Class FYP.** |

---

## 5. Prioritized 4-Week Roadmap Overview

Leveraging the newly confirmed **AMD Ryzen 7 5800X + NVIDIA RTX 4060 Ti 16GB** workstation, the project can be transformed into an outstanding BS AI project over 4 weeks:

```
[Week 1: P0 Security & Perception Stabilization]
  ├── Purge exposed HDF5 biometrics from Git history (git-filter-repo)
  ├── Harden session cookies (SameSite=Lax, Secure, cryptographic HMAC)
  ├── Restore neural plate detection (YOLOv8-nano license plate weights)
  ├── Implement CUDA inference engine for PyTorch/ONNX on RTX 4060 Ti
  └── Deliverable: Verified secure baseline + CUDA accelerated pipeline

[Week 2: P1 Deep Learning & Metric Validation]
  ├── Integrate & calibrate Deep Face Anti-Spoofing (MiniFASNet / FeatherNet ONNX)
  ├── Replace color histogram with deep Vehicle Re-ID (OSNet-IBN / ResNet50-ReID)
  ├── Quantitative Evaluation: Face ROC-AUC, EER, APCER/BPCER, Re-ID Rank-1/mAP
  └── Deliverable: Scientific evaluation tables with empirical benchmark metrics

[Week 3: P2 Advanced Intelligence & Analytics]
  ├── Learned low-light enhancement benchmark (Zero-DCE vs CLAHE on LLVIP)
  ├── Temporal anomaly baseline (Isolation Forest on feature vectors vs rules)
  ├── Local lightweight VLM incident explainer (SmolVLM-500M or Moondream2 on GPU)
  └── Deliverable: Multi-modal AI reporting grounded in visual evidence

[Week 4: P3 Viva Preparation & Academic Defense]
  ├── Update SRS v4.0 Scope Amendment & Final Project Report
  ├── Run 72-hour automated reliability & memory leak stress test
  ├── Record deterministic offline video demonstrations for viva backup
  └── Deliverable: Viva-ready presentation deck, mock defense, final report
```

---

## 6. Readiness Determination

**AUDIT VERDICT: CONDITIONALLY ACCEPTABLE FOR INTERIM 75% EVALUATION; REQUIRES P0/P1 UPGRADES FOR FINAL FYP-II DEFENSE.**

The software engineering, architectural robustness, and system integration of IntelliGate represent upper-decile undergraduate work. However, as a **Bachelor of Science in Artificial Intelligence** final year project, the system cannot be defended in its current state without addressing the AI/ML evaluation void. With 2–4 weeks of prioritized execution on the available RTX 4060 Ti GPU, IntelliGate can achieve an exemplary academic standard.
