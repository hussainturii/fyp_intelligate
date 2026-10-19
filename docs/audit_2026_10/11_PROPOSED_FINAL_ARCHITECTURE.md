# INTELLIGATE: PROPOSED TARGET ARCHITECTURE & SPECIFICATION

**Document ID:** AUD-2026-10-11  
**Target Repository:** `D:\MZHR_FYP\INTELLIGATE`  
**Architecture Style:** Modular Monolith with Asynchronous Ingestion & GPU Offloaded Inference  
**Audit Date:** October 18, 2026  

---

## 1. Architectural Philosophy & Core Principles

The target architecture is governed by four non-negotiable engineering principles:

1. **Strict Decoupling of Perception from Authority:**
   Artificial Intelligence models are probabilistic estimators with non-zero error distributions. An AI observation (e.g. a face match or appearance similarity) is an *advisory signal*, never an authoritative command. All access decisions and gate actuations are enforced by a deterministic, auditable **Central Access & Safety Policy** running a fail-secure state machine.
2. **Asynchronous Frame Ingestion & Dropping:**
   Video capture must never block on heavy neural network inference. The video ingestion layer operates on a separate thread, decoding frames continuously and maintaining a single-item drop-oldest buffer. If the inference worker takes 80 ms, intermediate video frames are safely dropped, guaranteeing zero preview lag.
3. **Explicit Confidence & Uncertainty Escalation:**
   Every perception result carries an explicit confidence score and method tag. High confidence on validated signals (e.g. exact registered plate $\ge 0.85$) triggers automated entry. Low confidence or unvalidated modalities (e.g. face matches without deep liveness, vehicle appearance fallback) escalate to a human operator via a 120-second sliding review window.
4. **Pragmatic Monolith over Distributed Microservices:**
   IntelliGate avoids unnecessary microservices, message brokers (Kafka/RabbitMQ), and distributed object stores (SeaweedFS). A cohesive, modular Python FastAPI monolith with in-memory thread queues and SQLite WAL persistence provides maximum developer velocity, zero deployment friction, and optimal viva defensibility.

---

## 2. End-to-End System Architecture Diagram

The target architecture spans eight functional tiers, explicitly color-coded by implementation status:
- **[GREEN]** Already Implemented & Operational
- **[BLUE]** Reusable with Configuration / Minor Changes
- **[AMBER]** Needs Replacement / Heuristic Upgrade
- **[PURPLE]** New Implementation

```mermaid
flowchart TD
    %% COLOR DEFINITIONS
    classDef implemented fill:#2ecc71,stroke:#27ae60,stroke-width:2px,color:#fff;
    classDef reusable fill:#3498db,stroke:#2980b9,stroke-width:2px,color:#fff;
    classDef replacement fill:#e67e22,stroke:#d35400,stroke-width:2px,color:#fff;
    classDef newimpl fill:#9b59b6,stroke:#8e44ad,stroke-width:2px,color:#fff;

    subgraph TIER1 ["1. Data Ingestion & Frame Scheduling"]
        A1[Live Webcam Stream Index 0/1]:::implemented --> B[Threaded CameraCapture camera_manager.py]:::implemented
        A2[Uploaded Surveillance MP4 Video]:::implemented --> B
        A3[Curated LLVIP Replay Dataset]:::implemented --> B
        B --> C[Single-Item Frame Buffer Drop-Oldest]:::implemented
        C --> D[Adaptive Frame Scheduler 5-15 FPS Worker]:::reusable
    end

    subgraph TIER2 ["2. Central Model Registry & GPU Engine"]
        D --> E[ModelRegistry & Hardware Manager]:::newimpl
        E -->|CUDA 13.3 / TensorRT Execution| F[(NVIDIA RTX 4060 Ti 16GB GPU)]:::implemented
        E --> G[Preprocessed RGB Tensor & BGR Frames]:::reusable
    end

    subgraph TIER3 ["3. Multi-Modal Vision & Perception Pipeline"]
        G --> H1[YuNet ONNX Face Detector 640x360]:::implemented
        H1 --> I1[SFace ONNX Face Embedder 128-D]:::implemented
        H1 --> I2[MiniFASNetV2 ONNX Anti-Spoofing Classifier]:::replacement
        
        G --> H2[YOLOv8-nano Scene & Vehicle Detector 416x416]:::implemented
        H2 --> I3[YOLOv8-nano Plate Localizer]:::replacement
        I3 --> I4[EasyOCR Plate Character Recognition]:::reusable
        H2 --> I5[OSNet-IBN Deep Vehicle Re-ID ONNX 512-D]:::replacement
        
        G --> H3[Zero-DCE Learned Low-Light Enhancement]:::newimpl
        H2 --> I6[ByteTrack Multi-Object Association Tracker]:::replacement
        I6 --> I7[Kinematic Isolation Forest Loitering Anomaly]:::newimpl
        I6 --> I8[Queue Corridor ROI Density Counter]:::reusable
    end

    subgraph TIER4 ["4. Identity Matching & Cryptographic Storage"]
        I1 --> J1[AES-256-GCM Enrolled Face Gallery Match]:::implemented
        I1 --> J2[Stranger Memory Clustering cpu_strangers]:::implemented
        I4 --> J3[VehicleDatabase Exact Plate Lookup]:::reusable
        I5 --> J4[VehicleDatabase Deep Re-ID Gallery Search]:::replacement
        J2 --> J5[Visitor Visit Frequency Ledger 24h Window]:::implemented
    end

    subgraph TIER5 ["5. Central Access Control & Safety Policy Engine"]
        I2 & J1 & J3 & J4 & J5 & I7 --> K[Central Access Policy SoftwareRuntime.observe]:::implemented
        
        K --> L{Evaluate Rules & Safety Boundaries}:::implemented
        L -->|Obstruction in Gate ROI or Lockdown| M1[Decision: DENY Reason: Interlock Active]:::implemented
        L -->|Exact Registered Plate & OCR Conf >= 85%| M2[Decision: ALLOW Reason: Registered Plate]:::implemented
        L -->|Known Face & Liveness >= Calibrated Tau| M3[Decision: ALLOW Reason: Verified Face]:::newimpl
        L -->|Face Liveness Unvalidated or Ambiguous| M4[Decision: REVIEW_REQUIRED Reason: Face Review]:::implemented
        L -->|Appearance Re-ID Match Only| M5[Decision: REVIEW_REQUIRED Reason: Appearance Fallback]:::implemented
        L -->|Unknown Face / Vehicle| M6[Decision: REVIEW_REQUIRED / stranger_alert]:::implemented
        
        M4 & M5 & M6 --> N[Create Operator Review Request TTL=120s]:::implemented
    end

    subgraph TIER6 ["6. Virtual Gate Simulation State Machine"]
        M2 & M3 --> O[SoftwareRuntime.command 'OPEN']:::implemented
        N -->|Operator Approves in Dashboard| O
        O --> P[Virtual Gate FSM State Transitions]:::implemented
        P --> Q[CLOSED -> OPENING 0.7s -> OPEN 5.0s -> CLOSING 0.7s -> CLOSED]:::implemented
        P -->|STOP Command or Obstruction| R[Immediate STOPPED State due=None]:::implemented
        P -->|LOCKDOWN Command| S[Immediate LOCKDOWN State]:::implemented
    end

    subgraph TIER7 ["7. Multimodal Incident Intelligence & Reporting"]
        L & P --> T[Incident Evidence Snapshot Dispatcher]:::newimpl
        T -->|Security Incident Frame Crop| U[Local Multimodal VLM Moondream2 GPU]:::newimpl
        U --> V[Factual 2-Sentence Incident Visual Summary]:::newimpl
        T --> W[Structured Event Aggregation Template Report]:::implemented
    end

    subgraph TIER8 ["8. Persistence, WebSockets & Operator Dashboard"]
        L & P & V & W --> X[(SQLite WAL Database logs/events.db)]:::reusable
        X --> Y[FastAPI Asynchronous Backend main.py]:::implemented
        Y -->|WebSocket /ws Broadcast| Z1[React 18 Dashboard SPA]:::implemented
        Y -->|MJPEG Multipart Stream| Z2[Camera Preview Canvas with Overlays]:::implemented
        Z1 --> AA[Operator Controls: Gate Panel, Reviews, DB, Test Center]:::implemented
    end
```

---

## 3. Implementation Status Taxonomy of Architectural Components

| Component / Layer | Source File / Module | Implementation Status | Technical Action in Roadmap |
|---|---|---|---|
| **Camera Capture Thread** | `camera_manager.py` | **Already Implemented** | Keep as-is. Thread-safe non-blocking capture buffer works reliably. |
| **YuNet Face Detector** | `vision.py` | **Already Implemented** | Keep as-is. 10.99 ms CPU latency; enable CUDA provider for ~3.0 ms GPU. |
| **SFace Face Embedder** | `vision.py` | **Already Implemented** | Keep as-is. 128-D L2-normalized embedding; enable CUDA provider. |
| **AES-256-GCM Biometric Store**| `vision.py`, `event_store.py` | **Already Implemented** | Keep as-is. Record-bound AAD prevents ciphertext transplant attacks. |
| **Virtual Gate State Machine**| `software_runtime.py` | **Already Implemented** | Keep as-is. Deterministic FSM, emergency stop preemption, restart recovery. |
| **Visitor Escalation Ledger**| `software_runtime.py` | **Already Implemented** | Keep as-is. 3 visits in 24h triggers HIGH alert. |
| **React 18 Operator Dashboard**| `module8_dashboard/frontend` | **Already Implemented** | Keep as-is. Optimize bundle size via code-splitting. |
| **Central Access Policy** | `software_runtime.py` | **Reusable with Changes** | Wire new deep liveness threshold into `observe()` to enable autonomous face opening. |
| **EasyOCR Engine** | `vision.py` | **Reusable with Changes** | Enable GPU device (`gpu=True`) to drop OCR latency from 380 ms to ~45 ms. |
| **SQLite Event Database** | `event_store.py` | **Reusable with Changes** | Add explicit Write-Ahead Logging (`PRAGMA journal_mode=WAL`) on connection bootstrap. |
| **License Plate Localizer** | `plate_detector.py` | **Needs Replacement** | Replace OpenCV contour fallback with local verified YOLOv8-nano plate weights. |
| **Face Anti-Spoofing** | `anti_spoofing/` | **Needs Replacement** | Replace untrained heuristics with pretrained **MiniFASNetV2 ONNX** model. |
| **Vehicle Appearance Re-ID** | `vision.py: appearance()` | **Needs Replacement** | Replace 768-D color histogram with **OSNet-IBN ONNX** metric embeddings. |
| **Object Tracking** | `vision.py: Tracker` | **Needs Replacement** | Replace basic IoU tracker with **ByteTrack** for occlusion resilience. |
| **Model Registry & Hardware Mgr**| `models/` | **New Implementation** | Centralized class that checks CUDA availability and validates model SHA-256 hashes. |
| **Learned Low-Light Enhancer**| `vision.py` | **New Implementation** | Integrate **Zero-DCE ONNX** for learned dynamic range adjustment. |
| **Kinematic Anomaly Detector**| `software_runtime.py` | **New Implementation** | Implement **Isolation Forest** over 5-D trajectory kinematics for loitering detection. |
| **Local Multimodal VLM** | `vlm_engine.py` | **New Implementation** | Deploy **Moondream2 (1.86B)** on RTX 4060 Ti GPU for visual incident summaries. |

---

## 4. Communication Contracts & Event Schemas

All inter-module communication is mediated by structured JSON event envelopes:

```json
{
  "event_id": 1420,
  "timestamp": "2026-10-18T12:30:00.123456Z",
  "source_id": "camera-0",
  "module": "face",
  "event_type": "access_granted",
  "identity": "Hussain",
  "confidence": 0.8842,
  "method": "SFace-ONNX",
  "decision": "ALLOW",
  "reason": "Registered face with verified deep liveness",
  "evidence": {
    "track_id": 4,
    "session_id": "sess-9a8b7c6d:visit-104",
    "liveness_model": "MiniFASNetV2",
    "liveness_score": 0.9621,
    "liveness_status": "PASS",
    "bounding_box": [180.2, 95.0, 400.5, 315.0],
    "simulated": true
  }
}
```

---

## 5. Safety Boundaries & Security Safeguards

1. **VLM Expository Isolation:**
   The multimodal Vision-Language Model (Moondream2) has **zero connection** to gate control endpoints. It operates purely as an event consumer: it reads saved incident image crops and writes descriptive text to the incident log. It cannot open, close, or stop the gate under any circumstances.
2. **Fail-Secure Interlock Priority:**
   Safety interlocks strictly supersede all perception decisions:
   $$\text{Decision} = \begin{cases} 
   \text{DENY (Interlock)}, & \text{if } \text{Obstruction} \lor \text{Lockdown} \\
   \text{ALLOW}, & \text{if } (\text{ExactPlate} \land \text{Conf} \ge 0.85) \lor (\text{KnownFace} \land \text{Liveness} \ge \tau) \\
   \text{REVIEW\_REQUIRED}, & \text{otherwise}
   \end{cases}$$
3. **Idempotent Actuation:**
   Every operator review approval and automated access event generates a cryptographically random command UUID. Repeating the request returns the prior state without re-triggering mechanical simulation timers.
