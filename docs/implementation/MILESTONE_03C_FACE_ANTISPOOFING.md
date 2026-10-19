# Milestone 3C: Deep Face Anti-Spoofing Integration & Security Architecture
**Document Reference:** `docs/implementation/MILESTONE_03C_FACE_ANTISPOOFING.md`  
**System:** IntelliGate Multimodal AI Access Control System  
**Lead Engineer:** Principal Computer Vision & Security Architect  
**Academic FYP:** COMSATS University Islamabad (SP23-BAI-057)  
**Date:** October 19, 2026  
**Status:** COMPLETE & VERIFIED  

---

## 1. Executive Summary & Objective

Following the acceptance of Milestone 3B.1 (YOLOv8n Pakistani License Plate Detector integration and verification), **Milestone 3C** integrates deep **Presentation Attack Detection (PAD / Face Anti-Spoofing)** into IntelliGate's active biometric perception pipeline.

Prior to Milestone 3C, facial processing relied solely on YuNet for face detection and SFace for 128-dimensional biometric identity matching. Face recognition events remained advisory-only and unvalidated for liveness (`validated_liveness: False`), routing strictly to operator review (`REVIEW_REQUIRED`). Milestone 3C introduces **MiniFASNetV2** from Minivision's *Silent-Face-Anti-Spoofing* to actively detect presentation attacks (printed photographs, digital screen replays, and video re-presentation attacks) before biometrics are presented to the access control policy engine.

---

## 2. Supply-Chain & Checkpoint Integrity

All model weights in IntelliGate must satisfy cryptographic supply-chain verification before being loaded into active memory.

### 2.1 Verified Model Weights Manifest
The MiniFASNetV2 weights file was verified and registered into `models/trusted_manifest.json`:

```json
"2.7_80x80_MiniFASNetV2.pth": {
  "relative_path": "models/2.7_80x80_MiniFASNetV2.pth",
  "sha256": "a5eb02e1843f19b5386b953cc4c9f011c3f985d0ee2bb9819eea9a142099bec0",
  "size_bytes": 1849453,
  "format": "PyTorch",
  "license": "Apache-2.0",
  "source": "https://github.com/minivision-ai/Silent-Face-Anti-Spoofing",
  "task": "presentation_attack_detection",
  "input_resolution": [80, 80]
}
```

### 2.2 Integrity Enforcement & Tamper Rejection
- **Integrity Validation:** Every initialization of `MiniFASNetV2Detector` runs `verify_model_file("models/2.7_80x80_MiniFASNetV2.pth", strict=True)`.
- **Safe PyTorch Loading:** Uses `torch.load(..., weights_only=True)` to prevent arbitrary code execution vulnerabilities common in unvalidated pickle payloads.
- **Fail-Closed on Tampering:** Any hash mismatch or byte corruption immediately aborts initialization with a `RuntimeError`, preventing compromised weights from executing.

---

## 3. Deep Learning Architecture & Preprocessing Specifications

### 3.1 Network Architecture (`MiniFASNetV2`)
MiniFASNetV2 is an efficient, depthwise-separable convolutional neural network specifically engineered for low-latency facial anti-spoofing on edge and embedded hardware.

- **Parameter Count:** ~0.43 million parameters (1.85 MB serialized weights).
- **Backbone Building Blocks:**
  - `Conv_block`: $3 \times 3$ Conv2d + BatchNorm2d + PReLU.
  - `Depth_Wise`: $1 \times 1$ pointwise conv $\rightarrow$ $3 \times 3$ depthwise conv (`groups=in_c`) $\rightarrow$ $1 \times 1$ linear projection block.
  - `Residual`: Sequentially stacked depthwise separable blocks with residual shortcut connections.
- **Channel Multipliers (`keep_dict['1.8M_']`):**
  - Stem: 32 channels.
  - Stage 1 (`conv_23`): 32 in $\rightarrow$ 103 intermediate $\rightarrow$ 64 out.
  - Stage 2 (`conv_3`): 4 residual blocks, 13 intermediate channels, 64 output channels.
  - Stage 3 (`conv_34`): 64 in $\rightarrow$ 231 intermediate $\rightarrow$ 128 out.
  - Stage 4 (`conv_4`): 6 residual blocks with asymmetric channel expansions (`[231, 52, 26, 77, 26, 26]`), 128 out.
  - Stage 5 (`conv_45`): 128 in $\rightarrow$ 308 intermediate $\rightarrow$ 128 out.
  - Stage 6 (`conv_5`): 2 residual blocks, 26 intermediate channels, 128 out.
  - Head (`conv_6_sep`, `conv_6_dw`): $1 \times 1$ Conv (512 out) $\rightarrow$ Depthwise Conv with **$5 \times 5$ kernel** (matching $80 \times 80$ input) $\rightarrow$ Flatten $\rightarrow$ Linear(512, 128) $\rightarrow$ BatchNorm1d $\rightarrow$ Linear(128, 3).

### 3.2 Preprocessing & Scaling Rules
1. **Bounding Box Expansion ($2.7\times$ Scale):**
   The model requires contextual background surrounding the face to identify display bezels, paper borders, and reflection artifacts.
   $$\text{scale} = 2.7$$
   $$\text{effective\_scale} = \min\left(\frac{H - 1}{h}, \min\left(\frac{W - 1}{w}, 2.7\right)\right)$$
   $$\text{new\_w} = w \times \text{effective\_scale}, \quad \text{new\_h} = h \times \text{effective\_scale}$$
   $$\text{center} = (x + w/2, y + h/2)$$
2. **Boundary Clamping:**
   If the expanded bounding box exceeds canvas boundaries $[0, W-1]$ or $[0, H-1]$, coordinates are shifted and clamped. If boundary clipping occurs, the crop is tagged with `pad_quality_status = "truncated_boundary"`.
3. **Color Space & Input Tensor:**
   - Color Format: **BGR** (OpenCV standard representation).
   - Normalization: $x / 255.0 \in [0.0, 1.0]$ via `torch.from_numpy(...).float().div(255.0)`.
   - **Crucial Requirement Met:** No ImageNet mean/standard deviation subtraction. Input is strictly scaled BGR in $[0.0, 1.0]$.
4. **Resolution:** Standardized $80 \times 80$ pixels via bilinear interpolation.

### 3.3 Output Class Mapping
The model predicts 3 class logits, converted to normalized probabilities via Softmax:
- **Class 0:** Presentation Attack — Print / Paper Spoof
- **Class 1:** **Bona Fide Presentation (Live Face)**
- **Class 2:** Presentation Attack — Digital Screen Replay Spoof

The bona fide liveness score is strictly defined as:
$$p_{\text{live}} = P(\text{class} = 1)$$

---

## 4. Pipeline Integration Architecture

```
[ Recorded Video / Replay Frame ]
                │
                ▼
      [ OpenCV YuNet (640x360) ]  ── Face Detection & 5 Landmarks
                │
                ▼
    [ MiniFASNetV2 Deep PAD ]   ── 2.7x Scale Crop, 80x80 BGR, CUDA/CPU
                │
         [ Quality Check ]
         ├── Crop < 35px ──► pad_status: NOT_EVALUATED
         ├── Truncated   ──► pad_status: INCONCLUSIVE
         ├── Attack Suspect► pad_status: ATTACK_SUSPECTED ──► Event: spoof_detected
         └── High Live p ──► pad_status: BONA_FIDE_CANDIDATE
                │
                ▼
     [ OpenCV SFace (112x112) ]  ── 128-dim Cosine Biometric Feature Matching
                │
                ▼
       [ Advisory Event ]        ── Event with Full PAD Metadata & Provenance
                │
                ▼
      [ SoftwareRuntime ]        ── Central Gate Access Policy Evaluation
         ├── Face Event ──────────► NEVER ALLOW! Decision: REVIEW_REQUIRED
         └── Spoof Event ─────────► Decision: DENY + HIGH Severity Security Alert
                │
                ▼
    [ Real-Time React Dashboard ]── Live Camera Stream & Review Requests
```

### 4.1 Conservative PAD Status Categories
No single uncalibrated scalar is treated as proof of human life. The pipeline outputs four conservative statuses:
1. `BONA_FIDE_CANDIDATE`: $p_{\text{live}} \ge 0.80$, predicted class is 1, and crop boundary is untruncated.
2. `ATTACK_SUSPECTED`: $p_{\text{live}} < 0.40$ or predicted class is in $\{0, 2\}$.
3. `INCONCLUSIVE`: $0.40 \le p_{\text{live}} < 0.80$ or crop boundary truncated by image borders.
4. `NOT_EVALUATED`: Face missing, degraded input, or face dimensions below minimum resolution ($< 35\text{ px}$).

### 4.2 Non-Negotiable Gate Security Invariant
**Strict Invariant Enforced:** Face recognition and PAD must **never independently trigger gate opening (`ALLOW`)**.
Regardless of whether PAD predicts `BONA_FIDE_CANDIDATE`:
- Facial recognition remains strictly advisory.
- Biometric events route to `REVIEW_REQUIRED`.
- Gate opening requires authorized human operator confirmation or exact-plate vehicle authorization ($\ge 0.85$ OCR confidence).
- Suspected spoof attacks route to `DENY` and trigger immediate high-priority security alarms.

---

## 5. Software & Regression Test Verification

The integration was validated with a dedicated regression test suite (`tests/test_milestone3c_anti_spoofing.py`) covering 11 specific test cases:

| Test Case | Description | Result |
| :--- | :--- | :---: |
| `test_checkpoint_integrity_and_manifest` | Verifies SHA-256 manifest registration and file integrity | **PASS** |
| `test_tampered_checkpoint_rejection` | Validates strict rejection of corrupted weights | **PASS** |
| `test_device_selection_and_cpu_fallback` | Validates explicit CPU fallback and CUDA device loading | **PASS** |
| `test_valid_face_crop_processing` | Validates $2.7\times$ crop, $80\times80$ resizing, and probability sum | **PASS** |
| `test_empty_and_degenerate_crops` | Validates safe degradation on null frames and small crops | **PASS** |
| `test_multiple_detected_faces` | Validates multi-face processing in a single video frame | **PASS** |
| `test_network_independent_pad_inference` | Validates 100% offline air-gapped startup & execution | **PASS** |
| `test_face_recognition_and_pad_never_opens_gate`| Confirms face events never grant automatic `ALLOW` | **PASS** |
| `test_suspected_spoof_routes_to_deny_and_alert` | Confirms `ATTACK_SUSPECTED` triggers `DENY` & security alert | **PASS** |
| `test_vehicle_exact_plate_authorization_preserved` | Verifies vehicle exact-plate authorization remains intact | **PASS** |
| `test_safety_interlocks_remain_authoritative` | Confirms STOP, LOCKDOWN, & Obstruction override biometrics | **PASS** |

**Full System Test Results:** 78 passed across the entire repository in 35.40s.  
**Frontend Production Build:** Built cleanly in 3.18s (`vite build`).

---

## 6. Exact Files Modified / Created

1. `models/trusted_manifest.json`: Added `2.7_80x80_MiniFASNetV2.pth` entry with SHA-256 `a5eb02e1843f19b5386b953cc4c9f011c3f985d0ee2bb9819eea9a142099bec0`.
2. `module1_face_recognition/anti_spoofing/deep_pad.py`: Implemented `MiniFASNet` architecture and `MiniFASNetV2Detector` inference class.
3. `module1_face_recognition/anti_spoofing/__init__.py`: Exported `MiniFASNetV2Detector`.
4. `module1_face_recognition/face_detector.py`: Made `facenet_pytorch` import graceful.
5. `module1_face_recognition/face_recognizer.py`: Made `timm` import graceful.
6. `module8_dashboard/backend/vision.py`: Integrated `MiniFASNetV2Detector` into `ReplayPipeline`, added PAD stage latency profiling, attached PAD metadata to face results, and updated event dispatching.
7. `module8_dashboard/frontend/src/components/CameraView.jsx`: Added real-time indicators for `pad_attack_suspected` and `pad_inconclusive`.
8. `module8_dashboard/frontend/src/components/Dashboard.jsx`: Added badge mapping for `spoof_detected`.
9. `module8_dashboard/frontend/src/components/EventLog.jsx`: Added badge mapping for `spoof_detected`.
10. `tests/test_milestone3c_anti_spoofing.py`: Comprehensive 11-test regression suite.
11. `tools/benchmark_face_pipeline.py`: Benchmark runner comparing CPU vs GPU latency and throughput.
