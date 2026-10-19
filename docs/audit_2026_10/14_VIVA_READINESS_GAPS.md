# INTELLIGATE: VIVA READINESS GAPS, EXAMINER TRAPS & EVIDENCE APPENDIX

**Document ID:** AUD-2026-10-14  
**Target Repository:** `D:\MZHR_FYP\INTELLIGATE`  
**Execution Baseline Date:** October 18, 2026  
**Auditing Focus:** Defensive Viva Preparedness, Live Demonstration Failsafes & Reproducibility Appendix  

---

## 1. Defensive Gap Analysis & Viva Vulnerability Matrix

This gap analysis identifies the exact technical weaknesses that an astute academic examination committee will target during the oral viva defense.

| Viva Vulnerability Area | Severity | Academic Hazard | Defensive Technical Strategy |
|---|---|---|---|
| **Claiming Foundation ViT when SFace runs** | **CRITICAL** | Immediate loss of research credibility; panel assumes students do not know what models are in their own code. | Transparently acknowledge SFace ONNX. Defend it on edge latency (7.8 ms vs. 420 ms), memory footprint, and CASIA-WebFace metric tuning. Present ablation data. |
| **Claiming Optical CLAHE creates Thermal Data** | **CRITICAL** | Severe penalty for scientific illiteracy; violates fundamental optical physics. | Delineate Visible spectrum ($380–740\text{ nm}$) vs. Long-Wave Infrared ($8–14\text{ }\mu\text{m}$). Reframe CLAHE as optical contrast enhancement; cite LLVIP as recorded LWIR evaluation. |
| **Self-Matching Cited as Recognition Accuracy** | **CRITICAL** | Accusation of data fabrication or basic research methodology incompetence. | Clarify that enrollment matching was a software sanity check, not validation accuracy. Present held-out LFW / CelebA ROC curves with identity-disjoint splits. |
| **Claiming 99.4% OCR without Test Data** | **HIGH** | Panel requests to see the test set and character confusion matrix, exposing that no evaluation was run. | Disclose that 99.4% was an uncalibrated design target. Present empirical evaluation on 100 annotated Pakistani plates with Levenshtein Character Error Rate (CER). |
| **Claiming Mamba / SSM Intent Prediction** | **HIGH** | Theoretical question on State Space Model discretization matrices ($A, B, C$) catches students unprepared. | Clarify that SSM research was deferred due to lack of longitudinal trajectory datasets. Defend the deterministic 30s dwell and 3-visit escalation rules as auditable industrial baselines. |
| **Live Photo Attack Spoofs Face Recognition** | **HIGH** | An examiner holds up their smartphone screen or ID card to the webcam and the gate unlocks. | Highlight the central fail-secure policy: face matches force `REVIEW_REQUIRED` and NEVER open autonomously. Present MiniFASNetV2 Fourier frequency analysis. |

---

## 2. Live Demonstration Failure Modes & Failsafe Playbook

During a university FYP defense, live hardware demos fail with high statistical probability due to uncalibrated room lighting, network blocks, or camera enumeration changes.

### Failure Mode 1: USB / Laptop Webcam Disconnect or Index Shift
- **Symptom:** Operator clicks "Open live camera"; dashboard displays black canvas or error: `Cannot open camera index 0`.
- **Root Cause:** Windows Camera privacy toggles, another app (Zoom, Teams) locking the device, or index shifted from 0 to 1.
- **Immediate Failsafe Action:**
  1. Try index 1 in the camera index dropdown.
  2. If the webcam remains locked, immediately switch to **Curated Video Replay**: select `face_smoke.mp4` or `llvip_visible.mp4` from the dropdown and click "Replay".
  3. If video playback fails, switch to the pre-rendered 1080p backup video recordings (`docs/viva/videos/01_face_recognition_approval.mp4`).

### Failure Mode 2: Extreme Backlight / Under-Exposure in Viva Room
- **Symptom:** YuNet detector fails to find face bounding boxes due to fluorescent glare or backlighting from an examination room window.
- **Root Cause:** SFace and YuNet require minimum contrast and resolution ($\ge 35\text{ px}$ face width).
- **Immediate Failsafe Action:**
  1. Have the participant move closer ($< 1\text{ meter}$) and face the primary light source.
  2. Switch camera processing mode to **Low Light Enhancement (CLAHE)** via the dropdown to expand local dynamic range.
  3. Explain to examiners: *"Notice that our dashboard displays an amber warning 'Face #1: move closer' rather than making an unreliable false prediction, fulfilling our fail-secure policy."*

### Failure Mode 3: License Plate OCR Misread on Demo Image
- **Symptom:** A test vehicle image is uploaded, but EasyOCR transposes a character (e.g. reading `LEA1284` instead of `LEA1234`).
- **Root Cause:** OCR confidence drops below the 85% autonomous threshold.
- **Immediate Failsafe Action:**
  1. Do not panic. The system is designed to handle this: the vehicle is flagged as an `appearance_candidate` with `REVIEW_REQUIRED`.
  2. Explain to the panel: *"Because the OCR confidence was 78% (below our 85% safety threshold), the gate policy safely denied autonomous entry and created an operator review request. This demonstrates our safety boundaries in action."*

---

## 3. Evidence Appendix: Reproducible Verification Commands

Every engineering and scientific claim in the audit reports can be independently verified on the host workstation using the following commands:

### A. Environment & Hardware Verification
```powershell
# 1. Verify NVIDIA RTX 4060 Ti GPU & VRAM
nvidia-smi

# 2. Verify PyTorch 2.7.1 CUDA execution capability
& "D:\MZHR_FYP\fyp_intelligate-1\.venv\Scripts\python.exe" -c "import torch; print('CUDA Available:', torch.cuda.is_available()); print('Device:', torch.cuda.get_device_name(0)); print('Capability:', torch.cuda.get_device_capability(0))"

# 3. Verify CPU Cores and Physical Memory
Get-CimInstance Win32_Processor | Select-Object Name, NumberOfCores, NumberOfLogicalProcessors
$os = Get-CimInstance Win32_OperatingSystem; [math]::Round($os.TotalVisibleMemorySize/1MB, 2).ToString() + " GB Total RAM, " + [math]::Round($os.FreePhysicalMemory/1MB, 2).ToString() + " GB Free RAM"
```

### B. Automated Test Suite Execution (100% Pass Verification)
```powershell
# Run the complete automated test suite (45 tests)
& "D:\MZHR_FYP\fyp_intelligate-1\.venv\Scripts\python.exe" -m pytest tests -v
```

### C. Isolated Component Smoke Verification (Test Center)
```powershell
# Run the 10-module component verification harness
& "D:\MZHR_FYP\fyp_intelligate-1\.venv\Scripts\python.exe" tools/run_project_checks.py
```

### D. Empirical Latency & Performance Benchmarks
```powershell
# Run CPU inference latency profiler on recorded video
& "D:\MZHR_FYP\fyp_intelligate-1\.venv\Scripts\python.exe" tools/benchmark_cpu.py --video logs/demo_data/llvip_visible.mp4 --frames 8

# Run LLVIP paired visible vs. infrared exploration benchmark
& "D:\MZHR_FYP\fyp_intelligate-1\.venv\Scripts\python.exe" tools/benchmark_llvip.py
```

### E. Frontend Production Build Verification
```powershell
# Verify that the React 18 single-page dashboard compiles cleanly
cd module8_dashboard/frontend
npm run build
cd ../..
```

### F. Model Checksum & Integrity Validation
```powershell
# Verify SHA-256 hashes of all active vision models against download manifest
Get-FileHash models/face_detection_yunet_2023mar.onnx, models/face_recognition_sface_2021dec.onnx, yolov8n.pt | Format-Table -AutoSize
```

### G. Live Application Launch Command
```powershell
# Launch the full operational application on local loopback
& "D:\MZHR_FYP\fyp_intelligate-1\.venv\Scripts\python.exe" -m uvicorn module8_dashboard.backend.main:app --host 127.0.0.1 --port 8000
# Access dashboard at: http://127.0.0.1:8000
```
