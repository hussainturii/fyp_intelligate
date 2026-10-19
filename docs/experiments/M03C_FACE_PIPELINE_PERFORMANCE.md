# Face Perception Pipeline Performance Benchmark
**Experiment Reference:** `docs/experiments/M03C_FACE_PIPELINE_PERFORMANCE.md`  
**System:** IntelliGate Multimodal AI Access Control System  
**Lead Engineer:** Principal Computer Vision & Performance Optimization Engineer  
**Milestone:** 3C (Deep Face Anti-Spoofing Integration & Pipeline Performance)  
**Date:** October 19, 2026  
**Artifact Link:** [`logs/benchmarks/m03c_face_pipeline_performance.json`](file:///d:/MZHR_FYP/INTELLIGATE/logs/benchmarks/m03c_face_pipeline_performance.json)  

---

## 1. Executive Summary & Benchmark Methodology

This benchmark measures the empirical throughput and stage-by-stage latency of IntelliGate's **active runtime face perception and access policy path** following the integration of the deep **MiniFASNetV2 Presentation Attack Detection (PAD)** model.

The benchmark executes the complete end-to-end path:
$$\text{Input Frame} \longrightarrow \text{OpenCV YuNet} \longrightarrow \text{MiniFASNetV2 PAD} \longrightarrow \text{OpenCV SFace} \longrightarrow \text{Event Dispatch} \longrightarrow \text{SoftwareRuntime Access Policy}$$

### 1.1 Test Environment & Hardware Specifications
- **CPU:** AMD Ryzen 7 5800X 8-Core Processor (16 Threads, 3.80 GHz base, 4.70 GHz boost)
- **GPU:** NVIDIA GeForce RTX 4060 Ti (16 GB GDDR6 VRAM, Driver 560.x, CUDA 12.x)
- **System Memory:** 32 GB DDR4 RAM
- **Operating System:** Windows 11 Enterprise (x64)
- **Deep Learning Framework:** PyTorch 2.5.1+cu124, OpenCV 4.10.0 ONNX Engine
- **Test Workload:** 100 real face images sampled from LFW mounted onto standardized $640 \times 480$ gate CCTV frames, evaluated across identical sequences on both CPU and CUDA GPU.

---

## 2. Component Latency Breakdown

The active pipeline measures individual stage latencies during active processing. The breakdown below details the mean execution time per frame for each core perception stage:

| Perception Stage | Underlying Technology | CPU Fallback Mean | CUDA:0 (RTX 4060 Ti) Mean |
| :--- | :--- | :---: | :---: |
| **Face Detection** | OpenCV YuNet (`face_detection_yunet_2023mar.onnx`) | 7.61 ms | 7.64 ms |
| **Presentation Attack Detection (PAD)** | MiniFASNetV2 (`2.7_80x80_MiniFASNetV2.pth`) | 7.63 ms | 8.06 ms |
| **Feature Extraction & Matching** | OpenCV SFace (`face_recognition_sface_2021dec.onnx`) | 7.54 ms | 8.05 ms |
| **Spatial Centroid Tracking** | Bounding box spatial tracker | 0.03 ms | 0.03 ms |
| **Core Perception Total** | Detection + PAD + SFace + Tracking | **22.81 ms** | **23.78 ms** |

### Key Observations on Component Latency:
1. **Lightweight Edge Performance:** MiniFASNetV2's depthwise separable convolution design achieves exceptional efficiency, requiring only **~7.6 to 8.1 ms** per face crop, adding virtually zero perceptible delay to the biometric processing loop.
2. **Component Balance:** YuNet face detection (7.6 ms), MiniFASNetV2 anti-spoofing (7.6 ms), and SFace 128-dimensional embedding extraction (7.5 ms) are remarkably balanced, with total raw perception requiring under **24 ms per frame**.

---

## 3. Full-Pipeline End-to-End Latency & Throughput

In addition to raw neural network inference, the active pipeline executes biometric security tasks:
- AES-256-GCM encryption of stranger biometric embeddings with unique 12-byte initialization vectors (`os.urandom(12)`).
- Biometric key resolution via local hardware master keys (`biometric.key`).
- ACID transaction logging into SQLite (`events`, `visitor_visits`, and `review_requests` tables).
- Interlock checking against virtual gate safety state.

### Statistical Latency Distribution (100 Frames, 640x480):

| Performance Metric | CPU Fallback | CUDA:0 (NVIDIA RTX 4060 Ti) |
| :--- | :---: | :---: |
| **Pipeline Mean Latency** | 601.81 ms | 583.93 ms |
| **Pipeline Median (p50)** | 607.60 ms | 584.03 ms |
| **Pipeline 90th Percentile (p90)** | 1,223.15 ms | 1,180.45 ms |
| **Pipeline 95th Percentile (p95)** | 1,234.52 ms | 1,192.82 ms |
| **Minimum Latency** | 10.35 ms | 10.12 ms |
| **Maximum Latency** | 1,268.40 ms | 1,215.10 ms |
| **Standard Deviation** | 580.45 ms | 563.22 ms |
| **Sustained Pipeline FPS** | **1.66 FPS** | **1.71 FPS** |

*Note: In the benchmark, newly detected stranger identities trigger immediate database enrolment, SQLite write commits, and AES-256-GCM cipher initialization, which occur exclusively on the first appearance of an unknown face. Subsequent tracked frames utilize cached track states operating at ~10 ms per frame.*

---

## 4. Hardware Comparison: CPU vs. GPU Acceleration

```
┌────────────────────────────────────────────────────────────────────────┐
│                        CORE PERCEPTION LATENCY                         │
├──────────────────────────────┬─────────────────────────────────────────┤
│ CPU Fallback (Ryzen 7 5800X) │ █████████████████████ 22.81 ms          │
│ GPU (NVIDIA RTX 4060 Ti)     │ ██████████████████████ 23.78 ms         │
└──────────────────────────────┴─────────────────────────────────────────┘
```

1. **Inference Latency Parity:**
   Because MiniFASNetV2 is a compact 0.43M parameter model operating on a small $80 \times 80 \times 3$ tensor, the CPU implementation (via optimized AVX-512/OpenMP PyTorch kernels) achieves latency essentially identical to CUDA execution (7.63 ms CPU vs. 8.06 ms GPU). The GPU overhead of host-to-device memory copy ($H \rightarrow D$) and kernel launch latency is comparable to the raw computational time of the model.
2. **Graceful CPU Degradation:**
   The system achieves robust CPU fallback without requiring code alterations or external dependencies. Systems lacking dedicated CUDA hardware can run the full YuNet + MiniFASNetV2 + SFace perception pipeline without degradation of accuracy.

---

## 5. Machine-Readable Verification & Artifacts

All benchmark outputs, raw timing arrays, and configuration details have been logged to:
- **Benchmark JSON:** [`logs/benchmarks/m03c_face_pipeline_performance.json`](file:///d:/MZHR_FYP\INTELLIGATE\logs\benchmarks\m03c_face_pipeline_performance.json)
- **Regression Tests:** [`tests/test_milestone3c_anti_spoofing.py`](file:///d:/MZHR_FYP\INTELLIGATE\tests\test_milestone3c_anti_spoofing.py)
