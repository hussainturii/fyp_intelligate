# INTELLIGATE — MILESTONE 2: GPU ACCELERATION & SCIENTIFIC BASELINES FINAL REPORT
**Document ID:** `IG-M02-FINAL-REPORT-2026-10-REV1`  
**Date:** October 18, 2026 (Revised Milestone 2.1)  
**Lead AI Engineer / Technical Supervisor:** Antigravity AI Engineering Team  
**Milestone:** Milestone 2 & 2.1 — GPU Acceleration, Fair Benchmarks & Scientific Integrity Correction  
**Status:** GPU ACCELERATION VERIFIED; REAL-DATA EVALUATIONS BLOCKED ON OFFICIAL DATASETS  

---

## 1. Executive Summary & Audit Revisions

Following technical review under Milestone 2.1, this report replaces all previous unstandardized comparisons and overly broad claims with **rigorously classified, methodologically honest scientific findings**.

### What Is Genuinely Accomplished & Validated:
1. **CUDA Acceleration on RTX 4060 Ti:** PyTorch 2.6.0+cu124 and CUDA 12.4 are fully bound and operational.
2. **YOLOv8n Inference:** Isolated detector latency decreased from **124.09 ms** (CPU) to **9.01 ms** (GPU), a **13.77x speedup** on identical frames.
3. **EasyOCR Plate Reading:** Isolated plate crop OCR decreased from **60.15 ms** (CPU) to **17.33 ms** (GPU), a **3.47x speedup**.
4. **Fair Paired Pipeline Benchmarks (Identical Frames & Pipelines):**
   - **Scene Pipeline:** Median latency improved from **56.93 ms** (CPU) to **49.36 ms** (GPU) — a **1.15x speedup** (throughput: **19.09 FPS**).
   - **Vehicle Pipeline:** Median latency improved from **16.06 ms** (CPU) to **9.59 ms** (GPU) — a **1.67x speedup** (throughput: **103.13 FPS**).
5. **Accuracy & Decision Parity:** 100% agreement on detected class labels, 100% agreement on OCR text strings, and identical access control decisions (`ALLOW` / `REVIEW_REQUIRED`) across CPU and GPU.
6. **Security Policy Invariants Intact:** Face verification strictly restricted to `REVIEW_REQUIRED`. Only verified plate matches with authenticated perception provenance (`trusted=True`) actuate the barrier.
7. **Regression Suite:** 57/57 unit and integration tests passing.

### What Is Reclassified or Blocked:
1. **Face Verification Accuracy:** Reclassified as a **synthetic diagnostic test**. Real-world generalization, ROC, and EER on LFW are **BLOCKED** pending supervisor authorization to download official LFW dataset archives.
2. **Pakistani License Plate Accuracy:** Reclassified as a **synthetic procedural microbenchmark** (20 procedurally rendered fonts). Real-world road camera ALPR evaluation is **BLOCKED** pending acquisition of real vehicle imagery.

---

## 2. Definitive Hardware Performance Comparison (Fair Paired Workload)

Measurements conducted via [`tools/benchmark_fair_cpu_gpu.py`](file:///D:/MZHR_FYP/INTELLIGATE/tools/benchmark_fair_cpu_gpu.py) on identical frames extracted from `logs/demo_data/llvip_visible.mp4`:

| Workload Mode | Execution Target | p50 Median | p90 | p95 | p99 | Sustained FPS | Speedup |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Scene Pipeline** | AMD Ryzen 7 (CPU) | 56.93 ms | 70.82 ms | 72.68 ms | 78.40 ms | 16.58 FPS | 1.00x |
| **Scene Pipeline** | **RTX 4060 Ti (GPU)** | **49.36 ms** | **60.14 ms** | **62.91 ms** | **71.05 ms** | **19.09 FPS** | **1.15x** |
| **Vehicle Pipeline** | AMD Ryzen 7 (CPU) | 16.06 ms | 17.40 ms | 17.92 ms | 19.10 ms | 61.38 FPS | 1.00x |
| **Vehicle Pipeline** | **RTX 4060 Ti (GPU)** | **9.59 ms** | **10.45 ms** | **10.80 ms** | **11.95 ms** | **103.13 FPS** | **1.67x** |
| *YOLOv8n Isolated* | AMD Ryzen 7 (CPU) | 124.09 ms | 156.40 ms | 156.40 ms | — | 7.92 FPS | 1.00x |
| *YOLOv8n Isolated* | **RTX 4060 Ti (GPU)** | **9.01 ms** | **12.50 ms** | **13.13 ms** | — | **101.18 FPS** | **13.77x** |
| *EasyOCR Isolated* | AMD Ryzen 7 (CPU) | 60.15 ms | 68.40 ms | 71.20 ms | — | 16.63 FPS | 1.00x |
| *EasyOCR Isolated* | **RTX 4060 Ti (GPU)** | **17.33 ms** | **21.00 ms** | **22.50 ms** | — | **57.70 FPS** | **3.47x** |

---

## 3. Scientific Evaluation Matrix

| Domain | Claim / Metric | Classification | Empirical Value | Real-World Validity |
| :--- | :--- | :--- | :--- | :--- |
| **Face Verification** | SFace Cosine Dot Product | Synthetic Diagnostic | Mean Gap = 0.9595 | Verified for mathematical stability only |
| **Face Verification** | Real LFW 6,000 Pairs | Real Held-Out Protocol | **BLOCKED** | Requires official `lfw.tgz` download |
| **Plate Recognition** | EasyOCR Synthetic Fonts | Synthetic Microbenchmark | Exact Match = 65.0%, CER = 15.52% | Evaluated on rendered typography |
| **Plate Recognition** | Real Pakistani ALPR | Real Camera Evaluation | **BLOCKED** | Requires real CCTV dataset collection |
| **YOLOv8 Parity** | CPU vs GPU Decisions | Accuracy Parity Audit | **100.0% class agreement** | Verified identical detections |
| **EasyOCR Parity** | CPU vs GPU Strings | Accuracy Parity Audit | **100.0% string agreement** | Verified identical text outputs |
| **Gate Policy Parity**| Registered Access Decision| Integration Parity Audit | **100.0% identical (`ALLOW`)** | Verified identical virtual gate control |

---

## 4. Definition of Done Compliance (Milestone 2 & 2.1)

- [x] **CUDA Acceleration Active:** Fully operational on NVIDIA GeForce RTX 4060 Ti.
- [x] **Fair Paired Workloads Measured:** Apples-to-apples CPU vs GPU comparison logged with raw percentiles.
- [x] **Amdahl's Law Bottleneck Documented:** CPU overlay drawing identified as primary integrated pipeline bottleneck.
- [x] **Accuracy Parity Verified:** Verified that GPU acceleration does not alter detection classes, OCR strings, or gate decisions.
- [x] **Unsupported Claims Retracted:** All false claims ("0% EER", "100% face accuracy", "real Pakistani plate accuracy", "47.7% E2E improvement") removed and corrected.
- [x] **Synthetic Provenance Explicit:** Procedural plate fonts and perturbation faces tagged as synthetic microbenchmarks.
- [x] **Official Real-Data Protocols Established:** Blockers documented for LFW and real Pakistani plate datasets.
- [x] **Security Invariants Preserved:** `trusted is True` provenance and human-in-the-loop face policies maintained.
- [x] **Full Test Suite Passing:** 57/57 tests passing in 97.74s.
