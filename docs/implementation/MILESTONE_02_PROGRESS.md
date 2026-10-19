# INTELLIGATE — MILESTONE 2 & 2.1 PROGRESS & RECOVERY LOG
**Document ID:** `IG-M02-PROGRESS-2026-10-REV1`  
**Date:** October 18, 2026 (Revised Milestone 2.1)  
**Lead:** Principal AI Engineer, Senior ML Researcher & Software Development Lead  
**Hardware Platform:** NVIDIA GeForce RTX 4060 Ti (16GB VRAM) + AMD Ryzen 7 5800X (8C/16T)  
**Status:** MILESTONE 2.1 SCIENTIFIC INTEGRITY CORRECTIONS APPLIED & VALIDATED  

---

## 1. Outage Recovery & Scientific Audit Summary

1. **Context Recovery:** Recovered after power failure; verified working tree and test suite (57/57 passed).
2. **Milestone 2 Initial Benchmarking:** Implemented GPU pipeline hooks and ran initial benchmarks.
3. **Milestone 2.1 Independent Review & Corrections:**
   - Identified methodology deficiencies in face verification (synthetic perturbations vs real LFW).
   - Identified procedural rendering in Pakistani plate OCR evaluation.
   - Identified non-equivalent comparison between sustained GPU scene pipeline and older CPU detector baseline.
   - Designed and ran **fair paired apples-to-apples CPU vs GPU benchmarks** on identical frames ([`tools/benchmark_fair_cpu_gpu.py`](file:///D:/MZHR_FYP/INTELLIGATE/tools/benchmark_fair_cpu_gpu.py)).
   - Evaluated accuracy and decision parity (100% agreement on class labels, OCR text, and gate access decisions).
   - Audited and updated all documentation in `docs/experiments/` and `docs/implementation/`.

---

## 2. Updated Scientific Provenance & Execution Matrix

| Workstream ID | Workstream Description | Provenance Classification | Empirical Measurement |
| :--- | :--- | :--- | :--- |
| **WS-01** | CUDA Environment Verification | Physical Hardware | `torch.cuda.is_available() == True` (RTX 4060 Ti 16GB) |
| **WS-02** | YOLOv8n Isolated Inference | Component Microbenchmark | CPU: 124.09 ms vs GPU: **9.01 ms** (**13.77x speedup**, 101.18 FPS) |
| **WS-03** | EasyOCR Isolated Plate Inference | Component Microbenchmark | CPU: 60.15 ms vs GPU: **17.33 ms** (**3.47x speedup**, 57.70 FPS) |
| **WS-04** | Paired Scene Pipeline Benchmark | Paired System Benchmark | CPU: 56.93 ms (16.58 FPS) vs GPU: **49.36 ms** (**19.09 FPS**, **1.15x speedup**) |
| **WS-05** | Paired Vehicle Pipeline Benchmark | Paired System Benchmark | CPU: 16.06 ms (61.38 FPS) vs GPU: **9.59 ms** (**103.13 FPS**, **1.67x speedup**) |
| **WS-06** | Accuracy & Decision Parity Audit | Parity Audit | **100% YOLO agreement**, **100% OCR text agreement**, **Identical gate decisions** |
| **WS-07** | Pakistani Plate OCR | Synthetic Microbenchmark | 20 procedural plate crops (Exact match: 65.0%, CER: 15.52%) |
| **WS-08** | Real Pakistani ALPR Road Evaluation | Real Camera Evaluation | **BLOCKED** (Requires real CCTV road imagery acquisition) |
| **WS-09** | SFace Cosine Dot Product Stability | Synthetic Diagnostic | 1 reference face + 300 random vectors (Diagnostic gap: 0.9595) |
| **WS-10** | Official LFW Protocol Evaluation | Real Held-Out Protocol | **BLOCKED** (Requires official `lfw.tgz` dataset acquisition) |
| **WS-11** | Gate Access Security Policy | Security Invariant | Preserved (`trusted is True` required, face advisory `REVIEW_REQUIRED`) |
| **WS-12** | Regression Test Suite | Full Regression Suite | **57/57 tests passing** (0 failures, 100% passing) |

---

## 3. Current Blockers & Next Actions
1. **LFW Dataset Acquisition:** Blocked pending supervisor download authorization.
2. **Real Pakistani Vehicle Imagery:** Blocked pending real CCTV footage collection.
3. **Readiness for Milestone 3:** The core engineering stack, hardware acceleration, and application security policies are fully validated. Ready to begin Milestone 3 (Fail-Secure Gate UI alerts, SRS Amendment, and viva presentation packaging).
