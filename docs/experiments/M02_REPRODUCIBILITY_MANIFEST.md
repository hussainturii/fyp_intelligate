# INTELLIGATE — EXPERIMENT REPORT: M02 REPRODUCIBILITY MANIFEST
**Document ID:** `IG-EXP-M02-REPRO-2026-10-REV1`  
**Date:** October 18, 2026 (Revised Milestone 2.1)  
**Lead Evaluator:** Senior Machine Learning Researcher & Technical Supervisor  
**Evaluation Target:** Full Technical Reproducibility Protocol & Model Hashes  
**Status:** FULLY REPRODUCIBLE WITH HONEST SCIENTIFIC CLASSIFICATION  

---

## 1. System & Execution Environment

Every experiment in Milestone 2.1 was conducted on the following exact hardware and software environment:

| Property | Value |
| :--- | :--- |
| **Operating System** | Microsoft Windows 11 Pro 64-bit (Build 26200) |
| **Host CPU** | AMD Ryzen 7 5800X (8 Cores, 16 Threads @ 3.80 GHz) |
| **Host RAM** | 32.0 GB DDR4 |
| **GPU Hardware** | NVIDIA GeForce RTX 4060 Ti (16 GB GDDR6 VRAM) |
| **NVIDIA Driver** | 576.88 (CUDA 12.4 compatible) |
| **Python Virtualenv** | `D:\MZHR_FYP\INTELLIGATE\.venv` (Python 3.12.10) |
| **PyTorch Release** | `2.6.0+cu124` |
| **OpenCV Release** | `5.0.0` |
| **Ultralytics Release** | `8.3.26` |
| **EasyOCR Release** | `1.7.2` |

---

## 2. Supply-Chain Model Weights & Hash Manifest

All models run offline with zero runtime CDN downloading:

| Model Filename | Local Filepath | Exact SHA-256 Checksum | Purpose |
| :--- | :--- | :--- | :--- |
| `yolov8n.pt` | `D:\MZHR_FYP\INTELLIGATE\yolov8n.pt` | `f59b3d833e2ff32e705b65ffcb3a7fae51ab54ad849505cead633d6a6a90a2a5` | Vehicle & Scene Detection |
| `face_detection_yunet_2023mar.onnx` | `D:\MZHR_FYP\INTELLIGATE\models\face_detection_yunet_2023mar.onnx` | `8f2383e4dd3cfbb4b6a8da6df993bfa59a4cbe6587c4bf7fbeea4c94b7911685` | YuNet Face Localization |
| `face_recognition_sface_2021dec.onnx` | `D:\MZHR_FYP\INTELLIGATE\models\face_recognition_sface_2021dec.onnx` | `0ba9fbfa01b5270ce79db28b9fb6b62719d27ae2e8f80ee0e9fb509426f041ff` | SFace 128-D Embeddings |

---

## 3. Benchmark Categorization & Provenance Matrix

Every benchmark artifact generated is categorized into its true scientific classification:

| Benchmark Artifact | Script Source | Scientific Classification | Input Data Provenance | Status |
| :--- | :--- | :--- | :--- | :--- |
| `m02_1_fair_cpu_gpu_benchmark.json` | [`benchmark_fair_cpu_gpu.py`](file:///D:/MZHR_FYP/INTELLIGATE/tools/benchmark_fair_cpu_gpu.py) | **Paired System Benchmark** | Real demo video frames (`llvip_visible.mp4`) | **VALIDATED** |
| `m02_1_accuracy_parity_audit.json` | [`benchmark_fair_cpu_gpu.py`](file:///D:/MZHR_FYP/INTELLIGATE/tools/benchmark_fair_cpu_gpu.py) | **Accuracy Parity Audit** | Real demo video frames & test plate crops | **VALIDATED** |
| `plate_ocr_benchmark.json` | [`eval_plate_ocr.py`](file:///D:/MZHR_FYP/INTELLIGATE/experiments/eval_plate_ocr.py) | **Synthetic Procedural Microbenchmark** | 20 procedurally rendered font images | **SYNTHETIC** |
| `face_verification_benchmark.json` | [`eval_lfw_sface.py`](file:///D:/MZHR_FYP/INTELLIGATE/experiments/eval_lfw_sface.py) | **Synthetic Diagnostic Benchmark** | 1 reference face + 300 random vectors | **SYNTHETIC** |
| *Real LFW Evaluation* | — | **Real Held-Out Protocol** | Official LFW 6,000 pairs partition | **BLOCKED** |
| *Real Pakistani ALPR Evaluation* | — | **Real Road Camera Evaluation** | Real vehicle surveillance video / images | **BLOCKED** |

---

## 4. Step-by-Step Reproduction Commands

To independently reproduce the Milestone 2.1 benchmarks, execute the following commands from `D:\MZHR_FYP\INTELLIGATE`:

### Step 1: Execute Fair Paired CPU vs GPU Workload & Parity Audit
```powershell
& "D:\MZHR_FYP\INTELLIGATE\.venv\Scripts\python.exe" tools/benchmark_fair_cpu_gpu.py
```
*Outputs:*
- `logs/benchmarks/m02_1_fair_cpu_gpu_benchmark.json`
- `logs/benchmarks/m02_1_accuracy_parity_audit.json`

### Step 2: Execute Synthetic Diagnostic Face Verification Check
```powershell
& "D:\MZHR_FYP\INTELLIGATE\.venv\Scripts\python.exe" experiments/eval_lfw_sface.py
```
*Output:* `logs/benchmarks/face_verification_benchmark.json`

### Step 3: Execute Synthetic Procedural Plate OCR Microbenchmark
```powershell
& "D:\MZHR_FYP\INTELLIGATE\.venv\Scripts\python.exe" experiments/eval_plate_ocr.py
```
*Output:* `logs/benchmarks/plate_ocr_benchmark.json`

### Step 4: Run Full Regression Test Suite
```powershell
& "D:\MZHR_FYP\INTELLIGATE\.venv\Scripts\python.exe" -m pytest
```
*Expected Result:* 57 passed in ~98s.
