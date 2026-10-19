# INTELLIGATE — MILESTONE 3D: DEEP VEHICLE RE-IDENTIFICATION IMPLEMENTATION REPORT

**Author:** Principal Computer Vision Engineer, Deep Metric Learning Research Scientist & Security-Focused Architect  
**Project Root:** `D:\MZHR_FYP\INTELLIGATE`  
**Milestone:** 3D — Deep Vehicle Re-Identification  
**Date:** October 19, 2026  
**Status:** Completed & Fully Verified (109/109 Backend Tests Passed, Production Frontend Build Clean)

---

## 1. Executive Summary & Progression Context

Milestone 3D investigates, implements, and scientifically benchmarks a genuine deep-learning vehicle Re-Identification (Re-ID) component to elevate IntelliGate's vehicle appearance verification beyond the legacy 768-dimensional RGB color histogram heuristic.

Prior to Milestone 3D:
1. `module8_dashboard/backend/vision.py` relied solely on `appearance(frame)`: a 768-D concatenated RGB color histogram (`cv2.calcHist`).
2. `module2_vehicle/vehicle_reid.py` contained an un-fine-tuned ImageNet Vision Transformer stub (`ViTVehicleEmbedder`), referencing an uninstalled library (`timm`) with zero vehicle-domain metric learning training.
3. No vehicle-domain metric checkpoints or multi-camera vehicle tracking benchmarks existed locally.

Following strict Phase 1 feasibility audit and checkpoint verification, IntelliGate has integrated the verified **Intel Open Model Zoo `vehicle-reid-0001`** model (`osnet_ain_x1_0_vehicle_reid.onnx`).

### Milestone Completion Taxonomy
To maintain strict scientific integrity, completion states are explicitly separated:
* **Model Acquired:** `models/osnet_ain_x1_0_vehicle_reid.onnx` (8,836,743 bytes) downloaded from official Intel Open Model Zoo storage and registered in `models/trusted_manifest.json` with verified SHA-384 and SHA-256 hashes.
* **Model Integrated:** Modular `OSNetVehicleReID` (and backward-compatible `VehicleReID`) feature extractor implemented in `module2_vehicle/vehicle_reid.py`. Integrated into `ReplayPipeline` (`vision.py`) and `VehicleDatabase` (`vehicle_db.py`).
* **Model Functionally Tested:** Complete 17-test suite (`tests/test_milestone3d_vehicle_reid.py`) validating shape contracts, determinism, aspect-ratio letterboxing, error handling, FAISS indexing, and gate interlocks. All 109 repository backend tests pass.
* **Model Scientifically Evaluated:** Benchmark suite (`experiments/eval_m03d_vehicle_reid.py`) executed on CPU. Official VeRi-776 benchmarks documented (Rank-1: 96.31%, mAP: 85.15%). Local multi-camera traffic correctly labeled as **NOT EVALUATED (Requires multi-camera benchmark dataset)**. Real local vehicle crop pairwise separation evaluated (1.61× discriminative margin advantage over color histograms).

---

## 2. Model Architecture, Provenance & Checksum Verification

```
Official Intel Open Model Zoo Repository:
https://github.com/openvinotoolkit/open_model_zoo/tree/master/models/public/vehicle-reid-0001

Model Checkpoint:
osnet_ain_x1_0_vehicle_reid.onnx

Model Source URL:
https://storage.openvinotoolkit.org/repositories/open_model_zoo/public/2022.1/vehicle-reid-0001/osnet_ain_x1_0_vehicle_reid.onnx
```

### Checkpoint Provenance & Manifest Verification
| Attribute | Manifest Value / Published Specification | Verified Local Measurement | Status |
| :--- | :--- | :--- | :---: |
| **Model Filename** | `osnet_ain_x1_0_vehicle_reid.onnx` | `models/osnet_ain_x1_0_vehicle_reid.onnx` | MATCH |
| **File Size** | 8,836,743 bytes | 8,836,743 bytes | MATCH |
| **SHA-384** | `0515ce72f653c39780d5b87dfed7255d396dd2b1e8b6e91fbaacdfad1da189166343157273c02f3b0fede3050ef7abb7` | `0515ce72f653c39780d5b87dfed7255d396dd2b1e8b6e91fbaacdfad1da189166343157273c02f3b0fede3050ef7abb7` | MATCH |
| **SHA-256** | `4aaad3e5db648618b0df3d2ff21c61323985ff9e50194c3d2edd4fb87c92d91f` | `4aaad3e5db648618b0df3d2ff21c61323985ff9e50194c3d2edd4fb87c92d91f` | MATCH |
| **License** | MIT License (Copyright (c) 2018 Kaiyang Zhou / Intel Corp.) | Registered in `models/trusted_manifest.json` | MATCH |
| **Base Architecture** | Omni-Scale Network + Adaptive Instance Norm (OSNet-AIN) | 2.183M parameters / 2.643 GFLOPs | MATCH |
| **Embedding Vector** | 512-dimensional float32, unit $L_2$ normalized | `shape: (512,)`, $\|v\|_2 = 1.00000$ | MATCH |
| **Training Dataset** | VeRi-776 (576 training vehicle IDs, 37,778 multi-view images) | Genuine vehicle domain (not person Re-ID) | MATCH |

---

## 3. Preprocessing & Geometric Invariants

The model expects whole vehicle body crops. Resizing rectangular vehicle crops directly to a square causes severe distortion of grill ratios and body proportions.

### Aspect-Ratio Preserving Letterbox Pipeline
1. **Input Validation:** Rejects `None`, empty, zero-sized, or sub-5×5 crops with explicit `ValueError`.
2. **Color Space Standardization:** Accepts NumPy BGR/BGRA/Grayscale or PIL Image; converts cleanly to RGB.
3. **Proportional Scaling:** Computes uniform scale factor $s = \min(208/w, 208/h)$ and resizes using bilinear interpolation (`cv2.INTER_LINEAR`).
4. **Symmetric Border Padding:** Pads the shorter dimension symmetrically to exactly $208 \times 208$ using constant neutral-gray padding (`pad_color = (124, 116, 104)`), matching ImageNet mean color.
5. **Channel Normalization:**
   $$x_{\text{norm}} = \frac{x_{\text{RGB}} - \mu_{\text{RGB}}}{\sigma_{\text{RGB}}}$$
   Where $\mu = [123.675, 116.28, 103.53]$ and $\sigma = [58.395, 57.12, 57.375]$.
6. **Layout & Tensor Transpose:** Formats as NCHW tensor `[1, 3, 208, 208]` float32.

---

## 4. Architecture & Component Design

```mermaid
flowchart TD
    A[Vehicle Frame / YOLO Detection] --> B[Vehicle Crop Extraction]
    B --> C{Crop Valid?}
    C -->|No| D[Reject / Return Safe Fallback]
    C -->|Yes| E[Letterbox Aspect Padding to 208x208]
    E --> F[RGB ImageNet Normalization]
    F --> G[OSNet-AIN ONNX Graph Execution]
    G --> H[Raw 512-D Feature Tensor]
    H --> I[L2 Normalization: v / ||v||2]
    I --> J[512-D Unit Embedding]
    J --> K[VehicleDatabase FAISS Search]
    K --> L{Similarity >= Threshold?}
    L -->|No| M[Advisory Stranger / Unmatched]
    L -->|Yes| N[Advisory Appearance Match]
    N --> O[SoftwareRuntime Decision Policy]
    O --> P[REVIEW_REQUIRED: Gate Remains CLOSED]
```

### Component Details
* **`module2_vehicle/vehicle_reid.py`:**
  * Defines `OSNetVehicleReID` (and `VehicleReID` alias).
  * Implements `extract_embedding(crop)` $\to 512\text{-D}$ float32 normalized vector.
  * Preserves `extract_histogram_baseline(crop)` $\to 768\text{-D}$ RGB histogram vector for baseline comparison.
  * Implements `compare(emb1, emb2)` returning cosine similarity and distance.
  * Enforces startup model integrity check against `models/trusted_manifest.json`.
* **`module2_vehicle/database/vehicle_db.py`:**
  * Default embedding dimension updated from 768 to 512.
  * Added resilient dynamic dimension handling: detects query embedding dimension and rejects mismatches without crashing FAISS.
  * `_rebuild_auth_index()` dynamically adapts to stored embedding dimensions.
* **`module8_dashboard/backend/vision.py`:**
  * `ReplayPipeline` instantiates `VehicleReID` once upon initialization.
  * In `_vehicle(crop, track_id)`:
    * Authoritative evaluation: license plate detection + EasyOCR. If registered plate matched with $conf \ge 0.85$, `method = 'exact_plate'`.
    * If plate OCR fails or confidence $< 0.85$: extracts 512-D OSNet embedding and searches `vehicle_db`.
    * If deep Re-ID matches ($\text{sim} \ge 0.70$): `method = 'deep_reid'`, `appearance_model = 'osnet_ain_x1_0_vehicle_reid'`, `embedding_dim = 512`.
    * If deep Re-ID does not match: evaluates legacy color histogram fallback.
    * Both appearance methods are strictly advisory (`advisory_match: True`).

---

## 5. Security & Access Decision Invariants

A central requirement of IntelliGate's access policy is that **appearance matching is strictly advisory and can never independently open the physical gate**:

1. **Gate Authorization Hierarchy:**
   * **AUTONOMOUS ALLOW:** Strictly requires an exact registered license plate (`method == 'exact_plate'`) with OCR confidence $\ge 0.85$ and trusted perception provenance (`provenance == 'perception_internal'`).
   * **ADVISORY CANDIDATE (`REVIEW_REQUIRED`):** Deep vehicle Re-ID appearance match (`method == 'deep_reid'`) or color histogram match (`method == 'appearance_histogram_baseline'`) routes strictly to operator review. The gate remains locked (`CLOSED`).
   * **SAFETY OVERRIDE (`DENY`):** Safety beam obstruction or emergency lockdown immediately forces `decision = 'DENY'`, overriding any valid plate or appearance match.
2. **Untrusted Provenance Protection:** External or spoofed events without trusted internal perception metadata fail closed into `REVIEW_REQUIRED`.

---

## 6. Execution Runtime & Hardware Benchmark Summary

* **Execution Runtime:** OpenCV DNN (`cv2.dnn.readNetFromONNX`).
* **Platform:** AMD Ryzen 7 5800X 8-Core Processor, 32 GB RAM, Windows 11 Enterprise.
* **Hardware Observation:** Under OpenCV 5.0.0.93 on Windows, `cv2.dnn` executes on the CPU graph engine (`DNN_TARGET_CPU`), providing rock-solid numerical determinism across runs without driver crashes.
* **Extraction Latency (100 iterations, 10 warmup):**
  * **Mean Latency:** 54.00 ms
  * **Median Latency:** 53.79 ms (18.5 FPS)
  * **95th Percentile:** 55.77 ms
  * **Standard Deviation:** 0.87 ms
* **FAISS Search Latency:**
  * 1 registered vehicle: 0.0069 ms (~139,000 queries/sec)
  * 100 registered vehicles: 0.0160 ms (~59,000 queries/sec)
  * 500 registered vehicles: 0.0300 ms (~33,000 queries/sec)

---

## 7. Verification & Regression Status

* **Dedicated Milestone 3D Tests (`tests/test_milestone3d_vehicle_reid.py`):** 17/17 passed in 9.57s.
* **Full Backend Regression Suite:** **109 passed**, 0 failed across all test modules (increased from 92 passed in Milestone 3C.2).
* **Frontend Production Build:** Vite build succeeded in 14.20s (`dist/index.html` 0.76 kB, `dist/assets/index-XMCyO23g.js` 625 kB).
* **Asset Preservation:** Zero frontend screenshots or prior evaluation assets overwritten.
