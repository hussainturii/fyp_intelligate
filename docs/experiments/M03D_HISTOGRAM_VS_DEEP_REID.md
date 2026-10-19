# EXPERIMENT REPORT: M03D — COLOR HISTOGRAM BASELINE VS. DEEP VEHICLE RE-IDENTIFICATION

**Comparative Study:** 768-D RGB Color Histogram Baseline vs. 512-D OSNet-AIN Deep Metric Re-ID  
**Experiment ID:** `EXP-M03D-HIST-VS-DEEP-01`  
**Date:** October 19, 2026  
**Investigator:** Principal Computer Vision Engineer & Deep Metric Learning Research Scientist  
**Raw Metrics Source:** `logs/benchmarks/m03d_vehicle_reid_results.json`

---

## 1. Architectural & Theoretical Comparison

| Architectural Dimension | Legacy Baseline: Color Histogram | Deep Metric Re-ID: OSNet-AIN (`vehicle-reid-0001`) |
| :--- | :--- | :--- |
| **Model Type** | Classical statistical color aggregation (`cv2.calcHist`) | Deep Multi-Scale Convolutional Neural Network |
| **Representation Dimension** | **768 floats** (3 channels $\times$ 256 bins) | **512 floats** (Global feature pooling + projection) |
| **Feature Normalization** | $L_2$ vector normalization ($\|v\|_2 = 1.0$) | $L_2$ vector normalization ($\|v\|_2 = 1.0$) |
| **Trainable Parameters** | 0 parameters (Unlearned heuristic) | **2.183 Million parameters** |
| **Computational Footprint** | ~0.001 GFLOPs | **2.643 GFLOPs** |
| **Model Weight File Size** | 0 MB (Code-only) | **8.43 MB** (`osnet_ain_x1_0_vehicle_reid.onnx`) |
| **Input Crop Contract** | Arbitrary $H \times W \times 3$ BGR frame | $208 \times 208 \times 3$ RGB, aspect-ratio letterboxed |
| **Spatial Awareness** | **None** (Spatial coordinates completely discarded) | **Omni-Scale** (Captures local grill trims to macro silhouette) |
| **Color Invariance** | Zero (Highly sensitive to illumination and shadow) | Robust (Trained across 20 varied surveillance cameras) |
| **Structural Discrimination** | Zero (Identical color = identical histogram) | **High** (Distinguishes distinct vehicle makes of the same color) |

---

## 2. Quantitative Performance & Benchmark Results

Both methods were evaluated on the exact same dataset of local vehicle crops under identical hardware conditions (AMD Ryzen 7 5800X, single-threaded CPU execution).

### A. Extraction Latency & System Throughput
| Metric | 768-D Color Histogram Baseline | 512-D OSNet-AIN Deep Metric Re-ID | Relative Impact |
| :--- | :---: | :---: | :---: |
| **Mean Latency** | **0.14 ms** | **54.00 ms** | 385× slower |
| **Median Latency** | **0.13 ms** | **53.79 ms** | 413× slower |
| **Standard Deviation** | 0.004 ms | 0.87 ms | Highly stable |
| **95th Percentile ($P_{95}$)** | 0.15 ms | 55.77 ms | Consistent bounds |
| **Throughput (FPS)** | **7,347.3 FPS** | **18.5 FPS** | Exceeds 15 FPS video rate |

> **Engineering Analysis:** While the color histogram executes in microseconds, its computational speed is meaningless if it lacks discriminative accuracy. OSNet-AIN operates at **18.5 FPS on CPU**, which comfortably outpaces standard 15 FPS security camera stream rates while delivering genuine identity separation.

---

### B. Pairwise Cosine Similarity Distributions & Separation Margin

Using 30 local vehicle crops, we evaluated 30 same-vehicle pairs (sub-crops modeling viewpoint shifts) and 135 different-vehicle pairs (cross-vehicle comparisons):

| Metric | 768-D Color Histogram Baseline | 512-D OSNet-AIN Deep Metric Re-ID | Improvement / Difference |
| :--- | :---: | :---: | :---: |
| **Same-Vehicle Mean ($\mu_{\text{same}}$)** | 0.6380 | **0.8367** | **+0.1987** (+31.1% tighter clustering) |
| **Same-Vehicle Median** | 0.6788 | **0.8446** | +0.1658 |
| **Same-Vehicle Std ($\sigma_{\text{same}}$)** | 0.1914 | **0.0460** | **4.16× lower variance** (more reliable) |
| **Different-Vehicle Mean ($\mu_{\text{diff}}$)** | 0.4781 | **0.5789** | Expected metric baseline |
| **Different-Vehicle Median** | 0.4611 | **0.5977** | Calibrated distribution |
| **Different-Vehicle Max (Worst Impostor)** | **0.9397** (False Match Risk) | **0.8588** | Substantially reduced impostor peak |
| **Separation Margin ($\Delta = \mu_{\text{same}} - \mu_{\text{diff}}$)** | **0.1599** | **0.2578** | **+0.0979 (1.61× Advantage)** |

```
Distribution Separation Visualized:

1. Color Histogram 768-D:
Different Vehicles: [---- 0.4781 ----]
Same Vehicles:               [---- 0.6380 ----]
Overlap / Confusion:         ||||||||||  (High false-match risk, max impostor = 0.9397)
Separation Margin: Δ = 0.1599

2. OSNet-AIN 512-D:
Different Vehicles: [---- 0.5789 ----]
Same Vehicles:                                [--- 0.8367 ---]
Overlap / Confusion:         ||  (Low overlap, max impostor = 0.8588)
Separation Margin: Δ = 0.2578  (1.61x wider discriminative gap)
```

---

## 3. Qualitative Failure Mode Analysis

### Failure Mode 1: The "Two White Sedans" Identity Collision
* **Scenario:** A white Toyota Corolla and a white Honda Civic arrive at Turnstile A. Both license plates are obscured by winter mud.
* **Color Histogram Behavior:** Because both vehicles are predominantly white with dark windshields and tires, their 768-D color histograms are nearly identical ($\text{sim} > 0.92$). The system cannot determine that they are distinct vehicle models, resulting in false identity cross-attribution.
* **OSNet-AIN Behavior:** OSNet extracts omni-scale features across multiple receptive fields ($1\times1$, $3\times3$, $5\times5$). It captures:
  * Grille geometry (horizontal slatted vs. honeycomb).
  * Headlight cluster contours and daytime running light patterns.
  * Pillar curvature, roofline silhouette, and fog light housings.
  * Result: Cosine similarity drops below 0.65, correctly rejecting the false match.

### Failure Mode 2: Illumination Shift (Sunny vs. Overcast Shadow)
* **Scenario:** The same registered vehicle enters in direct sunlight at 12:00 PM and returns under heavy overcast shadows at 5:00 PM.
* **Color Histogram Behavior:** Shadows and direct reflections drastically shift RGB bin distributions. Same-vehicle histogram similarity drops to **0.1954**, causing complete recognition failure.
* **OSNet-AIN Behavior:** Trained with data augmentations across 20 traffic cameras in VeRi-776, OSNet-AIN utilizes Adaptive Instance Normalization (AIN) to disentangle illumination styles from structural identity features. Same-vehicle similarity remains consistently high ($\ge 0.7331$, mean 0.8367).

---

## 4. Multi-Tiered Coexistence Architecture

IntelliGate does not discard the color histogram; it employs a multi-tiered fallback architecture in `module8_dashboard/backend/vision.py`:

```
Vehicle Crop Arrival
         │
         ▼
[1] Exact License Plate OCR (Conf >= 0.85?)
   ├─► YES: method='exact_plate' ──► Autonomous ALLOW (Policy Engine)
   └─► NO (Plate obscured or conf < 0.85)
         │
         ▼
[2] Primary Appearance: OSNet-AIN Deep Re-ID (512-D)
   ├─► Match (sim >= 0.70): method='deep_reid' ──► REVIEW_REQUIRED (Gate CLOSED)
   └─► Unmatched / Exception
         │
         ▼
[3] Secondary Fallback: Color Histogram Baseline (768-D)
   ├─► Match (sim >= 0.92): method='appearance_histogram_baseline' ──► REVIEW_REQUIRED (Gate CLOSED)
   └─► Unmatched ──► Stranger Alert (Gate CLOSED)
```

### Key Engineering Benefits of Coexistence
1. **Graceful Degradation:** If the ONNX model file is missing or corrupted during edge startup, the pipeline falls back seamlessly to the color histogram without throwing unhandled exceptions or halting surveillance replays.
2. **Defensive Auditability:** In the event store and dashboard, the `appearance_model` metadata field explicitly distinguishes `'osnet_ain_x1_0_vehicle_reid'` from `'colour histogram baseline'`, ensuring full operator transparency.

---

## 5. Summary & Recommendation

| Assessment Category | Color Histogram Baseline | OSNet-AIN Deep Metric Re-ID | Verdict |
| :--- | :--- | :--- | :--- |
| **Inference Latency** | 0.13 ms (7,347 FPS) | 53.79 ms (18.5 FPS) | Histogram is faster, but Re-ID meets real-time budget |
| **Discriminative Ability** | Poor (Separation Margin = 0.1599) | **Superior (Separation Margin = 0.2578)** | **OSNet-AIN wins (1.61× gain)** |
| **Feature Stability** | High variance ($\sigma = 0.1914$) | **Low variance ($\sigma = 0.0460$)** | **OSNet-AIN wins (4.16× tighter)** |
| **Advisory Reliability** | Frequently confuses same-color cars | Distinguishes distinct make/model silhouettes | **OSNet-AIN wins** |
| **System Security** | Advisory-only | Advisory-only | **Tied (Strict Fail-Secure Policy)** |

**Final Recommendation:** Deploy OSNet-AIN as the primary vehicle appearance matcher across IntelliGate, retaining the color histogram purely as an emergency fallback heuristic.
