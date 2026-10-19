# INTELLIGATE — MILESTONE 3C.1 EXPERIMENT REPORT
## Controlled OCR Preprocessing Ablation on Pakistani License Plates

**Author:** Hussain Raza (SP23-BAI-057)  
**Academic Role:** BS Artificial Intelligence FYP Lead Researcher  
**Dataset Reference:** 50 Human-Verified Pakistani Plate Crops (Development Set)  
**Experimental Status:** Controlled Exploratory Ablation (Development Set; Not Independent Test Set)  
**Machine-Readable Results:** [`logs/benchmarks/m03c_1_ocr_preprocessing_ablation.json`](file:///d:/MZHR_FYP/INTELLIGATE/logs/benchmarks/m03c_1_ocr_preprocessing_ablation.json)

---

## 1. Experimental Integrity & Methodology Declaration

> [!IMPORTANT]
> **Experimental Integrity & Set Partitioning:**  
> The 50 verified Pakistani plate crops evaluated in this experiment were previously inspected during error analysis and failure identification. Therefore, any preprocessing variant evaluated on these 50 samples represents **development-set tuning / exploratory ablation**, NOT an independent held-out test. In future milestones, an untouched validation set will be required for final generalization claims.

Six controlled preprocessing and structural reassembly variants were benchmarked under identical conditions using EasyOCR with GPU acceleration on an NVIDIA RTX 4060 Ti:
1. **Variant A (Baseline):** Raw crop as provided by the plate detector.
2. **Variant B (Border Padding):** Adds 12% symmetric margin padding using `cv2.BORDER_REPLICATE` to prevent character truncation on tight crops.
3. **Variant C (CLAHE Contrast Enhancement):** Applies Contrast Limited Adaptive Histogram Equalization ($clipLimit=2.5, tileGrid=(8,8)$) to the $L$-channel in LAB color space.
4. **Variant D (Multi-Line Reading-Order Sort):** Clusters text boxes into vertical lines using centroid $y$-coordinates and bounding box heights, sorts lines top-to-bottom, and sorts text boxes left-to-right within each line.
5. **Variant E (Padding + Multi-Line Sort):** Combines 12% border padding with multi-line reading-order sorting.
6. **Variant F (Combined All):** Combines border padding, CLAHE contrast enhancement, and multi-line reading-order sorting.

---

## 2. Quantitative Ablation Results

### Table 1: Preprocessing Ablation Performance Comparison ($N = 50$)

| Variant ID | Description | Exact Matches | Exact Acc (%) | CER (%) | Total Edits | Latency (ms) | Delta (+ / -) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **A_Baseline** | Raw Crop Baseline | 25 / 50 | 50.00% | 20.72% | 75 | 38.24 ms | Baseline |
| **B_Padding** | Border Padding (+12%) | 25 / 50 | 50.00% | 22.10% | 80 | 27.19 ms | +3 / -3 |
| **C_CLAHE** | Contrast Enhancement (CLAHE) | 25 / 50 | 50.00% | 25.97% | 94 | 22.26 ms | +3 / -3 |
| **D_MultiLine** | Multi-Line Reading-Order Sort | **26 / 50** | **52.00%** | **19.06%** | **69** | **23.35 ms** | **+1 / -0** |
| **E_Pad_MultiLine** | Padding + Multi-Line Sort | 26 / 50 | 52.00% | 20.17% | 73 | 23.35 ms | +3 / -2 |
| **F_Combined_All** | Padding + CLAHE + Multi-Line | **28 / 50** | **56.00%** | 21.55% | 78 | 23.43 ms | +5 / -2 |

---

## 3. Scientific Findings & Qualitative Analysis

### 3.1 Variant D (Multi-Line Reading-Order Sort) — The Cleanest Improvement
- **Accuracy:** Improved exact matches from 25 to 26 (+2.00%).
- **Error Rate:** Reduced CER from 20.72% to **19.06%** (a 1.66% absolute reduction; 6 fewer character errors).
- **Degradation:** **0 degraded samples**! Variant D corrected Sample ID 21 (`FSA-5750`, which baseline read inverted as `5750FSA`) without degrading any other plate.
- **Scientific Rationale:** Unlike spatial filtering or pixel transformations that can distort character strokes, multi-line clustering respects the physical two-tier layout of Pakistani license plates, eliminating reading-order permutations without risking stroke distortion.

### 3.2 Variant B (Border Padding) — Mixed Effects
- **Corrected Samples (+3):**
  - Sample 3: `LEE16P9091` $\to$ `LEE169091` (border padding separated the character stroke from the border edge).
  - Sample 21: `5750FSA` $\to$ `FSA5750`.
  - Sample 35: `JV38430` $\to$ `JV3843` (eliminated spurious zero caused by crop edge).
- **Degraded Samples (-3):**
  - Sample 1: `LEF123503` $\to$ `LEF3503` (padding caused small year code `12` to be grouped into background).
  - Sample 37: `UZ016` $\to$ `WZ016`.
  - Sample 46: `LED9812` $\to$ `9812LED`.
- **Verdict:** Replicate padding helps border-clipped characters but introduces surrounding bumper artifacts into the text detection window.

### 3.3 Variant C (CLAHE) — Adverse Side Effects
- **Accuracy:** Unchanged at 50.00%, but **CER worsened by 5.25%** (from 20.72% to 25.97%, total edits increased from 75 to 94).
- **Degraded Samples:** Sample 44 (`RL-3690`) was read correctly by baseline, but CLAHE amplified specular glare and surface texture to the point where EasyOCR produced an empty read!
- **Verdict:** CLAHE is not universally beneficial for license plates. While it enhances contrast in low-light environments, on Pakistani metal stamped plates it amplifies metal grain, scratches, and bolt holes, creating spurious edges that distract CRAFT text detection.

### 3.4 Why Perspective Correction Was Not Blindly Applied
Perspective rectification via planar homography ($H \in \mathbb{R}^{3\times 3}$) requires 4 corresponding corner coordinates:
$$x' = \frac{h_{11}x + h_{12}y + h_{13}}{h_{31}x + h_{32}y + h_{33}}, \quad y' = \frac{h_{21}x + h_{22}y + h_{23}}{h_{31}x + h_{32}y + h_{33}}$$
Because the crops provided in the dataset are tight axis-aligned bounding boxes without 4 manually annotated polygon corner vertices, attempting to automatically extract quadrilateral corners using OpenCV contour approximation (`cv2.approxPolyDP`) frequently snaps to plate text, license plate frames, or vehicle bumpers. Homography on flawed corners introduces severe keystone distortion, which destroys character legibility. Perspective correction is therefore mathematically ill-posed on axis-aligned crops and must only be deployed when 4-corner keypoints are explicitly localized by a specialized keypoint detector.

---

## 4. Engineering Recommendations for Milestone 3D

1. **Deploy Multi-Line Reading-Order Sort:** Integrate vertical band clustering and horizontal sorting into `PlateOCR` because it achieves +1/-0 sample delta and improves CER without degradation risk.
2. **Do Not Synthesize Missing Characters:** Even though missing 2-digit registration years account for 65.7% of failures, the system must **never hallucinate or invent missing characters** to improve OCR scores. Gate access security demands exact registry matching.
3. **Retain Dual-Tier Verification:** High-confidence plates ($conf \ge 0.85$) with exact registry matches may be automated; all ambiguous, partial, or multi-line un-read plates must remain routed to `REVIEW_REQUIRED`.
