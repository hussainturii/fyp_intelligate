# INTELLIGATE — EXPERIMENT REPORT: M02 PAKISTANI LICENSE PLATE OCR EVALUATION
**Document ID:** `IG-EXP-M02-PLATE-2026-10-REV1`  
**Date:** October 18, 2026 (Revised Milestone 2.1)  
**Lead Researcher / Evaluator:** Senior Machine Learning Researcher & CV Evaluation Lead  
**Evaluation Scope:** Synthetic Procedural License Plate OCR Microbenchmark  
**Evaluation Classification:** **SYNTHETIC PROCEDURAL BENCHMARK** (Procedurally generated OpenCV font renderings)  
**Operational Status:** **BLOCKED ON AUTHORIZED REAL PAKISTANI ROAD DATASET ACQUISITION**  

---

## 1. Executive Summary & Critical Errata Notice

> [!CAUTION]
> **Scientific Integrity Errata (Milestone 2.1 Correction):**  
> Earlier revisions of this report presented results as an evaluation of "Real Pakistani Vehicle License Plates" with "65% Exact Match Accuracy".  
> **These claims were inaccurate and are hereby formally corrected.**  
> Inspection of [eval_plate_ocr.py](file:///D:/MZHR_FYP/INTELLIGATE/experiments/eval_plate_ocr.py) confirms that all 20 test samples were **procedurally rendered synthetic images** generated using `cv2.putText()` with synthetic white backgrounds and artificial Gaussian noise.  
> 
> No real camera captures of Pakistani vehicles were utilized in this experiment. The results represent a **synthetic character recognition microbenchmark** under idealized raster typography, NOT empirical real-world camera accuracy.

---

## 2. Test Sample Provenance & Classification

Every sample in the existing evaluation was rendered procedurally in-memory:

| Sample Group | Formats Evaluated | Rendering Method | Provenance Classification |
| :--- | :--- | :--- | :--- |
| **ICT (5 Samples)** | `ICT-1234`, `ISB-5678`, `ICT-901`, `ISB-2468`, `ICT-3579` | OpenCV `FONT_HERSHEY_SIMPLEX` on synthetic $260 \times 80$ canvas | **100% Synthetic Procedural** |
| **Punjab (7 Samples)** | `LEA-1234`, `LXZ-5678`, `RIL-3456`, `FD-7890`, `MN-1122`, `LEA-9876`, `LXZ-1357` | OpenCV `FONT_HERSHEY_SIMPLEX` on synthetic canvas ($\pm$ noise) | **100% Synthetic Procedural** |
| **Sindh (5 Samples)** | `ABC-123`, `KHI-4567`, `AZ-890`, `KHI-8910`, `SND-1234` | OpenCV `FONT_HERSHEY_SIMPLEX` on synthetic canvas ($\pm$ noise) | **100% Synthetic Procedural** |
| **KPK (3 Samples)** | `KPK-1234`, `PES-5678`, `KPK-9012` | OpenCV `FONT_HERSHEY_SIMPLEX` on synthetic canvas ($\pm$ noise) | **100% Synthetic Procedural** |

---

## 3. Synthetic Microbenchmark Results

### 3.1 Measured Metrics (Synthetic Cropped Plates Only)
- **Sample Count:** 20 procedural plate crops.
- **Exact String Match Accuracy:** **65.0%** (13 / 20 correct).
- **Corpus-Level Character Error Rate (CER):** **15.52%** (Normalized Levenshtein edit distance).
- **Mean Character Accuracy:** **84.48%** ($1 - \text{CER}$).
- **Inference Speed:**
  - **GPU (RTX 4060 Ti):** **23.88 ms** median
  - **CPU (AMD Ryzen 7 5800X):** **149.95 ms** median
  - **Speedup Ratio:** **6.28x**

### 3.2 Distinction Between Cropped OCR vs Complete End-to-End ALPR
The measured 65.0% accuracy applies exclusively to **pre-cropped, upright, unskewed plate images**. In a real deployed automatic license plate recognition (ALPR) system:
$$\text{Accuracy}_{\text{ALPR}} = \text{Accuracy}_{\text{vehicle\_det}} \times \text{Accuracy}_{\text{plate\_loc}} \times \text{Accuracy}_{\text{plate\_ocr}}$$
Because plate localization failures, perspective distortions, motion blurs, and headlight glare compound along the perception pipeline, true end-to-end ALPR accuracy on real vehicles will be strictly lower than isolated cropped OCR accuracy.

---

## 4. Security Analysis of False Matches & Access Control

Unlike facial recognition (which is restricted to human guard advisory review), **an exact plate match on an authorized vehicle is configured to automatically actuate the gate barrier (`ACCESS_DECISION: ALLOW`)**.

This creates significant security implications:

### 4.1 False Positive vs False Negative Risks
1. **False Negative (Rejecting an Authorized Vehicle):**
   - If an authorized plate `LEA-1234` is misread as `LEA-123A`, the system fails closed to `REVIEW_REQUIRED`. The barrier remains closed, requiring manual security intervention. This causes minor operational delay but poses zero physical security breach.
2. **False Positive (Impersonating an Authorized Vehicle):**
   - If an unauthorized vehicle `LEA-1238` is misread as registered plate `LEA-1230` due to a digit confusion error ($8 \to 0$), the gate barrier would autonomously open.
   - **Mitigation Requirement:** Under no circumstances should fuzzy matching or partial string matching be permitted for automated gate actuation. **Exact alphanumeric matching must strictly remain mandatory** (`lookup_by_plate` requiring exact string equality).

---

## 5. Protocol for Future Real-World Pakistani ALPR Evaluation

When authorization and budget for real dataset acquisition are granted, the evaluation must follow this protocol:

1. **Curate Real Roadside Video/Imagery:**
   - 100+ annotated real high-definition camera captures from security surveillance gates across day, dusk, night, and rain conditions.
   - Covering official Sindh yellow embossed plates, Punjab laser-printed security plates, Islamabad blue-and-white plates, and private custom fonts.
2. **Ground Truth Annotation:**
   - Store ground truth in `tests/data/pakistani_plates_ground_truth.json` with fields: `image_path`, `ground_truth_plate`, `province`, `lighting_condition`, `plate_bbox`, `vehicle_bbox`.
3. **Dual Metric Reporting:**
   - Report **Stage 1 (Plate Localization mAP@50)**.
   - Report **Stage 2 (Cropped OCR Character Error Rate & Exact Match Rate)**.
   - Report **End-to-End System Ingress-to-Actuation Accuracy**.
