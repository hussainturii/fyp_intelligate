# INTELLIGATE EXPERIMENT REPORT — M03B: BASELINE COMPARISON
## Scientific Comparison: Classical Contour Baseline vs Trained YOLOv8n Plate Detector

**Experiment ID:** `EXP-M03B-COMP-01`  
**Date:** October 19, 2026  
**Investigator:** Hussain Raza (SP23-BAI-057)  
**Status:** **REPRODUCIBLE & VERIFIED**  
**Machine-Readable Results:** [logs/benchmarks/m03b_detector_comparison.json](file:///D:/MZHR_FYP/INTELLIGATE/logs/benchmarks/m03b_detector_comparison.json)  
**Visual Comparison Assets:** [docs/report_assets/m03b_detector_comparison/](file:///D:/MZHR_FYP/INTELLIGATE/docs/report_assets/m03b_detector_comparison/)

---

## 1. Overview and Scientific Motivation

The legacy IntelliGate vehicle subsystem utilized a classical edge-detection and morphological contour filtering algorithm ([PlateDetector._detect_classical](file:///D:/MZHR_FYP/INTELLIGATE/module2_vehicle/plate_detector.py#L149-L200)). While computationally lightweight ($1.7 - 7.5\text{ ms}$ on CPU), classical edge heuristics rely on rigid assumptions regarding aspect ratio ($1.5 - 6.0$) and edge density. In real Pakistani roadway conditions, horizontal radiator grille slats, bumper seams, decorative bumper text, and headlight borders routinely trigger false positive bounding boxes while genuine dirty or off-angle plates are frequently missed.

To evaluate the empirical necessity of transitioning to a deep learning detector, we performed a controlled, side-by-side scientific benchmark comparing:
- **Baseline A:** OpenCV Edge & Morphological Contour Detector (`method="classical"`).
- **Model B:** Fine-Tuned YOLOv8n Detector on NVIDIA RTX 4060 Ti GPU (`device="cuda:0"`).
- **Model B (CPU):** Fine-Tuned YOLOv8n Detector under CPU Fallback (`device="cpu"`).

---

## 2. Evaluation Datasets (Strictly Held-Out)

Evaluation was conducted on two completely independent test splits with zero overlap or training leakage:

1. **Dataset A Test Split (34 Images, 34 Ground-Truth Plates):**
   - High-resolution outdoor street photography ($1920\times 1080$ to $4000\times 3000$).
   - Complex full-vehicle scenes under natural outdoor sunlight, shadows, road background clutter, and varied perspective angles.
   - Plate area occupies $< 2.0\%$ of total frame area.
2. **Dataset B Test Split (155 Images, 159 Ground-Truth Plates):**
   - Standardized $640\times 640$ close-cropped vehicle rear and front plates.
   - High plate coverage ($15\% - 75\%$ of frame area) with Pakistani provincial styling.

---

## 3. Comprehensive Performance Metrics

A detection is counted as a **True Positive (TP)** if its Intersection over Union (IoU) with a ground-truth bounding box satisfies:
$$\text{IoU} = \frac{\text{Area}(B_{\text{pred}} \cap B_{\text{gt}})}{\text{Area}(B_{\text{pred}} \cup B_{\text{gt}})} \ge 0.50$$
Unmatched predictions are classified as **False Positives (FP)**, and unpredicted ground truths are classified as **False Negatives (FN)**.

### 3.1 Comparative Results Table

| Dataset & Split | Model Candidate | Device | GT Plates | Preds | TP | FP | FN | Precision | Recall | F1-Score | mAP@0.50 | Mean IoU | Mean Latency |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Dataset A (Road Scenes)** | Classical Contour Baseline | CPU | 34 | 80 | 12 | 68 | 22 | **15.00%** | **35.29%** | **21.05%** | N/A | 0.8089 | **7.45 ms** |
| **Dataset A (Road Scenes)** | **Trained YOLOv8n (GPU)** | **CUDA**| **34** | **34** | **34** | **0** | **0** | **100.00%** | **100.00%** | **100.00%** | **100.00%** | **0.8604** | **38.16 ms** |
| **Dataset A (Road Scenes)** | **Trained YOLOv8n (CPU)** | **CPU** | **34** | **34** | **34** | **0** | **0** | **100.00%** | **100.00%** | **100.00%** | **100.00%** | **0.8607** | **38.11 ms** |
| **Dataset B (Plate Crops)** | Classical Contour Baseline | CPU | 159 | 192 | 20 | 172 | 139 | **10.42%** | **12.58%** | **11.40%** | N/A | 0.8104 | **1.76 ms** |
| **Dataset B (Plate Crops)** | **Trained YOLOv8n (GPU)** | **CUDA**| **159** | **167** | **155** | **12** | **4** | **92.81%** | **97.48%** | **95.09%** | **98.47%** | **0.8316** | **10.24 ms** |

*Note: For the classical contour baseline, mAP is marked N/A because it outputs uncalibrated edge-density proxies rather than continuous class probability distributions required to parameterize monotonic precision-recall curves.*

---

## 4. IoU Distribution Analysis on Successful Detections

Evaluating the geometric tightness of the predicted bounding boxes reveals that the trained YOLO model not only detects more plates, but also localizes boundaries with tighter precision:

| Metric | Dataset A (Contour) | Dataset A (YOLOv8n) | Dataset B (Contour) | Dataset B (YOLOv8n) |
| :--- | :--- | :--- | :--- | :--- |
| **Minimum IoU** | 0.5067 | 0.5306 | 0.5247 | 0.5179 |
| **25th Percentile (Q1)** | 0.7716 | 0.8382 | 0.7829 | 0.7825 |
| **Median (Q2)** | 0.8459 | **0.8726** | 0.8343 | **0.8464** |
| **75th Percentile (Q3)** | 0.8910 | 0.9013 | 0.9018 | 0.8948 |
| **Maximum IoU** | 0.9468 | 0.9384 | 0.9449 | **0.9906** |
| **Mean IoU** | 0.8089 | **0.8604** | 0.8104 | **0.8316** |

---

## 5. Visual Error Analysis and Side-by-Side Comparisons

Side-by-side visualization panels were generated and saved in `docs/report_assets/m03b_detector_comparison/` illustrating exact detection behavior on held-out test scenes.

### 5.1 Case Study 1: Real Traffic Scene with Grille Interference
- **Sample:** [compare_sample_1_da_test_DSC_0329...jpg](file:///D:/MZHR_FYP/INTELLIGATE/docs/report_assets/m03b_detector_comparison/compare_sample_1_da_test_DSC_0329_jpg.rf.2.jpg)
- **Classical Contour Failure:** The edge filter generated 3 false positive bounding boxes across the radiator grille and front bumper air intake while completely missing the genuine license plate ($FN=1, FP=3$).
- **YOLOv8n Behavior:** Precisely located the genuine plate with **0.89 confidence** and **0.87 IoU** ($TP=1, FP=0, FN=0$).

### 5.2 Case Study 2: Distant Vehicle with Shadowed Bumper
- **Sample:** [compare_sample_2_da_test_DSC_0363...jpg](file:///D:/MZHR_FYP/INTELLIGATE/docs/report_assets/m03b_detector_comparison/compare_sample_2_da_test_DSC_0363_jpg.rf.4.jpg)
- **Classical Contour Failure:** Edge density in the dark bumper shadow fell below the morphological threshold; 0 detections produced.
- **YOLOv8n Behavior:** Deep convolutional feature extraction successfully localized the plate despite strong ambient shadowing with **0.82 confidence**.

### 5.3 Case Study 3: Dual Plate / Close Urban Scene
- **Sample:** [compare_sample_3_da_test_DSC_0399...jpg](file:///D:/MZHR_FYP/INTELLIGATE/docs/report_assets/m03b_detector_comparison/compare_sample_3_da_test_DSC_0399_jpg.rf.1.jpg)
- **Classical Contour Failure:** Cluttered background contours triggered false alarms across street pavement lines.
- **YOLOv8n Behavior:** 100% precision with clean foreground/background discrimination.

### 5.4 Case Study 4: Close-Cropped Plate (Dataset B)
- **Sample:** [compare_sample_4_db_test_103...jpg](file:///D:/MZHR_FYP/INTELLIGATE/docs/report_assets/m03b_detector_comparison/compare_sample_4_db_test_103_jpg.rf.0f5fff.jpg)
- **Classical Contour Failure:** The inner character text contours were grouped separately from the plate boundary, yielding an undersized bounding box (IoU $< 0.40$).
- **YOLOv8n Behavior:** High-fidelity bounding box enclosing the full plate perimeter (IoU $= 0.91$).

---

## 6. Scientific Findings & Engineering Conclusions

1. **Catastrophic False Positive Rate of Classical Heuristics:**
   - The contour baseline produced **240 false positives** across the 189 test scenes ($68$ in Dataset A, $172$ in Dataset B). In a production ALPR pipeline, every false positive triggers an expensive downstream OCR inference pass, wasting compute and flooding audit logs.
2. **Elimination of Clutter by YOLOv8n:**
   - YOLOv8n achieved **0 false positives** on Dataset A road scenes and only **12 false positives** across 155 scenes in Dataset B (a 95.0% reduction in false detections).
3. **Robustness Across Modalities:**
   - On full-frame traffic scenes (Dataset A), YOLOv8n achieved a perfect **100.00% Recall and 100.00% Precision**, verifying that combining multi-scale datasets provides robust generalization across distant roadside cameras and close barrier checkpoints.
4. **Viability of CPU Fallback:**
   - Under CPU fallback, YOLOv8n maintains **38.11 ms latency** per frame (~26.2 FPS), proving that even in environments where GPU acceleration is offline, the deep detector remains fast enough for real-time video surveillance without falling back to error-prone edge heuristics.
