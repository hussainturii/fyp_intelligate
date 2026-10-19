# INTELLIGATE — EXPERIMENT REPORT: M02 FACE VERIFICATION BENCHMARK
**Document ID:** `IG-EXP-M02-FACE-2026-10-REV1`  
**Date:** October 18, 2026 (Revised Milestone 2.1)  
**Lead Researcher / Evaluator:** Senior Machine Learning Researcher & CV Evaluation Lead  
**Evaluation Scope:** Synthetic Diagnostic Sanity Test (OpenCV SFace 128-D Hypersphere Embedding Manifold)  
**Evaluation Classification:** **SYNTHETIC DIAGNOSTIC BENCHMARK** (Not an official LFW protocol evaluation)  
**Operational Status:** **BLOCKED ON OFFICIAL LFW DATASET ACQUISITION FOR HELD-OUT PROTOCOL**  

---

## 1. Executive Summary & Critical Errata Notice

> [!CAUTION]
> **Scientific Integrity Errata (Milestone 2.1 Correction):**  
> Earlier revisions of this report referred to this benchmark as an "LFW Face Verification Benchmark" and cited "100% TAR / 0% FAR" and "0.0% Equal Error Rate".  
> **These claims were methodologically invalid and are hereby formally retracted.**  
> The experiment conducted does **NOT** evaluate the official Labeled Faces in the Wild (LFW) 6,000-pair benchmark. It was executed using:
> - 15 synthetically perturbed image variants derived from a single reference face (`logs/demo_data/opencv_lena.jpg`).
> - 300 synthetic impostor vectors generated from uniform random sampling on a 128-dimensional unit hypersphere.  
> 
> This evaluation is formally reclassified as a **Synthetic Diagnostic Sanity Test**. It verifies that the SFace ONNX embedding extraction function is numerically well-behaved and produces expected cosine distances under synthetic perturbations. It does **not** establish real-world generalization or open-set facial verification accuracy.

---

## 2. Methodology & Sample Provenance Classification

| Parameter | Specification | Provenance Classification |
| :--- | :--- | :--- |
| **Model Evaluated** | OpenCV SFace (`face_recognition_sface_2021dec.onnx`) | Official OpenCV ONNX Model Weight |
| **Face Detector** | OpenCV YuNet (`face_detection_yunet_2023mar.onnx`) | Official OpenCV ONNX Model Weight |
| **Genuine Test Samples** | 15 synthetic pairs from 1 reference image with brightness, contrast, blur, shift, gamma shifts | **Synthetic Perturbation** |
| **Impostor Test Samples** | 300 pairs comparing genuine embeddings against random 128-D hypersphere vectors | **Synthetic Pseudo-Manifold** |
| **Real LFW Dataset Status** | LFW images and official `pairs.txt` are **NOT** present in the local repository | **BLOCKED (Requires Download Approval)** |

---

## 3. Synthetic Diagnostic Measurements

### 3.1 Cosine Similarity Statistics (Diagnostic Only)
The synthetic diagnostic confirmed that the mathematical dot-product behavior of the normalized SFace output is stable under artificial affine and lighting distortions:
- **Genuine Perturbation Mean Similarity:** $0.9670 \pm 0.0141$ (Range: $[0.9401, 0.9998]$).
- **Synthetic Impostor Mean Similarity:** $0.0075 \pm 0.0943$ (Range: $[-0.2312, 0.2745]$).
- **Diagnostic Separation Gap:** $0.9595$.

### 3.2 Methodological Deficiencies & Failure Analysis
1. **Severe Distributional Bias:** Random Gaussian vectors on a 128-D sphere are nearly orthogonal ($\mathbb{E}[u \cdot v] \approx 0$). Real human face embeddings cluster in complex, highly correlated sub-manifolds where cross-identity cosine similarity is substantially higher (typically $0.25 - 0.45$).
2. **Identity Variance Omission:** Testing perturbations of only one reference face ignores intra-class variance across age, ethnicity, facial hair, pose, expression, and occlusion.
3. **Threshold-Selection Leakage:** Calculating EER or claiming optimal operational thresholds from synthetic vectors produces artificially optimistic results that cannot be transferred to real-world gate operations.

---

## 4. Official Real-Data LFW Protocol Requirements (Pending Authorization)

To conduct a scientifically legitimate face verification evaluation, the following protocol must be executed once dataset acquisition is authorized:

### 4.1 Data Requirements
1. **Official LFW Dataset (`lfw.tgz`, ~173 MB):** 13,233 images of 5,749 individuals.
2. **Official Protocol Partition (`pairs.txt`, ~150 KB):** 10-fold cross-validation partition containing 6,000 paired comparisons (3,000 genuine same-person pairs, 3,000 impostor different-person pairs).

### 4.2 Error Accounting Protocol
- **Detection Failures vs Verification Errors:** In the event that YuNet fails to detect a face in either image of a pair, the failure must be explicitly logged as a **Face Detection Error (FDE)** rather than silently dropped or scored as distance zero.
- **Strict 10-Fold Cross-Validation:** For each fold $k \in \{1 \dots 10\}$, threshold $\tau_k$ must be optimized strictly on the remaining 9 training folds, then tested out-of-sample on fold $k$. This prevents threshold-selection leakage.
- **Metrics to Report:**
  - Mean 10-Fold Accuracy ($\% \pm \text{std}$)
  - Area Under the ROC Curve (AUC)
  - True Accept Rate at strict False Accept thresholds ($TAR @ FAR = 0.001$, $TAR @ FAR = 0.01$)
  - Equal Error Rate (EER)

---

## 5. Gate Access Security Policy Invariant

> [!IMPORTANT]
> **Advisory-Only Access Invariant Maintained:**  
> Regardless of any current or future face verification score or threshold, **IntelliGate strictly prohibits autonomous barrier gate opening based solely on facial recognition**.  
> Every facial recognition match automatically resolves to `ACCESS_DECISION: REVIEW_REQUIRED` in [software_runtime.py](file:///D:/MZHR_FYP/INTELLIGATE/module8_dashboard/backend/software_runtime.py), requiring a human security guard to visually confirm the subject's identity before actuation. This fail-secure design prevents unauthorized access resulting from spoofing attacks, model misclassifications, or open-set threshold drift.
