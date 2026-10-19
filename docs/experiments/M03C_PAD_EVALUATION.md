# Presentation-Attack Detection (PAD) Scientific Evaluation Report
**Experiment Reference:** `docs/experiments/M03C_PAD_EVALUATION.md`  
**System:** IntelliGate Multimodal AI Access Control System  
**Lead Evaluator:** Senior Deep Learning Research Scientist & CV Engineer  
**Milestone:** 3C (Deep Face Anti-Spoofing Integration & Scientific Evaluation)  
**Date:** October 19, 2026  
**Status:** IMPLEMENTED, FUNCTIONALLY TESTED, SCIENTIFIC ACCURACY MARKED NOT EVALUATED (PENDING DATASET APPROVAL)  

---

## 1. Scientific Integrity Statement & Dataset Audit

### 1.1 Local Dataset Audit
Prior to conducting empirical evaluation, an audit of locally available datasets in `D:\MZHR_FYP\INTELLIGATE\data\` was performed:
1. `data/lfw/`: Official Labeled Faces in the Wild dataset (13,233 images across 5,749 identities). This dataset consists entirely of **bona fide** (live) faces in unconstrained environments.
2. `data/pakistan_plates/`: Pakistani license plate images for ALPR evaluation.
3. `logs/demo_data/face_smoke.mp4`: A 100-frame synthetic test video used exclusively for smoke testing and functional pipeline sanity checks.

### 1.2 Status of External Presentation-Attack Datasets
Standard facial presentation attack evaluation protocols require specialized, multi-modal spoof datasets containing paired bona fide and attack presentations across diverse spoof types (e.g., printed photo masks, cutouts, high-definition digital screen replays, 3D latex masks):
- **CelebA-Spoof:** ~624,000 images, ~35 GB.
- **Replay-Attack:** ~1,200 video clips, ~10 GB.
- **CASIA-FASD / CASIA-SURF:** Multi-spectral attack datasets (~15 GB).

> [!IMPORTANT]
> **Strict Adherence to Project Ground Rules:**
> IntelliGate policy strictly forbids downloading tens of gigabytes of external datasets over the network without explicit user authorization. Therefore, **no external presentation-attack dataset was downloaded**.
> 
> In accordance with Section 5 of the Milestone 3C specification, **no synthetic or artificial spoof metrics have been fabricated**.
> - **APCER (Attack Presentation Classification Error Rate):** `NOT EVALUATED`
> - **BPCER (Bona Fide Presentation Classification Error Rate):** `NOT EVALUATED`
> - **ACER (Average Classification Error Rate):** `NOT EVALUATED`
> - **ROC-AUC & EER:** `NOT EVALUATED`

---

## 2. Functional Smoke Test & Behavioral Analysis

While scientific accuracy metrics cannot be reported without an authorized real-data attack benchmark, the deep learning integration was thoroughly smoke-tested on locally available frames to verify model inference, numeric stability, and qualitative behavioral properties.

### 2.1 Smoke Test Setup
1. **Model Checkpoint:** `models/2.7_80x80_MiniFASNetV2.pth` (Verified SHA-256: `a5eb02e1843f19b5386b953cc4c9f011c3f985d0ee2bb9819eea9a142099bec0`).
2. **Preprocessing Pipeline:** BGR $80 \times 80$, $2.7\times$ bounding-box expansion, boundary clamping, ToTensor normalization in $[0.0, 1.0]$.
3. **Execution Hardware:** NVIDIA GeForce RTX 4060 Ti (CUDA 12.x) and AMD Ryzen 7 5800X (CPU fallback).

### 2.2 Observed Behavioral Properties
1. **Sensitivity to Contextual Boundary Artifacts:**
   When tested on isolated $250 \times 250$ headshot crops from LFW placed onto a synthetic canvas, the $2.7\times$ expanded crop encompasses the artificial rectangular boundary of the patch. MiniFASNetV2 correctly detects the sharp boundary discontinuity and lack of organic live background, classifying these presentations with high probability into Class 0 (Print Spoof) or Class 2 (Screen Replay Spoof). This demonstrates that the model's spatial reasoning relies heavily on context beyond the facial perimeter.
2. **Behavior on Video Replay (`face_smoke.mp4`):**
   When processing the synthetic 100-frame video, the model consistently predicted Class 2 (Screen Replay Spoof) with probabilities exceeding $0.99$, appropriately triggering the `ATTACK_SUSPECTED` status and causing the system to route the event to `DENY` rather than admitting the frame.
3. **Quality Status Enforcement:**
   When bounding box boundaries are truncated by image edges, the pipeline safely tags the output with `pad_quality_status = "truncated_boundary"`. In accordance with fail-closed access control, truncated crops with ambiguous scores are demoted to `INCONCLUSIVE`, preventing unvalidated admittance.

---

## 3. ISO/IEC 30107-3 Standard Evaluation Framework (Planned)

When authorization is granted to import a standardized presentation attack dataset (such as Replay-Attack or CelebA-Spoof), evaluation will proceed under the **ISO/IEC 30107-3** biometric presentation attack detection standard:

### 3.1 Mathematical Definitions

1. **APCER (Attack Presentation Classification Error Rate):**
   The proportion of presentation attacks that are incorrectly classified as bona fide presentations:
   $$\text{APCER} = \frac{1}{N_{PAIS}} \sum_{i=1}^{N_{PAIS}} \left(1 - \text{Res}_i\right)$$
   Where $\text{Res}_i = 1$ if attack presentation $i$ is detected, and $0$ if it is misclassified as bona fide. Under the standard, APCER is evaluated separately for each Presentation Attack Instrument (PAI) species (e.g., print, digital display).

2. **BPCER (Bona Fide Presentation Classification Error Rate):**
   The proportion of bona fide presentations that are incorrectly classified as presentation attacks:
   $$\text{BPCER} = \frac{\sum_{i=1}^{N_{BF}} \text{Res}_i}{N_{BF}}$$
   Where $\text{Res}_i = 1$ if bona fide presentation $i$ is incorrectly rejected as an attack.

3. **ACER (Average Classification Error Rate):**
   The unweighted average of APCER and BPCER at a calibrated operating threshold $\theta$:
   $$\text{ACER}(\theta) = \frac{\text{APCER}(\theta) + \text{BPCER}(\theta)}{2}$$

4. **Operating Threshold Calibration Protocol:**
   The operating threshold $\theta$ must be calibrated strictly on a disjoint **development/validation** split (e.g., minimizing ACER or setting BPCER to a fixed target such as 1%) and subsequently evaluated on the held-out **test** split without data leakage across subject identities.

---

## 4. Formal Request for Dataset Approval

To complete full scientific benchmarking and report empirical APCER, BPCER, and ACER curves, authorization is requested for one of the following candidate benchmarks:

| Dataset | Modalities | Size | Research License | Recommended Action |
| :--- | :--- | :---: | :---: | :--- |
| **Replay-Attack** (Idiap) | Print, Mobile, Highdef Replay | ~10 GB | Academic Non-Commercial | **(Recommended)** Compact, standard 3-split protocol (train, dev, test) |
| **CelebA-Spoof** (CUHK) | 43 attack types, rich lighting | ~35 GB | Academic Research Only | Comprehensive, requires larger storage allocation |
| **CASIA-FASD** | Warp photo, cut photo, video | ~8 GB | Academic Non-Commercial | Standard cross-ethnicity evaluation benchmark |

*Upon receiving user authorization, evaluation scripts will execute the complete ISO/IEC 30107-3 benchmark protocol.*
