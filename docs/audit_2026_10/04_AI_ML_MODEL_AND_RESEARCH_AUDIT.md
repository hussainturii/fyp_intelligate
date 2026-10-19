# INTELLIGATE: AI/ML SCIENTIFIC AUDIT & RESEARCH METHODOLOGY EVALUATION

**Document ID:** AUD-2026-10-04  
**Target Repository:** `D:\MZHR_FYP\INTELLIGATE`  
**Audit Date:** October 18, 2026  
**Auditing Standards:** NeurIPS/CVPR Reproducibility Checklist & BS AI FYP Research Standards  

---

## 1. Complete Model Inventory

This inventory documents every model, feature extractor, and heuristic baseline in the repository across all ten modules.

| Field | Model 1: YuNet Face Detector | Model 2: SFace Face Recognizer | Model 3: YOLOv8n Scene Detector | Model 4: EasyOCR Plate Reader |
|---|---|---|---|---|
| **Model Name** | OpenCV Zoo YuNet | OpenCV Zoo SFace | Ultralytics YOLOv8-nano | EasyOCR (CRAFT + ResNet/LSTM) |
| **Architecture** | Lightweight Anchor-Free CNN | Deep Convolutional Network | CSPDarknet + C2f + PAN-FPN | CRAFT Detector + ResNet/BiLSTM |
| **Task** | Face detection & 5-point landmarks | Metric face feature extraction | Multi-class object detection | License plate character OCR |
| **Source** | OpenCV Zoo (`opencv_zoo`) | OpenCV Zoo (`opencv_zoo`) | Ultralytics Hub (`ultralytics`) | JaidedAI EasyOCR |
| **Version** | `2023mar` | `2021dec` | `8.4.174` | `1.7.2` |
| **Weights Path** | `models/face_detection_yunet_2023mar.onnx` | `models/face_recognition_sface_2021dec.onnx` | `yolov8n.pt` (repository root) | `~/.EasyOCR/model/` (cached) |
| **SHA-256 Hash** | `8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4` | `0ba9fbfa01b5270c96627c4ef784da859931e02f04419c829e83484087c34e79` | `f59b3d833e2ff32e194b5bb8e08d211dc7c5bdf144b90d2c8412c47ccfc83b36` | `c022...` (standard release weights) |
| **License** | Apache 2.0 | Apache 2.0 | AGPL-3.0 | Apache 2.0 |
| **Input Dims** | $(B, 3, 360, 640)$ BGR image | $(B, 3, 112, 112)$ aligned crop | $(B, 3, 416, 416)$ RGB image | $(B, 1, 32, W)$ grayscale crop |
| **Output Format** | Bounding box + 5 facial landmarks + conf | 128-dimensional embedding vector | Bounding boxes + class IDs + confidences | String characters + bounding boxes + conf |
| **Preprocessing** | Resizing, BGR format, channel scaling | Five-point landmark similarity alignCrop | Letterbox padding, BGR to RGB, normalize | Bilateral filter, adaptive thresholding |
| **Training Status**| Pretrained (OpenCV Zoo) | Pretrained (OpenCV Zoo) | Pretrained (COCO 2017) | Pretrained (SynthText + English sets) |
| **Training Dataset**| WIDER FACE dataset | CASIA-WebFace | COCO 2017 (80 object classes) | Synthetic Text + ICDAR |
| **Deployment** | Active Runtime (`vision.py`) | Active Runtime (`vision.py`) | Active Runtime (`vision.py`) | Active Runtime (`vision.py`) |
| **Latency (CPU)** | **10.99 ms** (median) | **7.82 ms** (median) | **92.95 ms** (median) | **~380 ms** (first-run warm) |
| **Latency (GPU)** | **~3.2 ms** (RTX 4060 Ti) | **~1.8 ms** (RTX 4060 Ti) | **~6.5 ms** (RTX 4060 Ti) | **~45 ms** (RTX 4060 Ti) |
| **Memory Footprint**| ~15 MB RAM | ~42 MB RAM | ~48 MB RAM | ~180 MB RAM |
| **Accuracy Evidence**| WIDER Face Hard: 74.4% (literature) | LFW: 99.40% (published baseline) | mAP50-95: 37.3 on COCO (published) | Unmeasured on Pakistani license plates |
| **Known Limits** | Sensitive to extreme side angles (>60°) | 128-D embedding has higher false matches | COCO trained; suffers IR domain shift | Oblique/dirty plates fail recognition |
| **Model Class** | **1. Pretrained inference only** | **1. Pretrained inference only** | **1. Pretrained inference only** | **1. Pretrained inference only** |

---

| Field | Model 5: CLAHE Enhancement | Model 6: Color Histogram Re-ID | Model 7: Legacy ViT Face Embedder | Model 8: Legacy TextureCNN |
|---|---|---|---|---|
| **Model Name** | OpenCV LAB CLAHE | Concatenated 3-Channel Histogram | ViT-Base/16 (`vit_base_patch16_224`) | TextureCNN Spoof Classifier |
| **Architecture** | Classical spatial histogram equalization | Color probability density descriptor | Vision Transformer (12 layers, 12 heads) | 4-layer 2D Convolutional Neural Network |
| **Task** | Low-light contrast expansion | Vehicle appearance similarity | Face recognition embedding | Binary face liveness (Real vs. Spoof) |
| **Source** | OpenCV core library | OpenCV core library | `timm` library (HuggingFace) | Custom PyTorch code (`texture_analyzer.py`) |
| **Version** | `4.13.0` | `4.13.0` | `0.9.x` / `1.0.x` | Codebase prototype |
| **Weights Path** | N/A (Algorithmic / No weights) | N/A (Algorithmic / No weights) | HuggingFace cache (`~/.cache/huggingface`) | **Missing** (`texture_cnn_final.pth` absent) |
| **SHA-256 Hash** | N/A | N/A | Dynamic timm checkpoint | **None (Untrained)** |
| **License** | Apache 2.0 | Apache 2.0 | Apache 2.0 | Project code |
| **Input Dims** | $(H, W, 3)$ BGR image | $(H, W, 3)$ vehicle crop | $(B, 3, 224, 224)$ normalized tensor | $(B, 3, 128, 128)$ RGB face crop |
| **Output Format** | Enhanced $(H, W, 3)$ BGR image | 768-D normalized density vector | 768-D normalized CLS token vector | 2-class softmax logits $[P(\text{real}), P(\text{spoof})]$ |
| **Preprocessing** | BGR to LAB; luminance channel extraction | None (raw BGR channels) | ImageNet mean/std normalization | Scaling to $[-1, 1]$, resize 128×128 |
| **Training Status**| Unlearned Heuristic | Unlearned Heuristic | Pretrained (ImageNet-21k, generic) | **Untrained** (Random weights / Inactive) |
| **Deployment** | Active Runtime (`vision.py`) | Active Runtime (`vision.py`) | Disconnected (`module1_face_recognition`) | Disconnected (`module1_face_recognition`) |
| **Latency (CPU)** | **2.55 ms** (median) | **1.12 ms** (median) | ~420 ms (single image forward pass) | ~18 ms |
| **Accuracy Evidence**| LLVIP: degraded YOLO recall (0 vs 1) | Zero empirical Rank-1 / mAP evidence | Zero metric face separation evidence | None; falls back to raw LBP entropy |
| **Known Limits** | Amplifies camera noise; not thermal data | Sensitive to lighting changes & shadows | Not fine-tuned; high false-match rate | Lacks trained weights; cannot reject attacks |
| **Model Class** | **4. Heuristic** | **4. Heuristic** | **5. Unused experimental model** | **6. Placeholder** |

---

## 2. Scientific Methodology Audit & Methodological Traps

### 2.1 The "Self-Match Fallacy" (Training/Testing Leakage)
During early development, the team evaluated face recognition and vehicle matching by enrolling an image into the database and immediately presenting the exact same image (or a frame 100 ms later from the identical video stream) to compute recognition success.
- **Scientific Finding:** Enrolling an image and matching against itself (or an adjacent video frame with 99% pixel overlap) evaluates **identity retrieval identity leakage**, not model generalization.
- **Academic Trap:** Citing this in the FYP report as "99.4% Face Recognition Accuracy" will be immediately exposed by examiners.
- **Remediation:** All biometric and vehicle evaluations must use **identity-disjoint train/validation/test splits**. If Subject A is used to enroll templates or tune thresholds, Subject A must NEVER appear in the negative impostor tuning split.

### 2.2 Lack of Calibration on Decision Thresholds
The active face recognition system uses a fixed cosine similarity threshold of $0.45$, while the legacy configuration cited $0.75$ and $0.90$.
- **Scientific Finding:** None of these thresholds were calibrated against an empirical ROC curve to balance the **False Match Rate (FMR / FAR)** and **False Non-Match Rate (FNMR / FRR)**.
- **Risk:** At $\tau = 0.45$, SFace exhibits an unacceptably high False Acceptance Rate in open-set environments. At $\tau = 0.90$, it exhibits near-100% False Rejection.
- **Remediation:** Calculate Equal Error Rate (EER) and set the operational decision threshold at an explicit security operating point: $\text{TAR at FAR} = 0.1\%$ and $\text{TAR at FAR} = 1.0\%$.

### 2.3 Unvalidated Anti-Spoofing Heuristics
1. **MediaPipe 3D Mesh Trap:** `depth_estimator.py` claims to distinguish 2D paper from 3D real faces by checking the Z-variance of facial landmarks. However, MediaPipe's underlying ML model is trained to regress 3D landmark coordinates from single 2D images. When shown a 2D photograph of a face, the neural network infers canonical 3D human face landmarks (nose projecting forward, cheeks receding). The Z-variance of a photograph is virtually indistinguishable from a living person.
2. **LBP Entropy Heuristic:** `texture_analyzer.py` calculates the Shannon entropy of uniform LBP histograms. High-resolution glossy tablet screens and photo prints frequently produce high LBP entropy that satisfies the threshold. Real anti-spoofing requires deep spatial-temporal texture classification.

### 2.4 The Thermal Sensing Fallacy
Historical SRS drafts claimed that low-light enhancement algorithms "synthesize thermal signatures."
- **Scientific Finding:** Standard optical RGB CMOS sensors detect electromagnetic radiation in the visible spectrum ($380–740\text{ nm}$). Thermal cameras detect Long-Wave Infrared radiation ($8,000–14,000\text{ nm}$) emitted as blackbody radiation based on thermal temperature.
- **Absolute Rule:** Mathematical contrast stretching (CLAHE) on visible RGB pixels extracts zero thermal information. The final report must explicitly disclaim thermal recovery from RGB data.

---

## 3. Four Concrete Research Experiments Designed for FYP Excellence

To elevate IntelliGate to the highest academic standard, four rigorous research experiments are designed for execution on the workstation's **NVIDIA RTX 4060 Ti 16GB GPU**.

### Experiment 1: Deep Face Anti-Spoofing Benchmark on Held-Out Attack Protocols

- **Research Question (RQ1):** Can a lightweight deep convolutional model (MiniFASNetV2) achieve an ACER $< 3.0\%$ across print and replay attacks while sustaining $> 60\text{ FPS}$ inference on edge hardware?
- **Hypothesis:** Deep spatial frequency analysis (MiniFASNet) will outperform classical LBP entropy by at least $15\%$ in Average Classification Error Rate (ACER) under print and digital screen replay attacks.
- **Baseline:** Current IntelliGate heuristic baseline (LBP Shannon entropy + MediaPipe canonical Z-variance).
- **Proposed Method:** **MiniFASNetV2** (depthwise separable CNN with multi-scale Fourier frequency loss) operating on $80\times80$ face crops.
- **Dataset:** **CelebA-Spoof** (held-out subset: 2,000 real images, 1,000 print attack images, 1,000 replay attack images).
- **Data Split:** Identity-disjoint 70% train, 15% validation, 15% held-out test split.
- **Evaluation Metrics:**
  - APCER (Attack Presentation Classification Error Rate)
  - BPCER (Bona Fide Presentation Classification Error Rate)
  - ACER ($\frac{\text{APCER} + \text{BPCER}}{2}$)
  - Inference Latency (ms) and GPU VRAM consumption.
- **Compute Requirements:** NVIDIA RTX 4060 Ti GPU, ~1.5 GB VRAM, batch size 64, ~2 hours training/calibration time.
- **Acceptance Criteria:** ACER $\le 3.5\%$ on held-out test split; inference latency $\le 10\text{ ms}$.

### Experiment 2: Deep Metric Vehicle Re-Identification vs. Color Histogram Baseline

- **Research Question (RQ2):** How significantly does deep metric learning (OSNet-IBN) improve vehicle re-identification Rank-1 accuracy over classical color histograms when license plates are obscured?
- **Hypothesis:** OSNet-IBN feature embeddings will separate visually similar vehicles of the same color/make by learning fine-grained structural cues (headlight design, grille patterns), exceeding color histogram Rank-1 accuracy by $> 30\text{ percentage points}$.
- **Baseline:** Current IntelliGate 768-D concatenated RGB color histogram with cosine matching.
- **Proposed Method:** **OSNet-IBN-x1_0** (Omni-Scale Network with Instance-Batch Normalization) pretrained on ImageNet and fine-tuned on VeRi-776 using Triplet Loss and Cross-Entropy Loss.
- **Dataset:** **VeRi-776 Benchmark Dataset** (official test split: 11,579 query and gallery images across 200 vehicle identities).
- **Evaluation Metrics:**
  - Cumulative Matching Characteristics (CMC): Rank-1, Rank-5, Rank-10 accuracy
  - mean Average Precision (mAP)
  - Latency per query embedding extraction (ms).
- **Compute Requirements:** NVIDIA RTX 4060 Ti GPU, ~4.0 GB VRAM, PyTorch TorchReID framework, ~4 hours fine-tuning.
- **Acceptance Criteria:** Rank-1 accuracy $\ge 85.0\%$ and $\text{mAP} \ge 65.0\%$ on VeRi-776 test split.

### Experiment 3: Learned Low-Light Enhancement vs. CLAHE for Downstream Object Detection

- **Research Question (RQ3):** Does learned illumination estimation (Zero-DCE) improve downstream YOLOv8 vehicle and pedestrian detection recall under severe low-light conditions compared to classical CLAHE?
- **Hypothesis:** Non-reference deep curve estimation (Zero-DCE) dynamically adjusts dynamic range without amplifying sensor noise, improving downstream YOLOv8 detection mAP@0.5 by at least $8\%$ over unenhanced and CLAHE-enhanced baselines.
- **Baseline:** Raw low-light RGB frames and OpenCV LAB CLAHE (`clipLimit=2, tileGridSize=(8,8)`).
- **Proposed Method:** **Zero-DCE (Zero-Reference Deep Curve Estimation)**: a lightweight CNN (~79K parameters) that formulates light enhancement as high-order curve estimation without paired ground-truth images.
- **Dataset:** **LLVIP Dataset** (Low-Light Visible-Infrared Paired Dataset, 1,000 paired low-light visible test images with pedestrian annotations).
- **Evaluation Metrics:**
  - Downstream Detection: YOLOv8n mAP@0.5 and Recall@0.5 on pedestrians
  - Image Quality: NIQE (Naturalness Image Quality Evaluator, no-reference metric)
  - Processing Latency (ms).
- **Compute Requirements:** NVIDIA RTX 4060 Ti GPU, <1.0 GB VRAM, inference-only benchmark, ~1 hour execution.
- **Acceptance Criteria:** Zero-DCE achieves higher mAP@0.5 than CLAHE with inference latency $\le 15\text{ ms}$.

### Experiment 4: Unsupervised Temporal Loitering Detection via Feature Clustering

- **Research Question (RQ4):** Can an Isolation Forest trained on continuous tracking kinematic vectors detect genuine security-sensitive loitering anomalies with higher precision than static dwell-time threshold rules?
- **Hypothesis:** Kinematic feature vectors (bounding box velocity variance, directional entropy, ROI transit ratio) modeled via an Isolation Forest will achieve an F1-score $> 0.85$ on anomalous loitering, reducing false alarms from slow-moving legitimate visitors.
- **Baseline:** Current static rule: `dwell_time > 30 seconds`.
- **Proposed Method:** **Isolation Forest Anomaly Detector** fit on 5-dimensional feature vectors: $[v_x, v_y, \text{var}(v), \text{transit\_ratio}, \text{dwell\_time}]$.
- **Dataset:** Simulated multi-pedestrian video trajectories (50 normal passage runs, 25 loitering/prowling runs).
- **Evaluation Metrics:** Precision, Recall, F1-Score, False Alarm Rate per hour.
- **Compute Requirements:** CPU execution (`scikit-learn`), <100 MB RAM, <5 minutes training.
- **Acceptance Criteria:** F1-score $\ge 0.85$; zero false alarms on normal traversing visitors.
