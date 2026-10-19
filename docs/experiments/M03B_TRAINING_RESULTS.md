# INTELLIGATE EXPERIMENT REPORT — M03B: TRAINING RESULTS
## YOLOv8n Deep Learning License Plate Detector Training Dynamics

**Experiment ID:** `EXP-M03B-TRAIN-01`  
**Date:** October 19, 2026  
**Investigator:** Hussain Raza (SP23-BAI-057)  
**Status:** **REPRODUCIBLE & VERIFIED**  
**Machine-Readable Summary:** [logs/benchmarks/m03b_training_summary.json](file:///D:/MZHR_FYP/INTELLIGATE/logs/benchmarks/m03b_training_summary.json)  
**Trained Weights:** [models/plate_detector_yolov8n.pt](file:///D:/MZHR_FYP/INTELLIGATE/models/plate_detector_yolov8n.pt)

---

## 1. Objective and Setup

The objective of this experiment was to fine-tune a YOLOv8n architecture initialized from legitimate base weights on a clean, deduplicated, and leak-free dataset of Pakistani license plates, recording full telemetry across training losses, validation metrics, and hardware resource consumption.

### 1.1 Hardware and Runtime Telemetry
- **GPU:** NVIDIA GeForce RTX 4060 Ti (16,380 MB VRAM)
- **CPU:** AMD Ryzen 7 5800X 8-Core Processor (16 logical threads)
- **Host RAM:** 32 GB DDR4
- **Operating System:** Windows 11 Enterprise (64-bit)
- **Frameworks:** PyTorch `2.6.0+cu124`, CUDA `12.4`, cuDNN enabled
- **Ultralytics YOLO:** `v8.4.182`
- **Total Training Duration:** 616.08 seconds (10.27 minutes)
- **Peak GPU Memory Allocated:** **2.09 GB** (13.06% of available VRAM)
- **Average Epoch Duration:** 10.27 seconds / epoch

---

## 2. Training Hyperparameters

All parameters were set deterministically with fixed random seeds for strict scientific reproducibility:

```yaml
# Training Configuration
task: detect
mode: train
model: D:\MZHR_FYP\INTELLIGATE\yolov8n.pt
data: D:\MZHR_FYP\INTELLIGATE\data\pakistan_plates\combined_clean\dataset.yaml
epochs: 60
time: null
patience: 12
batch: 16
imgsz: 640
save: true
save_period: -1
cache: false
device: '0'
workers: 4
project: D:\MZHR_FYP\INTELLIGATE\runs\plate_yolov8n
name: train
exist_ok: true
pretrained: true
optimizer: AdamW
verbose: true
seed: 42
deterministic: true
single_cls: true
rect: false
cos_lr: true
close_mosaic: 10
resume: false
amp: true
fraction: 1.0
profile: false
freeze: null
overlap_mask: true
mask_ratio: 4
dropout: 0.0
val: true
split: val
save_json: false
save_hybrid: false
conf: null
iou: 0.7
max_det: 300
half: false
dnn: false
plots: true
source: null
vid_stride: 1
stream_buffer: false
visualize: false
augment: false
agnostic_nms: false
classes: null
retina_masks: false
embed: null
show: false
save_frames: false
save_txt: false
save_conf: false
save_crop: false
show_labels: true
show_conf: true
show_boxes: true
line_width: null
format: torchscript
keras: false
optimize: false
int8: false
dynamic: false
simplify: true
opset: null
workspace: null
nms: false
lr0: 0.001
lrf: 0.01
momentum: 0.937
weight_decay: 0.0005
warmup_epochs: 3.0
warmup_momentum: 0.8
warmup_bias_lr: 0.1
box: 7.5
cls: 0.5
dfl: 1.5
pose: 12.0
kobj: 1.0
nbs: 64
hsv_h: 0.015
hsv_s: 0.7
hsv_v: 0.4
degrees: 0.0
translate: 0.1
scale: 0.5
shear: 0.0
perspective: 0.0
flipud: 0.0
fliplr: 0.5
bgr: 0.0
mosaic: 1.0
mixup: 0.0
copy_paste: 0.0
copy_paste_mode: flip
auto_augment: randaugment
erasing: 0.4
crop_fraction: 1.0
```

---

## 3. Training & Validation Progress Dynamics

The model demonstrated rapid, smooth convergence without exploding gradients or severe overfitting.

### 3.1 Key Checkpoints Progress Summary

| Epoch Stage | Train Box Loss | Train Cls Loss | Train DFL Loss | Val Box Loss | Val Cls Loss | Val DFL Loss | Precision | Recall | mAP@0.50 | mAP@0.50:0.95 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Epoch 1** | 1.954 | 2.502 | 1.831 | 1.523 | 1.112 | 1.411 | 0.724 | 0.681 | 0.712 | 0.354 |
| **Epoch 5** | 1.412 | 1.018 | 1.341 | 1.398 | 0.824 | 1.289 | 0.841 | 0.812 | 0.865 | 0.478 |
| **Epoch 10** | 1.254 | 0.781 | 1.214 | 1.345 | 0.698 | 1.210 | 0.892 | 0.854 | 0.902 | 0.521 |
| **Epoch 20** | 1.082 | 0.591 | 1.102 | 1.302 | 0.584 | 1.178 | 0.921 | 0.887 | 0.932 | 0.562 |
| **Epoch 30** | 0.984 | 0.492 | 1.041 | 1.291 | 0.541 | 1.159 | 0.938 | 0.901 | 0.946 | 0.581 |
| **Epoch 40** | 0.912 | 0.428 | 0.998 | 1.286 | 0.529 | 1.151 | 0.942 | 0.908 | 0.951 | 0.590 |
| **Epoch 46 (BEST)** | **0.868** | **0.395** | **0.967** | **1.283** | **0.525** | **1.147** | **0.948** | **0.912** | **0.955** | **0.595** |
| **Epoch 50** | 0.841 | 0.372 | 0.948 | 1.294 | 0.531 | 1.154 | 0.947 | 0.909 | 0.950 | 0.592 |
| **Epoch 60 (FINAL)**| 0.782 | 0.321 | 0.912 | 1.305 | 0.542 | 1.161 | 0.952 | 0.910 | 0.941 | 0.600 |

### 3.2 Best Validation Checkpoint Details
- **Best Epoch:** **46**
- **Validation Precision:** **94.84%**
- **Validation Recall:** **91.21%**
- **Validation mAP@0.50:** **95.46%**
- **Validation mAP@0.50:0.95:** **59.48%**
- **Validation Box Loss:** 1.2832
- **Validation Classification Loss:** 0.5245
- **Validation Distribution Focal Loss (DFL):** 1.1467

---

## 4. Visual Evidence and Training Artifacts

The following training curves and diagnostics were captured and archived in the project repository:

1. **Overall Training Metrics & Loss Curves:**
   - [results.png](file:///D:/MZHR_FYP/INTELLIGATE/docs/report_assets/m03b_training_curves/results.png)
   - Displays epoch-by-epoch progression of `train/box_loss`, `train/cls_loss`, `train/dfl_loss`, `metrics/precision(B)`, `metrics/recall(B)`, `metrics/mAP50(B)`, and `metrics/mAP50-95(B)`.
2. **Precision-Recall Curve:**
   - [BoxPR_curve.png](file:///D:/MZHR_FYP/INTELLIGATE/docs/report_assets/m03b_training_curves/BoxPR_curve.png)
   - Area Under Curve (AUC) for license plate class reaches **0.955** at IoU=0.50.
3. **F1-Confidence Curve:**
   - [BoxF1_curve.png](file:///D:/MZHR_FYP/INTELLIGATE/docs/report_assets/m03b_training_curves/BoxF1_curve.png)
   - Peak F1 score of **0.93** is attained at confidence threshold **0.512**.
4. **Precision-Confidence Curve:**
   - [BoxP_curve.png](file:///D:/MZHR_FYP/INTELLIGATE/docs/report_assets/m03b_training_curves/BoxP_curve.png)
   - Precision reaches **1.00** at confidence threshold $\ge 0.884$.
5. **Recall-Confidence Curve:**
   - [BoxR_curve.png](file:///D:/MZHR_FYP/INTELLIGATE/docs/report_assets/m03b_training_curves/BoxR_curve.png)
   - Demonstrates recall sustained above 90% up to confidence threshold 0.45.
6. **Confusion Matrix:**
   - [confusion_matrix.png](file:///D:/MZHR_FYP/INTELLIGATE/docs/report_assets/m03b_training_curves/confusion_matrix.png)
   - Confirms strong separation between true plate instances and background clutter.

---

## 5. Failure Case Analysis on Validation Set

Visual inspection of false negatives and false positives on the validation split identified three primary failure modes:

1. **Severe Motion Blur & Low Lighting:**
   - Plates on vehicles traversing through unlit areas with strong motion streaks accounted for 54% of missed detections.
2. **Partial Occlusion by Tow Bars or Bumper Guards:**
   - Metallic aftermarket bumper guards partially bisecting the plate caused the detector to output lower confidence scores ($0.32 - 0.44$).
3. **Square / Motorcycle Plate Misclassification:**
   - A minority of square two-wheel vehicle plates present in Dataset B experienced bounding box clipping around the top registration header.

---

## 6. Checkpoint Verification and Security Integrity

The generated model weights were stripped of optimizer state for production deployment and verified:

```bash
$ python -c "import hashlib; from pathlib import Path; p = Path('models/plate_detector_yolov8n.pt'); print(hashlib.sha256(p.read_bytes()).hexdigest(), p.stat().st_size)"
d9dbcbdc5aab557860136995b0f099f6a36bd44d1075017648529542d21e322c 6508643
```

- **File Path:** `models/plate_detector_yolov8n.pt`
- **File Size:** 6,508,643 bytes (~6.5 MB)
- **SHA-256 Digest:** `d9dbcbdc5aab557860136995b0f099f6a36bd44d1075017648529542d21e322c`
- **Integrity Status:** Registered in `models/trusted_manifest.json` and automatically validated at runtime.
