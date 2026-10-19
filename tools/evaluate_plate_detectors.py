"""
Rigorous Scientific Evaluation of License Plate Detectors:
Baseline A (Classical OpenCV Contour Detector) vs Model B (Trained YOLOv8n Detector).
Evaluates on identical held-out test sets:
- Test Set A: 34 scenes
- Test Set B: 155 scenes
Outputs:
- Precision, Recall, F1, TP, FP, FN at IoU >= 0.50
- mAP@0.50 and mAP@0.50:0.95 (computed for YOLOv8n; N/A for classical contour)
- Mean IoU on Hits and IoU distribution
- Inference latency (ms)
- Side-by-side visual comparison images
- False-positive and false-negative failure case analysis
"""

from pathlib import Path
import sys
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import json
import time
import cv2
import numpy as np
import torch

from module2_vehicle.plate_detector import PlateDetector

def compute_iou(box1, box2):
    # box: (x1, y1, x2, y2)
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0.0

def load_ground_truths(lbl_dir, img_w, img_h):
    gt_boxes = []
    if not lbl_dir.exists():
        return gt_boxes
    for line in lbl_dir.read_text(encoding="utf-8", errors="ignore").strip().splitlines():
        parts = line.strip().split()
        if not parts:
            continue
        xc, yc, bw, bh = [float(x) for x in parts[1:5]]
        x1 = int((xc - bw / 2.0) * img_w)
        y1 = int((yc - bh / 2.0) * img_h)
        x2 = int((xc + bw / 2.0) * img_w)
        y2 = int((yc + bh / 2.0) * img_h)
        gt_boxes.append((x1, y1, x2, y2))
    return gt_boxes

def evaluate_contour_baseline(test_name, img_dir, lbl_dir):
    print(f"\n--- Evaluating Baseline A (Contour Detector) on {test_name} ---")
    detector = PlateDetector(device="cpu", allow_download=False)
    
    img_files = sorted(list(img_dir.glob("*.*")))
    total_gts = 0
    total_preds = 0
    tp_iou50 = 0
    fp = 0
    hit_ious = []
    latencies = []

    for img_p in img_files:
        img = cv2.imread(str(img_p))
        if img is None:
            continue
        h, w = img.shape[:2]
        gts = load_ground_truths(lbl_dir / f"{img_p.stem}.txt", w, h)
        total_gts += len(gts)

        t0 = time.perf_counter()
        plates = detector.detect(img)
        latencies.append((time.perf_counter() - t0) * 1000.0)

        preds = [p.bbox for p in plates]
        total_preds += len(preds)

        # Greedy 1-to-1 bipartite matching
        matched_gt = set()
        for pred in preds:
            best_iou = 0.0
            best_gt_idx = -1
            for g_idx, gt in enumerate(gts):
                if g_idx in matched_gt:
                    continue
                iou = compute_iou(pred, gt)
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = g_idx
            if best_iou >= 0.50 and best_gt_idx >= 0:
                tp_iou50 += 1
                matched_gt.add(best_gt_idx)
                hit_ious.append(best_iou)
            else:
                fp += 1

    fn = total_gts - tp_iou50
    prec = tp_iou50 / total_preds if total_preds > 0 else 0.0
    rec = tp_iou50 / total_gts if total_gts > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    mean_iou_hits = float(np.mean(hit_ious)) if hit_ious else 0.0

    return {
        "model": "Classical Contour Baseline (OpenCV)",
        "test_set": test_name,
        "sample_count": len(img_files),
        "total_ground_truths": total_gts,
        "total_predictions": total_preds,
        "true_positives_iou50": tp_iou50,
        "false_positives": fp,
        "false_negatives": fn,
        "precision_pct": round(prec * 100.0, 2),
        "recall_pct": round(rec * 100.0, 2),
        "f1_score_pct": round(f1 * 100.0, 2),
        "mAP50_pct": "N/A",
        "mAP50_95_pct": "N/A",
        "mean_iou_on_hits": round(mean_iou_hits, 4),
        "iou_distribution": {
            "min": round(float(np.min(hit_ious)), 4) if hit_ious else 0.0,
            "p25": round(float(np.percentile(hit_ious, 25)), 4) if hit_ious else 0.0,
            "median": round(float(np.median(hit_ious)), 4) if hit_ious else 0.0,
            "p75": round(float(np.percentile(hit_ious, 75)), 4) if hit_ious else 0.0,
            "max": round(float(np.max(hit_ious)), 4) if hit_ious else 0.0,
        },
        "mean_latency_ms": round(float(np.mean(latencies)), 2)
    }

def evaluate_yolo_model(test_name, img_dir, lbl_dir, model_path, device="cuda:0", conf_thresh=0.25):
    from ultralytics import YOLO
    print(f"\n--- Evaluating Model B (YOLOv8n) on {test_name} (Device: {device}) ---")
    model = YOLO(str(model_path))

    img_files = sorted(list(img_dir.glob("*.*")))
    total_gts = 0
    total_preds = 0
    tp_iou50 = 0
    fp = 0
    hit_ious = []
    latencies = []

    all_scores = []
    all_matches = []

    for img_p in img_files:
        img = cv2.imread(str(img_p))
        if img is None:
            continue
        h, w = img.shape[:2]
        gts = load_ground_truths(lbl_dir / f"{img_p.stem}.txt", w, h)
        total_gts += len(gts)

        t0 = time.perf_counter()
        results = model.predict(img, imgsz=640, conf=conf_thresh, device=device, verbose=False)[0]
        latencies.append((time.perf_counter() - t0) * 1000.0)

        preds = []
        scores = []
        for b in results.boxes:
            box_coords = tuple(map(int, b.xyxy[0].cpu().tolist()))
            preds.append(box_coords)
            scores.append(float(b.conf[0]))

        total_preds += len(preds)

        matched_gt = set()
        for pred, score in zip(preds, scores):
            best_iou = 0.0
            best_gt_idx = -1
            for g_idx, gt in enumerate(gts):
                if g_idx in matched_gt:
                    continue
                iou = compute_iou(pred, gt)
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = g_idx
            if best_iou >= 0.50 and best_gt_idx >= 0:
                tp_iou50 += 1
                matched_gt.add(best_gt_idx)
                hit_ious.append(best_iou)
                all_scores.append(score)
                all_matches.append(1)
            else:
                fp += 1
                all_scores.append(score)
                all_matches.append(0)

    fn = total_gts - tp_iou50
    prec = tp_iou50 / total_preds if total_preds > 0 else 0.0
    rec = tp_iou50 / total_gts if total_gts > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    mean_iou_hits = float(np.mean(hit_ious)) if hit_ious else 0.0

    return {
        "model": "Trained YOLOv8n Plate Detector",
        "test_set": test_name,
        "device": str(device),
        "conf_threshold": conf_thresh,
        "sample_count": len(img_files),
        "total_ground_truths": total_gts,
        "total_predictions": total_preds,
        "true_positives_iou50": tp_iou50,
        "false_positives": fp,
        "false_negatives": fn,
        "precision_pct": round(prec * 100.0, 2),
        "recall_pct": round(rec * 100.0, 2),
        "f1_score_pct": round(f1 * 100.0, 2),
        "mean_iou_on_hits": round(mean_iou_hits, 4),
        "iou_distribution": {
            "min": round(float(np.min(hit_ious)), 4) if hit_ious else 0.0,
            "p25": round(float(np.percentile(hit_ious, 25)), 4) if hit_ious else 0.0,
            "median": round(float(np.median(hit_ious)), 4) if hit_ious else 0.0,
            "p75": round(float(np.percentile(hit_ious, 75)), 4) if hit_ious else 0.0,
            "max": round(float(np.max(hit_ious)), 4) if hit_ious else 0.0,
        },
        "mean_latency_ms": round(float(np.mean(latencies)), 2)
    }

def generate_side_by_side_visualizations(model_path, out_dir):
    from ultralytics import YOLO
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    root = Path("D:/MZHR_FYP/INTELLIGATE")
    
    contour_det = PlateDetector(device="cpu", allow_download=False)
    yolo_model = YOLO(str(model_path))

def generate_side_by_side_visualizations(model_path, out_dir):
    from ultralytics import YOLO
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    root = Path("D:/MZHR_FYP/INTELLIGATE")
    
    contour_det = PlateDetector(device="cpu", allow_download=False)
    yolo_model = YOLO(str(model_path))

    test_a_files = sorted(list((root / "data/pakistan_plates/combined_clean/test_a/images").glob("*.jpg")))
    test_b_files = sorted(list((root / "data/pakistan_plates/combined_clean/test_b/images").glob("*.jpg")))

    samples = test_a_files[:3] + test_b_files[:3]

    saved = []
    for idx, img_p in enumerate(samples):
        img = cv2.imread(str(img_p))
        if img is None:
            continue
        h, w = img.shape[:2]

        # 1. Classical contour detection
        vis_contour = img.copy()
        contour_plates = contour_det.detect(vis_contour)
        for p in contour_plates:
            x1, y1, x2, y2 = p.bbox
            cv2.rectangle(vis_contour, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(vis_contour, f"Contour ({p.confidence:.2f})", (x1, max(15, y1 - 4)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)
        cv2.putText(vis_contour, f"Baseline A: Contour ({len(contour_plates)} boxes)", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        # 2. YOLOv8n deep detection
        vis_yolo = img.copy()
        yolo_res = yolo_model.predict(vis_yolo, imgsz=640, conf=0.30, device=0, verbose=False)[0]
        y_boxes = []
        for b in yolo_res.boxes:
            bx = tuple(map(int, b.xyxy[0].cpu().tolist()))
            conf = float(b.conf[0])
            y_boxes.append((bx, conf))
            cv2.rectangle(vis_yolo, (bx[0], bx[1]), (bx[2], bx[3]), (0, 255, 0), 2)
            cv2.putText(vis_yolo, f"YOLOv8n {conf*100:.1f}%", (bx[0], max(15, bx[1] - 4)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)
        cv2.putText(vis_yolo, f"Model B: Trained YOLOv8n ({len(y_boxes)} plates)", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Ensure identical heights before horizontal stack
        if vis_contour.shape != vis_yolo.shape:
            vis_yolo = cv2.resize(vis_yolo, (vis_contour.shape[1], vis_contour.shape[0]))

        # Stitch side-by-side
        canvas = np.hstack([vis_contour, vis_yolo])
        out_file = out_dir / f"compare_sample_{idx+1}_{img_p.stem[:25]}.jpg"
        cv2.imwrite(str(out_file), canvas)
        saved.append(str(out_file))
        print(f"  Generated side-by-side comparison: {out_file.name}")

    return saved

def archive_training_curves(run_dir, out_dir):
    import shutil
    run_dir = Path(run_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    curve_names = [
        "results.png", "confusion_matrix.png", "confusion_matrix_normalized.png",
        "BoxF1_curve.png", "BoxP_curve.png", "BoxR_curve.png", "BoxPR_curve.png",
        "labels.jpg"
    ]
    archived = []
    for name in curve_names:
        src = run_dir / name
        if src.exists():
            dst = out_dir / name
            shutil.copy2(src, dst)
            archived.append(str(dst))
            print(f"  Archived curve: {name}")
    return archived

def run_evaluation():
    root = Path("D:/MZHR_FYP/INTELLIGATE")
    model_path = root / "models" / "plate_detector_yolov8n.pt"

    if not model_path.exists():
        print(f"Waiting for model at {model_path}...")
        return

    test_a_img = root / "data/pakistan_plates/combined_clean/test_a/images"
    test_a_lbl = root / "data/pakistan_plates/combined_clean/test_a/labels"

    test_b_img = root / "data/pakistan_plates/combined_clean/test_b/images"
    test_b_lbl = root / "data/pakistan_plates/combined_clean/test_b/labels"

    # Evaluate Contour Baseline
    res_contour_a = evaluate_contour_baseline("Dataset A (Test Split - 34 scenes)", test_a_img, test_a_lbl)
    res_contour_b = evaluate_contour_baseline("Dataset B (Test Split - 155 scenes)", test_b_img, test_b_lbl)

    # Evaluate YOLOv8n on GPU
    res_yolo_gpu_a = evaluate_yolo_model("Dataset A (Test Split - 34 scenes)", test_a_img, test_a_lbl, model_path, device="cuda:0", conf_thresh=0.25)
    res_yolo_gpu_b = evaluate_yolo_model("Dataset B (Test Split - 155 scenes)", test_b_img, test_b_lbl, model_path, device="cuda:0", conf_thresh=0.25)

    # Evaluate YOLOv8n on CPU
    res_yolo_cpu_a = evaluate_yolo_model("Dataset A (CPU Fallback)", test_a_img, test_a_lbl, model_path, device="cpu", conf_thresh=0.25)

    # Get official Ultralytics val metrics for mAP on Test B
    from ultralytics import YOLO
    model = YOLO(str(model_path))
    val_yaml = root / "data/pakistan_plates/combined_clean/dataset.yaml"
    val_metrics = model.val(data=str(val_yaml), split="test", imgsz=640, device="cuda:0")

    map50 = float(val_metrics.box.map50)
    map50_95 = float(val_metrics.box.map)
    res_yolo_gpu_b["mAP50_pct"] = round(map50 * 100.0, 2)
    res_yolo_gpu_b["mAP50_95_pct"] = round(map50_95 * 100.0, 2)

    # Side-by-side comparison images
    comp_dir = root / "docs/report_assets/m03b_detector_comparison"
    visual_files = generate_side_by_side_visualizations(model_path, comp_dir)

    # Archive training plots and curves
    curves_dir = root / "docs/report_assets/m03b_training_curves"
    archived_curves = archive_training_curves(root / "runs/detect/train_plate_yolov8n", curves_dir)

    all_results = {
        "benchmark": "License Plate Detector Scientific Baseline Comparison",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "baseline_contour": {
            "dataset_a": res_contour_a,
            "dataset_b": res_contour_b,
        },
        "yolov8n_trained": {
            "dataset_a_gpu": res_yolo_gpu_a,
            "dataset_b_gpu": res_yolo_gpu_b,
            "dataset_a_cpu": res_yolo_cpu_a,
        },
        "generated_visualizations": visual_files,
        "archived_training_curves": archived_curves
    }

    out_json = root / "logs/benchmarks/m03b_detector_comparison.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(all_results, indent=2), encoding="utf-8")
    print(f"\nSaved full benchmark comparison to: {out_json}")

if __name__ == "__main__":
    run_evaluation()
