"""Recompute Plate Localization Metrics directly from raw detections with strict mathematical accounting.
Resolves:
1. True Positives (IoU >= 0.50), False Positives, False Negatives under 1-to-1 greedy matching.
2. Strict separation between:
   - Mean IoU on Hits (Successful Matches >= 0.50) -> strictly >= 0.50
   - Mean IoU on All Predictions (including false positives)
   - Mean IoU on All Ground Truths (including missed plates)
3. Separate reporting for Test and Validation splits on Dataset A and Dataset B.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import cv2
import numpy as np

from module2_vehicle.plate_detector import PlateDetector


def compute_iou(boxA: list[int], boxB: list[int]) -> float:
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = max(0, boxA[2] - boxA[0]) * max(0, boxA[3] - boxA[1])
    boxBArea = max(0, boxB[2] - boxB[0]) * max(0, boxB[3] - boxB[1])

    unionArea = float(boxAArea + boxBArea - interArea)
    return interArea / unionArea if unionArea > 0 else 0.0


def yolo_to_xyxy(yolo_box: list[float], img_w: int, img_h: int) -> list[int]:
    xc, yc, w, h = yolo_box
    x1 = int((xc - w / 2.0) * img_w)
    y1 = int((yc - h / 2.0) * img_h)
    x2 = int((xc + w / 2.0) * img_w)
    y2 = int((yc + h / 2.0) * img_h)
    return [max(0, x1), max(0, y1), min(img_w, x2), min(img_h, y2)]


def parse_yolo_label(line: str) -> list[float] | None:
    parts = line.strip().split()
    if len(parts) == 5:
        # Standard YOLO bbox: class xc yc w h
        try:
            return [float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])]
        except ValueError:
            return None
    elif len(parts) > 5 and len(parts) % 2 == 1:
        # YOLO segmentation polygon: class x1 y1 x2 y2 ... xn yn
        try:
            xs = [float(parts[i]) for i in range(1, len(parts), 2)]
            ys = [float(parts[i]) for i in range(2, len(parts), 2)]
            x_min, x_max = min(xs), max(xs)
            y_min, y_max = min(ys), max(ys)
            xc = (x_min + x_max) / 2.0
            yc = (y_min + y_max) / 2.0
            w = x_max - x_min
            h = y_max - y_min
            return [xc, yc, w, h]
        except ValueError:
            return None
    return None


def load_split(dataset_dir: Path, split_name: str):
    imgs_dir = dataset_dir / split_name / "images"
    lbls_dir = dataset_dir / split_name / "labels"

    if not imgs_dir.exists():
        return []

    samples = []
    img_files = sorted(list(imgs_dir.glob("*.jpg")) + list(imgs_dir.glob("*.png")) + list(imgs_dir.glob("*.jpeg")))

    for img_p in img_files:
        lbl_p = lbls_dir / f"{img_p.stem}.txt"
        gt_boxes = []
        if lbl_p.exists():
            for line in lbl_p.read_text(encoding="utf-8").strip().splitlines():
                parsed = parse_yolo_label(line)
                if parsed is not None:
                    gt_boxes.append(parsed)
        samples.append({
            "image_path": img_p,
            "yolo_boxes": gt_boxes,
        })
    return samples


def evaluate_split_rigorous(samples: list[dict], detector: PlateDetector, split_name: str):
    total_gt = 0
    total_det = 0
    tp = 0
    fp = 0

    hit_ious = []              # IoU on successful matches (>= 0.50)
    all_pred_max_ious = []     # Max IoU for each prediction (can be 0 if no overlap)
    all_gt_best_ious = []      # Best IoU achieved for each ground-truth plate

    latencies = []

    for s in samples:
        img = cv2.imread(str(s["image_path"]))
        if img is None:
            continue
        h, w = img.shape[:2]

        gt_boxes = [yolo_to_xyxy(b, w, h) for b in s["yolo_boxes"]]
        total_gt += len(gt_boxes)

        t0 = time.perf_counter()
        preds = detector.detect(img)
        latencies.append((time.perf_counter() - t0) * 1000.0)

        pred_boxes = [list(p.bbox) for p in preds]
        total_det += len(pred_boxes)

        # Build IoU matrix: shape (|preds|, |gt|)
        iou_matrix = np.zeros((len(pred_boxes), len(gt_boxes)), dtype=np.float32)
        for i, p_box in enumerate(pred_boxes):
            for j, g_box in enumerate(gt_boxes):
                iou_matrix[i, j] = compute_iou(p_box, g_box)

        # For each prediction, record its max IoU with any GT
        for i in range(len(pred_boxes)):
            max_iou = float(np.max(iou_matrix[i])) if len(gt_boxes) > 0 else 0.0
            all_pred_max_ious.append(max_iou)

        # For each GT, record best IoU achieved by any prediction
        for j in range(len(gt_boxes)):
            best_iou = float(np.max(iou_matrix[:, j])) if len(pred_boxes) > 0 else 0.0
            all_gt_best_ious.append(best_iou)

        # Greedy 1-to-1 matching (highest IoU first)
        matched_gt = set()
        matched_pred = set()

        if len(pred_boxes) > 0 and len(gt_boxes) > 0:
            # Flatten pairs sorted by descending IoU
            pairs = []
            for i in range(len(pred_boxes)):
                for j in range(len(gt_boxes)):
                    pairs.append((iou_matrix[i, j], i, j))
            pairs.sort(key=lambda x: x[0], reverse=True)

            for iou_val, p_idx, g_idx in pairs:
                if p_idx in matched_pred or g_idx in matched_gt:
                    continue
                if iou_val >= 0.50:
                    tp += 1
                    matched_pred.add(p_idx)
                    matched_gt.add(g_idx)
                    hit_ious.append(float(iou_val))

        # Remaining predictions are false positives
        fp += (len(pred_boxes) - len(matched_pred))

    fn = total_gt - tp
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    mean_iou_on_hits = float(np.mean(hit_ious)) if hit_ious else 0.0
    mean_iou_all_preds = float(np.mean(all_pred_max_ious)) if all_pred_max_ious else 0.0
    mean_iou_all_gts = float(np.mean(all_gt_best_ious)) if all_gt_best_ious else 0.0

    return {
        "split": split_name,
        "sample_count": len(samples),
        "ground_truth_plates": total_gt,
        "total_predicted_boxes": total_det,
        "true_positives_iou50": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "precision_percent": round(precision * 100.0, 2),
        "recall_percent": round(recall * 100.0, 2),
        "f1_score_percent": round(f1 * 100.0, 2),
        "mean_iou_on_hits_iou50": round(mean_iou_on_hits, 4),  # MUST BE >= 0.50
        "mean_iou_across_all_predictions": round(mean_iou_all_preds, 4),
        "mean_iou_across_all_ground_truths": round(mean_iou_all_gts, 4),
        "avg_latency_ms": round(float(np.mean(latencies)), 2) if latencies else 0.0,
    }


def main():
    print("=== RECOMPUTING PLATE LOCALIZATION METRICS DIRECTLY FROM RAW SAMPLES ===", flush=True)

    dataset_a = ROOT / "data/pakistan_plates/dataset_a"
    dataset_b = ROOT / "data/pakistan_plates/dataset_b"

    detector = PlateDetector(device="cuda:0", allow_download=False)

    # Dataset A splits
    a_test_samples = load_split(dataset_a, "test")
    a_valid_samples = load_split(dataset_a, "valid")
    a_combined = a_test_samples + a_valid_samples

    print("\n--- Evaluating Dataset A (Outdoor Vehicle Scenes) ---", flush=True)
    res_a_test = evaluate_split_rigorous(a_test_samples, detector, "Dataset A (Test Split - 34 images)")
    res_a_valid = evaluate_split_rigorous(a_valid_samples, detector, "Dataset A (Validation Split - 112 images)")
    res_a_combined = evaluate_split_rigorous(a_combined, detector, "Dataset A (Held-Out Combined - 146 images)")

    print(f"Dataset A Test:   P@50 = {res_a_test['precision_percent']}% | R@50 = {res_a_test['recall_percent']}% | F1 = {res_a_test['f1_score_percent']}% | Mean IoU on Hits = {res_a_test['mean_iou_on_hits_iou50']}")
    print(f"Dataset A Valid:  P@50 = {res_a_valid['precision_percent']}% | R@50 = {res_a_valid['recall_percent']}% | F1 = {res_a_valid['f1_score_percent']}% | Mean IoU on Hits = {res_a_valid['mean_iou_on_hits_iou50']}")
    print(f"Dataset A Total:  P@50 = {res_a_combined['precision_percent']}% | R@50 = {res_a_combined['recall_percent']}% | F1 = {res_a_combined['f1_score_percent']}% | Mean IoU on Hits = {res_a_combined['mean_iou_on_hits_iou50']}")
    print(f"                  Mean IoU on all preds: {res_a_combined['mean_iou_across_all_predictions']} | Mean IoU on all GTs: {res_a_combined['mean_iou_across_all_ground_truths']}")

    # Dataset B splits
    b_test_samples = load_split(dataset_b, "test")
    b_valid_samples = load_split(dataset_b, "valid")

    print("\n--- Evaluating Dataset B (Close-Cropped Plates) ---", flush=True)
    res_b_test = evaluate_split_rigorous(b_test_samples, detector, "Dataset B (Test Split - 155 images)")
    res_b_valid = evaluate_split_rigorous(b_valid_samples, detector, "Dataset B (Validation Split - 310 images)")

    print(f"Dataset B Test:   P@50 = {res_b_test['precision_percent']}% | R@50 = {res_b_test['recall_percent']}% | F1 = {res_b_test['f1_score_percent']}% | Mean IoU on Hits = {res_b_test['mean_iou_on_hits_iou50']}")
    print(f"Dataset B Valid:  P@50 = {res_b_valid['precision_percent']}% | R@50 = {res_b_valid['recall_percent']}% | F1 = {res_b_valid['f1_score_percent']}% | Mean IoU on Hits = {res_b_valid['mean_iou_on_hits_iou50']}")

    results = {
        "benchmark": "Rigorous Plate Localization Verification (Classical Contour Baseline)",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "dataset_a": {
            "test": res_a_test,
            "valid": res_a_valid,
            "combined": res_a_combined,
        },
        "dataset_b": {
            "test": res_b_test,
            "valid": res_b_valid,
        },
        "mathematical_corrections": [
            "Fixed Mean IoU on Hits: previous report calculated average across all non-zero overlaps (0.3736).",
            f"Correct Mean IoU on Hits (IoU >= 0.50) on Dataset A Combined is {res_a_combined['mean_iou_on_hits_iou50']:.4f} (strictly >= 0.50).",
            f"Mean IoU across all predicted candidate boxes on Dataset A Combined is {res_a_combined['mean_iou_across_all_predictions']:.4f}.",
            f"Mean IoU across all ground-truth plates on Dataset A Combined is {res_a_combined['mean_iou_across_all_ground_truths']:.4f}.",
            "Strict 1-to-1 greedy bipartite matching enforced (no double counting of ground-truth plates)."
        ]
    }

    out_file = ROOT / "logs/benchmarks/m03a_plate_localization_verified.json"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\n[SAVED] Verified localization metrics to: {out_file}", flush=True)


if __name__ == "__main__":
    main()
