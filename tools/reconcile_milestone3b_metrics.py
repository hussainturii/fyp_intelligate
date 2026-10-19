"""
Reconcile Milestone 3B Metrics:
1. Ground-truth verified detection metrics (TP, FP, FN, Precision, Recall, F1, mAP50, mAP50-95, Mean IoU)
   with Wilson 95% Confidence Intervals on Dataset A (34 scenes) and Dataset B (155 scenes).
2. Comparison across confidence thresholds (0.25 vs 0.50).
3. End-to-end ALPR pipeline reconciliation distinguishing:
   - Frames processed
   - Ground-truth plates
   - Detected vehicles
   - Candidate plate crops
   - Ground-truth verified localizations (IoU >= 0.50)
   - False positive crops
   - Missed plates (FN)
   - OCR invocations
   - Exact string matches
   - Gate decisions
"""

from pathlib import Path
import sys
import json
import math
import time
import cv2
import numpy as np
import torch
from ultralytics import YOLO

ROOT = Path("D:/MZHR_FYP/INTELLIGATE")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from module2_vehicle.plate_detector import PlateDetector
from module2_vehicle.plate_ocr import PlateOCR
from module2_vehicle.database.vehicle_db import VehicleDatabase

def wilson_score_interval(successes, total, confidence=0.95):
    if total == 0:
        return 0.0, 0.0
    z = 1.95996  # 95% confidence
    p_hat = successes / total
    denom = 1 + (z**2) / total
    centre = (p_hat + (z**2) / (2 * total)) / denom
    spread = (z * math.sqrt((p_hat * (1 - p_hat) + (z**2) / (4 * total)) / total)) / denom
    lower = max(0.0, centre - spread)
    upper = min(1.0, centre + spread)
    return round(lower * 100.0, 2), round(upper * 100.0, 2)

def compute_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0.0

def load_ground_truths(lbl_path, img_w, img_h):
    gt_boxes = []
    if not lbl_path.exists():
        return gt_boxes
    for line in lbl_path.read_text(encoding="utf-8", errors="ignore").strip().splitlines():
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

def evaluate_detector_on_set(detector_type, model_or_det, img_dir, lbl_dir, conf_thresh=0.25, device="cuda:0"):
    img_files = sorted(list(img_dir.glob("*.*")))
    total_gts = 0
    total_preds = 0
    tp = 0
    fp = 0
    ious = []
    latencies = []

    for img_p in img_files:
        img = cv2.imread(str(img_p))
        if img is None:
            continue
        h, w = img.shape[:2]
        gts = load_ground_truths(lbl_dir / f"{img_p.stem}.txt", w, h)
        total_gts += len(gts)

        t0 = time.perf_counter()
        preds = []
        if detector_type == "contour":
            plates = model_or_det.detect(img)
            preds = [p.bbox for p in plates]
        else:
            # YOLO
            res = model_or_det.predict(img, imgsz=640, conf=conf_thresh, device=device, verbose=False)[0]
            preds = [tuple(map(int, b.xyxy[0].cpu().tolist())) for b in res.boxes]
        latencies.append((time.perf_counter() - t0) * 1000.0)

        total_preds += len(preds)

        # One-to-one greedy matching
        matched_gt = set()
        for p in preds:
            best_iou = 0.0
            best_idx = -1
            for g_idx, gt in enumerate(gts):
                if g_idx in matched_gt:
                    continue
                iou = compute_iou(p, gt)
                if iou > best_iou:
                    best_iou = iou
                    best_idx = g_idx
            if best_iou >= 0.50 and best_idx >= 0:
                tp += 1
                matched_gt.add(best_idx)
                ious.append(best_iou)
            else:
                fp += 1

    fn = total_gts - tp
    prec = tp / total_preds if total_preds > 0 else 0.0
    rec = tp / total_gts if total_gts > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

    p_ci = wilson_score_interval(tp, total_preds)
    r_ci = wilson_score_interval(tp, total_gts)

    return {
        "sample_count": len(img_files),
        "total_gts": total_gts,
        "total_preds": total_preds,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision_pct": round(prec * 100.0, 2),
        "precision_ci95": p_ci,
        "recall_pct": round(rec * 100.0, 2),
        "recall_ci95": r_ci,
        "f1_score_pct": round(f1 * 100.0, 2),
        "mean_iou_on_hits": round(float(np.mean(ious)), 4) if ious else 0.0,
        "mean_latency_ms": round(float(np.mean(latencies)), 2)
    }

def run_e2e_reconciliation():
    test_dir_a = ROOT / "data/pakistan_plates/combined_clean/test_a/images"
    test_lbl_a = ROOT / "data/pakistan_plates/combined_clean/test_a/labels"
    test_dir_b = ROOT / "data/pakistan_plates/combined_clean/test_b/images"
    test_lbl_b = ROOT / "data/pakistan_plates/combined_clean/test_b/labels"

    test_samples = []
    for p in sorted(list(test_dir_a.glob("*.*"))):
        test_samples.append((p, test_lbl_a / f"{p.stem}.txt", "test_a"))
    for p in sorted(list(test_dir_b.glob("*.*"))):
        test_samples.append((p, test_lbl_b / f"{p.stem}.txt", "test_b"))

    print(f"\n=======================================================")
    print(f"RUNNING E2E ALPR GROUND-TRUTH RECONCILIATION ({len(test_samples)} frames)")
    print(f"=======================================================")

    vehicle_model = YOLO("yolov8n.pt")
    ocr = PlateOCR(languages=["en"], gpu=True, n_reads=1, min_confidence=0.55)
    vdb = VehicleDatabase(str(ROOT / "logs/cpu_vehicles.h5"), str(ROOT / "logs/cpu_unknown_vehicles.h5"))

    pipelines = {
        "pipeline_a_contour": PlateDetector(device="cpu", allow_download=False, force_classical=True),
        "pipeline_b_yolov8n": PlateDetector(device="cuda:0", model_path="models/plate_detector_yolov8n.pt")
    }

    e2e_results = {}

    for pipe_key, det in pipelines.items():
        is_contour = (pipe_key == "pipeline_a_contour")
        print(f"\nProcessing {pipe_key}...")
        
        total_frames = len(test_samples)
        total_gt_plates = 0
        total_vehicles = 0
        total_crops_generated = 0
        gt_verified_localizations = 0
        false_positive_crops = 0
        missed_gt_plates = 0
        ocr_calls = 0
        exact_matches = 0
        
        decisions = {"ALLOW": 0, "REVIEW_REQUIRED": 0, "DENY": 0}
        per_frame_details = []

        for img_p, lbl_p, split_name in test_samples:
            img = cv2.imread(str(img_p))
            if img is None:
                continue
            h, w = img.shape[:2]
            gts = load_ground_truths(lbl_p, w, h)
            total_gt_plates += len(gts)

            # Stage 1: Vehicle Detection
            veh_res = vehicle_model.predict(img, imgsz=640, conf=0.35, classes=[2, 3, 5, 7], device="cuda:0", verbose=False)[0]
            vehicle_boxes = [tuple(map(int, b.xyxy[0].cpu().tolist())) for b in veh_res.boxes]
            total_vehicles += len(vehicle_boxes)

            search_regions = []
            if vehicle_boxes:
                for vx1, vy1, vx2, vy2 in vehicle_boxes:
                    cvx1, cvy1 = max(0, vx1), max(0, vy1)
                    cvx2, cvy2 = min(w, vx2), min(h, vy2)
                    if cvx2 > cvx1 and cvy2 > cvy1:
                        search_regions.append((cvx1, cvy1, img[cvy1:cvy2, cvx1:cvx2]))
            else:
                search_regions.append((0, 0, img))

            # Stage 2: Plate Localization
            proposed_crops = []
            for ox, oy, crop in search_regions:
                plates = det.detect(crop, offset=(ox, oy))
                if plates:
                    # In runtime pipeline, takes top candidate
                    proposed_crops.append(plates[0])
                    break # Take top candidate per frame

            total_crops_generated += len(proposed_crops)

            # Check IoU of proposed crop against ground truth
            frame_verified_hit = False
            hit_iou = 0.0
            if proposed_crops:
                top_p = proposed_crops[0]
                for gt in gts:
                    iou = compute_iou(top_p.bbox, gt)
                    if iou >= 0.50:
                        frame_verified_hit = True
                        hit_iou = iou
                        break

            if frame_verified_hit:
                gt_verified_localizations += 1
            elif proposed_crops:
                false_positive_crops += 1

            if not frame_verified_hit and gts:
                missed_gt_plates += 1

            # Stage 3: OCR
            clean_plate = ""
            ocr_conf = 0.0
            if proposed_crops and proposed_crops[0].crop.size > 0:
                ocr_calls += 1
                ocr_res = ocr.read(proposed_crops[0].crop)
                raw_txt = ocr_res.clean_text.upper()
                clean_plate = "".join(c for c in raw_txt if c.isalnum())
                ocr_conf = float(ocr_res.confidence)

            # Stage 4: Access Decision Policy
            decision = "DENY"
            match_owner = None
            if clean_plate and ocr_conf >= 0.55:
                match = vdb.lookup_by_plate(clean_plate)
                if match and match.get("method") == "exact_plate":
                    exact_matches += 1
                    match_owner = match.get("owner_name")
                    if ocr_conf >= 0.85:
                        decision = "ALLOW"
                    else:
                        decision = "REVIEW_REQUIRED"
                else:
                    decision = "DENY" if ocr_conf >= 0.80 else "REVIEW_REQUIRED"
            decisions[decision] += 1

            per_frame_details.append({
                "file": img_p.name,
                "split": split_name,
                "has_gt": len(gts) > 0,
                "crop_proposed": len(proposed_crops) > 0,
                "gt_verified_localization": frame_verified_hit,
                "localization_iou": round(hit_iou, 4),
                "extracted_plate": clean_plate,
                "ocr_conf": round(ocr_conf, 4),
                "exact_registered_match": match_owner is not None,
                "decision": decision,
                "owner": match_owner
            })

        e2e_results[pipe_key] = {
            "total_frames": total_frames,
            "total_gt_plates": total_gt_plates,
            "total_vehicles_detected": total_vehicles,
            "candidate_crops_proposed": total_crops_generated,
            "candidate_crop_rate_pct": round(total_crops_generated / total_frames * 100.0, 2),
            "gt_verified_localizations": gt_verified_localizations,
            "gt_verified_localization_rate_pct": round(gt_verified_localizations / total_gt_plates * 100.0, 2),
            "false_positive_crops": false_positive_crops,
            "missed_gt_plates": missed_gt_plates,
            "ocr_invocations": ocr_calls,
            "exact_registered_matches": exact_matches,
            "decisions": decisions,
            "frames": per_frame_details
        }

    return e2e_results

def main():
    root = ROOT
    model_path = root / "models/plate_detector_yolov8n.pt"
    yolo_model = YOLO(str(model_path))
    contour_det = PlateDetector(device="cpu", allow_download=False, force_classical=True)

    test_a_img = root / "data/pakistan_plates/combined_clean/test_a/images"
    test_a_lbl = root / "data/pakistan_plates/combined_clean/test_a/labels"
    test_b_img = root / "data/pakistan_plates/combined_clean/test_b/images"
    test_lbl_b = root / "data/pakistan_plates/combined_clean/test_b/labels"

    print("--- 1. Evaluating Contour Baseline ---")
    contour_a = evaluate_detector_on_set("contour", contour_det, test_a_img, test_a_lbl)
    contour_b = evaluate_detector_on_set("contour", contour_det, test_b_img, test_lbl_b)

    print("--- 2. Evaluating YOLOv8n @ conf=0.25 ---")
    yolo_a_25 = evaluate_detector_on_set("yolo", yolo_model, test_a_img, test_a_lbl, conf_thresh=0.25)
    yolo_b_25 = evaluate_detector_on_set("yolo", yolo_model, test_b_img, test_lbl_b, conf_thresh=0.25)

    print("--- 3. Evaluating YOLOv8n @ conf=0.50 (High-Precision) ---")
    yolo_a_50 = evaluate_detector_on_set("yolo", yolo_model, test_a_img, test_a_lbl, conf_thresh=0.50)
    yolo_b_50 = evaluate_detector_on_set("yolo", yolo_model, test_b_img, test_lbl_b, conf_thresh=0.50)

    # Ultralytics official val on test splits
    print("--- 4. Computing Official mAP on Test Sets ---")
    val_yaml = root / "data/pakistan_plates/combined_clean/dataset.yaml"
    val_metrics = yolo_model.val(data=str(val_yaml), split="test", imgsz=640, device="cuda:0")

    map50_b = float(val_metrics.box.map50) * 100.0
    map50_95_b = float(val_metrics.box.map) * 100.0

    # Also compute on Dataset A
    # Create temporary dataset yaml for test_a
    yaml_a_content = f"path: {root.as_posix()}/data/pakistan_plates/combined_clean\ntrain: train/images\nval: test_a/images\ntest: test_a/images\nnc: 1\nnames:\n  0: license_plate\n"
    tmp_yaml_a = root / "logs/benchmarks/dataset_a_test.yaml"
    tmp_yaml_a.write_text(yaml_a_content, encoding="utf-8")
    val_metrics_a = yolo_model.val(data=str(tmp_yaml_a), split="test", imgsz=640, device="cuda:0")
    map50_a = float(val_metrics_a.box.map50) * 100.0
    map50_95_a = float(val_metrics_a.box.map) * 100.0

    detection_reconciliation = {
        "dataset_a_34_scenes": {
            "ground_truth_plates": 34,
            "contour_baseline": contour_a,
            "yolov8n_conf_025": {**yolo_a_25, "mAP50_pct": round(map50_a, 2), "mAP50_95_pct": round(map50_95_a, 2)},
            "yolov8n_conf_050": {**yolo_a_50, "mAP50_pct": round(map50_a, 2), "mAP50_95_pct": round(map50_95_a, 2)},
        },
        "dataset_b_155_scenes": {
            "ground_truth_plates": 159,
            "contour_baseline": contour_b,
            "yolov8n_conf_025": {**yolo_b_25, "mAP50_pct": round(map50_b, 2), "mAP50_95_pct": round(map50_95_b, 2)},
            "yolov8n_conf_050": {**yolo_b_50, "mAP50_pct": round(map50_b, 2), "mAP50_95_pct": round(map50_95_b, 2)},
        }
    }

    # Run E2E ALPR Reconciliation
    e2e_reconciliation = run_e2e_reconciliation()

    full_output = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "detection_evaluation_reconciled": detection_reconciliation,
        "end_to_end_alpr_reconciled": e2e_reconciliation
    }

    out_file = root / "logs/benchmarks/m03b_reconciled_metrics.json"
    out_file.write_text(json.dumps(full_output, indent=2), encoding="utf-8")
    print(f"\nSaved reconciled metrics to: {out_file}")

if __name__ == "__main__":
    main()
