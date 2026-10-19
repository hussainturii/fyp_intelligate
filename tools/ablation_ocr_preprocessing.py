"""
tools/ablation_ocr_preprocessing.py — Controlled OCR Preprocessing Ablation on Verified Crops
=============================================================================================
Evaluates controlled OCR preprocessing variants on the 50 human-verified Pakistani plate crops:
  Variant A: Baseline (Existing EasyOCR on raw crop)
  Variant B: Margin / Crop Padding (adds 12% padded border to prevent character border truncation)
  Variant C: Contrast Enhancement (CLAHE on L-channel in LAB color space)
  Variant D: Multi-Line Reading-Order Reassembly (Y-band clustering + X-sort)
  Variant E: Combined (Padding + CLAHE + Multi-Line Reassembly)

Reports: Exact-match accuracy, CER, mean latency, sample delta (improved vs degraded).
NOTE: As required by experimental integrity, these 50 samples are development-set results.
"""

import csv
import json
import re
import time
from pathlib import Path
import cv2
import easyocr
import numpy as np
from loguru import logger

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MANIFEST_CSV = PROJECT_ROOT / "data" / "pakistan_plates" / "annotations" / "ocr_annotation_manifest_verified.csv"
OUTPUT_DIR = PROJECT_ROOT / "logs" / "benchmarks"
OUTPUT_JSON = OUTPUT_DIR / "m03c_1_ocr_preprocessing_ablation.json"

def levenshtein_ops(ref: str, hyp: str):
    m, n = len(ref), len(hyp)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if ref[i - 1] == hyp[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])
    return dp[m][n]

def normalize_plate_text(text: str) -> str:
    if not text:
        return ""
    return re.sub(r"[^A-Za-z0-9]", "", text).upper()

# ─────────────────────────────────────────────────────────────────────────────
# Preprocessing Transforms
# ─────────────────────────────────────────────────────────────────────────────

def transform_baseline(img_bgr: np.ndarray) -> np.ndarray:
    """Variant A: Raw crop as-is."""
    return img_bgr

def transform_padding(img_bgr: np.ndarray, pad_pct: float = 0.12) -> np.ndarray:
    """Variant B: Border padding to prevent border character truncation."""
    h, w = img_bgr.shape[:2]
    pad_y = int(h * pad_pct)
    pad_x = int(w * pad_pct)
    # Border replicate or reflect prevents introducing artificial sharp edges
    padded = cv2.copyMakeBorder(img_bgr, pad_y, pad_y, pad_x, pad_x, cv2.BORDER_REPLICATE)
    return padded

def transform_clahe(img_bgr: np.ndarray) -> np.ndarray:
    """Variant C: CLAHE contrast enhancement on L-channel of LAB."""
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    enhanced = cv2.merge((cl, a, b))
    return cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)

# ─────────────────────────────────────────────────────────────────────────────
# EasyOCR Runner with optional multi-line reordering
# ─────────────────────────────────────────────────────────────────────────────

def run_easyocr(reader: easyocr.Reader, img_bgr: np.ndarray, multiline_sort: bool = False):
    t0 = time.perf_counter()
    try:
        detections = reader.readtext(
            img_bgr,
            detail=1,
            paragraph=False,
            allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-",
        )
    except Exception as e:
        logger.debug(f"EasyOCR error: {e}")
        return "", 0.0, (time.perf_counter() - t0) * 1000.0

    latency_ms = (time.perf_counter() - t0) * 1000.0
    if not detections:
        return "", 0.0, latency_ms

    valid_dets = [d for d in detections if d[2] >= 0.25]
    if not valid_dets:
        return "", 0.0, latency_ms

    if multiline_sort and len(valid_dets) > 1:
        # Compute centers and bounding boxes
        items = []
        for bbox, text, conf in valid_dets:
            pts = np.array(bbox)
            y_center = float(np.mean(pts[:, 1]))
            x_min = float(np.min(pts[:, 0]))
            height = float(np.max(pts[:, 1]) - np.min(pts[:, 1]))
            items.append({"bbox": bbox, "text": text.strip().upper(), "conf": conf,
                          "y": y_center, "x": x_min, "h": height})

        # Cluster into vertical lines
        # Sort initially by y
        items.sort(key=lambda item: item["y"])
        lines = []
        curr_line = [items[0]]
        for item in items[1:]:
            # If vertical distance is small relative to box height, same line
            avg_h = max(curr_line[-1]["h"], item["h"], 1.0)
            if abs(item["y"] - curr_line[-1]["y"]) < avg_h * 0.45:
                curr_line.append(item)
            else:
                lines.append(curr_line)
                curr_line = [item]
        lines.append(curr_line)

        # Sort each line left-to-right
        ordered_texts = []
        total_conf = 0.0
        n_parts = 0
        for line in lines:
            line.sort(key=lambda item: item["x"])
            for item in line:
                ordered_texts.append(item["text"])
                total_conf += item["conf"]
                n_parts += 1

        pred_text = " ".join(ordered_texts)
        avg_conf = total_conf / max(n_parts, 1)
    else:
        parts = [d[1].strip().upper() for d in valid_dets]
        pred_text = " ".join(parts)
        avg_conf = float(np.mean([d[2] for d in valid_dets]))

    return pred_text, avg_conf, latency_ms

# ─────────────────────────────────────────────────────────────────────────────
# Main Ablation Runner
# ─────────────────────────────────────────────────────────────────────────────

def run_ablation():
    logger.info("Initializing EasyOCR reader (GPU if available)...")
    reader = easyocr.Reader(["en"], gpu=True, verbose=False)

    with open(MANIFEST_CSV, "r", encoding="utf-8") as f:
        samples = list(csv.DictReader(f))

    logger.info(f"Loaded {len(samples)} samples for ablation.")

    variants = [
        {"id": "A_Baseline", "name": "Existing Baseline (Raw Crop)", "pad": False, "clahe": False, "multiline": False},
        {"id": "B_Padding", "name": "Border Padding (+12%)", "pad": True, "clahe": False, "multiline": False},
        {"id": "C_CLAHE", "name": "Contrast Enhancement (CLAHE)", "pad": False, "clahe": True, "multiline": False},
        {"id": "D_MultiLine", "name": "Multi-Line Reading-Order Sort", "pad": False, "clahe": False, "multiline": True},
        {"id": "E_Pad_MultiLine", "name": "Padding + Multi-Line Sort", "pad": True, "clahe": False, "multiline": True},
        {"id": "F_Combined_All", "name": "Combined (Padding + CLAHE + Multi-Line)", "pad": True, "clahe": True, "multiline": True},
    ]

    ablation_results = {}
    baseline_predictions = {}

    for var in variants:
        v_id = var["id"]
        v_name = var["name"]
        logger.info(f"Running Ablation Variant: {v_id} ({v_name})...")

        exact_matches = 0
        total_edits = 0
        total_ref_chars = 0
        latencies = []
        variant_preds = {}

        for s in samples:
            s_id = s["sample_id"]
            crop_path = PROJECT_ROOT / s["crop_relative_path"]
            gt_text = s["human_verified_gt_text"].strip()
            norm_gt = normalize_plate_text(gt_text)

            img = cv2.imread(str(crop_path))
            if img is None:
                continue

            processed_img = img
            if var["pad"]:
                processed_img = transform_padding(processed_img, pad_pct=0.12)
            if var["clahe"]:
                processed_img = transform_clahe(processed_img)

            pred_raw, conf, lat = run_easyocr(reader, processed_img, multiline_sort=var["multiline"])
            norm_pred = normalize_plate_text(pred_raw)
            latencies.append(lat)

            edits = levenshtein_ops(norm_gt, norm_pred)
            is_exact = (norm_gt == norm_pred)
            if is_exact:
                exact_matches += 1
            total_edits += edits
            total_ref_chars += len(norm_gt)

            variant_preds[s_id] = {
                "norm_gt": norm_gt,
                "norm_pred": norm_pred,
                "is_exact": is_exact,
                "edits": edits,
                "confidence": round(conf, 4)
            }

        acc = round(exact_matches / len(samples) * 100.0, 2)
        cer = round(total_edits / max(total_ref_chars, 1) * 100.0, 2)
        mean_lat = round(float(np.mean(latencies)), 2)

        if v_id == "A_Baseline":
            baseline_predictions = variant_preds

        # Compare with baseline
        improved = []
        degraded = []
        if v_id != "A_Baseline":
            for s_id, data in variant_preds.items():
                b_data = baseline_predictions[s_id]
                if not b_data["is_exact"] and data["is_exact"]:
                    improved.append({"sample_id": s_id, "gt": data["norm_gt"], "before": b_data["norm_pred"], "after": data["norm_pred"]})
                elif b_data["is_exact"] and not data["is_exact"]:
                    degraded.append({"sample_id": s_id, "gt": data["norm_gt"], "before": b_data["norm_pred"], "after": data["norm_pred"]})

        ablation_results[v_id] = {
            "variant_name": v_name,
            "exact_matches": exact_matches,
            "exact_accuracy_pct": acc,
            "cer_pct": cer,
            "total_edits": total_edits,
            "mean_latency_ms": mean_lat,
            "corrected_samples_count": len(improved),
            "degraded_samples_count": len(degraded),
            "corrected_samples": improved,
            "degraded_samples": degraded
        }

    payload = {
        "dataset": "50 human-verified Pakistani plate crops",
        "dataset_manifest": str(MANIFEST_CSV),
        "evaluation_type": "Development-Set Preprocessing Ablation (Not Independent Held-Out Test)",
        "results": ablation_results
    }

    OUTPUT_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    logger.info(f"Wrote ablation results to {OUTPUT_JSON}")

    # Print summary
    print("\n" + "=" * 80)
    print("INTELLIGATE — SCIENTIFIC OCR PREPROCESSING ABLATION RESULTS (DEV SET)")
    print("=" * 80)
    print(f"{'Variant ID':<18} | {'Exact Matches':<14} | {'Exact Acc':<10} | {'CER (%)':<9} | {'Mean Latency':<12} | {'Delta (+ / -)':<14}")
    print("-" * 80)
    for v_id, res in ablation_results.items():
        delta_str = "Baseline" if v_id == "A_Baseline" else f"+{res['corrected_samples_count']} / -{res['degraded_samples_count']}"
        print(f"{v_id:<18} | {res['exact_matches']:2d}/50          | {res['exact_accuracy_pct']:5.2f}%    | {res['cer_pct']:5.2f}%   | {res['mean_latency_ms']:6.2f} ms   | {delta_str:<14}")
    print("=" * 80)

if __name__ == "__main__":
    run_ablation()
