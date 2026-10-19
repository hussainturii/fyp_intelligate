"""
tools/evaluate_verified_ocr.py — Independent Real Pakistani Plate OCR Evaluation
===============================================================================
Evaluates EasyOCR predictions against human-verified ground truth on the 50
verified Pakistani plate crops. Computes:
- Evaluation A: Raw exact transcription metrics & raw CER
- Evaluation B: Normalized registration metrics (exact match, CER, subs, dels, ins,
  confidence stats, by province, by 1-line vs 2-line).
- Detailed failure analysis and character edit logs.
"""

import csv
import json
import re
from pathlib import Path
import cv2
import numpy as np
from loguru import logger

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MANIFEST_CSV = PROJECT_ROOT / "data" / "pakistan_plates" / "annotations" / "ocr_annotation_manifest_verified.csv"
OUTPUT_DIR = PROJECT_ROOT / "logs" / "benchmarks"
OUTPUT_JSON = OUTPUT_DIR / "m03c_1_ocr_evaluation_results.json"
OUTPUT_FAILURES_CSV = OUTPUT_DIR / "m03c_1_ocr_failures.csv"

def levenshtein_ops(ref: str, hyp: str):
    """Computes Levenshtein distance, substitutions, deletions (from ref), and insertions (into hyp)."""
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
                dp[i][j] = 1 + min(
                    dp[i - 1][j],      # deletion from ref
                    dp[i][j - 1],      # insertion into hyp
                    dp[i - 1][j - 1]   # substitution
                )

    # Backtrace
    i, j = m, n
    subs, dels, ins = 0, 0, 0
    while i > 0 or j > 0:
        if i > 0 and j > 0 and ref[i - 1] == hyp[j - 1]:
            i -= 1
            j -= 1
        elif i > 0 and j > 0 and dp[i][j] == dp[i - 1][j - 1] + 1:
            subs += 1
            i -= 1
            j -= 1
        elif i > 0 and dp[i][j] == dp[i - 1][j] + 1:
            dels += 1
            i -= 1
        elif j > 0 and dp[i][j] == dp[i][j - 1] + 1:
            ins += 1
            j -= 1
        else:
            break

    return dp[m][n], subs, dels, ins

def normalize_plate_text(text: str) -> str:
    """
    Standard Pakistani Plate Normalization Policy:
    - Uppercase
    - Strip hyphens, spaces, and non-alphanumeric punctuation
    - Retain every meaningful letter and digit (including registration year codes and 2nd-line digits)
    """
    if not text:
        return ""
    cleaned = re.sub(r"[^A-Za-z0-9]", "", text).upper()
    return cleaned

def evaluate():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(MANIFEST_CSV, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    logger.info(f"Loaded {len(rows)} verified samples from {MANIFEST_CSV}")

    # Trackers for Raw Evaluation
    raw_exact_matches = 0
    raw_total_edit_distance = 0
    raw_total_ref_chars = 0

    # Trackers for Normalized Evaluation
    norm_exact_matches = 0
    norm_total_edit_distance = 0
    norm_total_ref_chars = 0
    norm_total_subs = 0
    norm_total_dels = 0
    norm_total_ins = 0

    empty_ocr_count = 0
    confidences = []
    correct_confs = []
    incorrect_confs = []

    province_stats = {}
    format_stats = {"1-line": {"total": 0, "exact": 0, "edits": 0, "ref_chars": 0},
                    "2-line": {"total": 0, "exact": 0, "edits": 0, "ref_chars": 0}}

    failure_records = []
    sample_evaluations = []

    for r in rows:
        sample_id = r["sample_id"]
        source_img = r["source_image"]
        crop_rel = r["crop_relative_path"]
        crop_path = PROJECT_ROOT / crop_rel
        raw_pred = r["easyocr_predicted_text"].strip()
        raw_gt = r["human_verified_gt_text"].strip()
        conf = float(r["easyocr_confidence"]) if r["easyocr_confidence"] else 0.0
        province = r["jurisdiction_province"]

        # Check crop geometry for 1-line vs 2-line plate format
        img = cv2.imread(str(crop_path))
        if img is not None:
            h, w = img.shape[:2]
            aspect_ratio = round(w / max(h, 1), 2)
            plate_format = "1-line" if aspect_ratio >= 2.0 else "2-line"
        else:
            h, w, aspect_ratio = 0, 0, 0.0
            plate_format = "unknown"

        # --- Evaluation A: Raw exact transcription ---
        raw_dist, raw_s, raw_d, raw_i = levenshtein_ops(raw_gt, raw_pred)
        raw_exact = (raw_gt == raw_pred)
        if raw_exact:
            raw_exact_matches += 1
        raw_total_edit_distance += raw_dist
        raw_total_ref_chars += len(raw_gt)

        # --- Evaluation B: Normalized registration text ---
        norm_gt = normalize_plate_text(raw_gt)
        norm_pred = normalize_plate_text(raw_pred)
        norm_dist, norm_s, norm_d, norm_i = levenshtein_ops(norm_gt, norm_pred)
        norm_exact = (norm_gt == norm_pred)

        if norm_exact:
            norm_exact_matches += 1
            correct_confs.append(conf)
        else:
            incorrect_confs.append(conf)

        if not norm_pred:
            empty_ocr_count += 1

        norm_total_edit_distance += norm_dist
        norm_total_ref_chars += len(norm_gt)
        norm_total_subs += norm_s
        norm_total_dels += norm_d
        norm_total_ins += norm_i
        confidences.append(conf)

        # Province stats
        if province not in province_stats:
            province_stats[province] = {"total": 0, "exact": 0, "edits": 0, "ref_chars": 0}
        province_stats[province]["total"] += 1
        if norm_exact:
            province_stats[province]["exact"] += 1
        province_stats[province]["edits"] += norm_dist
        province_stats[province]["ref_chars"] += len(norm_gt)

        # Format stats
        if plate_format in format_stats:
            format_stats[plate_format]["total"] += 1
            if norm_exact:
                format_stats[plate_format]["exact"] += 1
            format_stats[plate_format]["edits"] += norm_dist
            format_stats[plate_format]["ref_chars"] += len(norm_gt)

        # Failure classification
        error_type = "NONE"
        if not norm_exact:
            if not norm_pred:
                error_type = "EMPTY_OUTPUT"
            elif norm_d > 0 and norm_s == 0 and norm_i == 0:
                error_type = "DELETION_ONLY" # e.g. dropped year code
            elif norm_s > 0 and norm_d == 0 and norm_i == 0:
                error_type = "SUBSTITUTION_ONLY" # e.g. letter/digit confusion
            elif norm_i > 0 and norm_s == 0 and norm_d == 0:
                error_type = "INSERTION_ONLY" # e.g. border artifact read as char
            elif norm_d > 0 and norm_s > 0:
                error_type = "DROPPED_YEAR_AND_CHAR_CONFUSION"
            else:
                error_type = "COMPLEX_MISREAD"

            failure_records.append({
                "sample_id": sample_id,
                "province": province,
                "plate_format": plate_format,
                "crop_dimensions": f"{w}x{h} (ar={aspect_ratio})",
                "human_verified_gt": raw_gt,
                "easyocr_predicted": raw_pred,
                "normalized_gt": norm_gt,
                "normalized_pred": norm_pred,
                "edit_distance": norm_dist,
                "subs": norm_s,
                "dels": norm_d,
                "ins": norm_i,
                "confidence": conf,
                "error_type": error_type,
                "crop_relative_path": crop_rel
            })

        sample_evaluations.append({
            "sample_id": sample_id,
            "raw_gt": raw_gt,
            "raw_pred": raw_pred,
            "raw_exact": raw_exact,
            "norm_gt": norm_gt,
            "norm_pred": norm_pred,
            "norm_exact": norm_exact,
            "edit_distance": norm_dist,
            "confidence": conf,
            "province": province,
            "plate_format": plate_format,
            "aspect_ratio": aspect_ratio
        })

    n_samples = len(rows)
    raw_exact_acc = round(raw_exact_matches / n_samples * 100.0, 2)
    raw_cer = round(raw_total_edit_distance / max(raw_total_ref_chars, 1) * 100.0, 2)

    norm_exact_acc = round(norm_exact_matches / n_samples * 100.0, 2)
    norm_cer = round(norm_total_edit_distance / max(norm_total_ref_chars, 1) * 100.0, 2)

    avg_conf = round(float(np.mean(confidences)), 4) if confidences else 0.0
    avg_corr_conf = round(float(np.mean(correct_confs)), 4) if correct_confs else 0.0
    avg_incorr_conf = round(float(np.mean(incorrect_confs)), 4) if incorrect_confs else 0.0

    # Summary by Province
    prov_summary = {}
    for p, st in province_stats.items():
        prov_summary[p] = {
            "samples": st["total"],
            "exact_matches": st["exact"],
            "exact_accuracy_pct": round(st["exact"] / max(st["total"], 1) * 100.0, 2),
            "cer_pct": round(st["edits"] / max(st["ref_chars"], 1) * 100.0, 2)
        }

    # Summary by Format
    format_summary = {}
    for fmt, st in format_stats.items():
        if st["total"] > 0:
            format_summary[fmt] = {
                "samples": st["total"],
                "exact_matches": st["exact"],
                "exact_accuracy_pct": round(st["exact"] / max(st["total"], 1) * 100.0, 2),
                "cer_pct": round(st["edits"] / max(st["ref_chars"], 1) * 100.0, 2)
            }

    results = {
        "metadata": {
            "eval_name": "Milestone 3C.1 Independent Real Pakistani Plate EasyOCR Evaluation",
            "dataset_manifest": str(MANIFEST_CSV),
            "total_evaluated_samples": n_samples,
            "empty_ocr_outputs": empty_ocr_count,
        },
        "evaluation_a_raw_transcription": {
            "exact_matches": raw_exact_matches,
            "exact_accuracy_pct": raw_exact_acc,
            "total_edit_distance": raw_total_edit_distance,
            "total_reference_characters": raw_total_ref_chars,
            "cer_pct": raw_cer
        },
        "evaluation_b_normalized_registration": {
            "exact_matches": norm_exact_matches,
            "exact_accuracy_pct": norm_exact_acc,
            "total_edit_distance": norm_total_edit_distance,
            "total_reference_characters": norm_total_ref_chars,
            "cer_pct": norm_cer,
            "breakdown_operations": {
                "substitutions": norm_total_subs,
                "deletions": norm_total_dels,
                "insertions": norm_total_ins
            },
            "confidence_metrics": {
                "average_confidence": avg_conf,
                "correct_predictions_avg_conf": avg_corr_conf,
                "incorrect_predictions_avg_conf": avg_incorr_conf
            },
            "by_province": prov_summary,
            "by_plate_format": format_summary
        },
        "preliminary_comparison": {
            "reported_preliminary_exact_matches": 18,
            "reported_preliminary_exact_acc_pct": 36.00,
            "reported_preliminary_cer_pct": 23.76,
            "independently_reproduced_exact_matches": norm_exact_matches,
            "independently_reproduced_exact_acc_pct": norm_exact_acc,
            "independently_reproduced_cer_pct": norm_cer,
            "exact_match_difference": norm_exact_matches - 18,
            "cer_difference": round(norm_cer - 23.76, 2)
        },
        "sample_evaluations": sample_evaluations
    }

    OUTPUT_JSON.write_text(json.dumps(results, indent=2), encoding="utf-8")
    logger.info(f"Saved OCR evaluation results to {OUTPUT_JSON}")

    # Write failure log
    with open(OUTPUT_FAILURES_CSV, "w", newline="", encoding="utf-8") as f:
        if failure_records:
            writer = csv.DictWriter(f, fieldnames=list(failure_records[0].keys()))
            writer.writeheader()
            writer.writerows(failure_records)
    logger.info(f"Saved {len(failure_records)} failure cases to {OUTPUT_FAILURES_CSV}")

    # Print summary table
    print("\n" + "=" * 76)
    print("INTELLIGATE — REAL PAKISTANI PLATE EASYOCR INDEPENDENT EVALUATION")
    print("=" * 76)
    print(f"Total Evaluated Verified Samples: {n_samples}")
    print("-" * 76)
    print("EVALUATION A — RAW EXACT TRANSCRIPTION:")
    print(f"  Exact Matches:        {raw_exact_matches}/{n_samples} ({raw_exact_acc:.2f}%)")
    print(f"  Raw CER:              {raw_cer:.2f}% (Edits: {raw_total_edit_distance} / Ref chars: {raw_total_ref_chars})")
    print("-" * 76)
    print("EVALUATION B — NORMALIZED REGISTRATION TEXT:")
    print(f"  Exact Matches:        {norm_exact_matches}/{n_samples} ({norm_exact_acc:.2f}%)")
    print(f"  Normalized CER:       {norm_cer:.2f}% (Edits: {norm_total_edit_distance} / Ref chars: {norm_total_ref_chars})")
    print(f"  Edit Operations:      Subs={norm_total_subs} | Dels={norm_total_dels} | Ins={norm_total_ins}")
    print(f"  Empty Outputs:        {empty_ocr_count}")
    print(f"  Avg Confidence:       Overall={avg_conf:.4f} | Correct={avg_corr_conf:.4f} | Incorrect={avg_incorr_conf:.4f}")
    print("-" * 76)
    print("BREAKDOWN BY PROVINCE:")
    for p, st in prov_summary.items():
        print(f"  - {p:15s}: {st['exact_matches']:2d}/{st['samples']:2d} ({st['exact_accuracy_pct']:5.2f}%) | CER: {st['cer_pct']:5.2f}%")
    print("-" * 76)
    print("BREAKDOWN BY PLATE FORMAT:")
    for fmt, st in format_summary.items():
        print(f"  - {fmt:15s}: {st['exact_matches']:2d}/{st['samples']:2d} ({st['exact_accuracy_pct']:5.2f}%) | CER: {st['cer_pct']:5.2f}%")
    print("-" * 76)
    print("COMPARISON WITH PRELIMINARY ESTIMATES:")
    print(f"  Preliminary:          18/50 (36.00%) Exact | 23.76% CER")
    print(f"  Reproduced:           {norm_exact_matches}/50 ({norm_exact_acc:.2f}%) Exact | {norm_cer:.2f}% CER")
    print("=" * 76)

if __name__ == "__main__":
    evaluate()
