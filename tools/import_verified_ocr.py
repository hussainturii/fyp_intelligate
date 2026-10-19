"""
tools/import_verified_ocr.py — Import and Validate Human-Verified Pakistani Plate OCR Dataset
==============================================================================================
Validates the human-verified OCR manifest from Downloads, checks integrity against existing
crops in data/pakistan_plates/verified_crops/, and exports a versioned copy into
data/pakistan_plates/annotations/ocr_annotation_manifest_verified.csv.
"""

import csv
import json
import os
import shutil
from pathlib import Path
from loguru import logger

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DOWNLOADS_CSV = Path(r"C:\Users\user\Downloads\ocr_annotation_manifest_verified.csv")
TARGET_DIR = PROJECT_ROOT / "data" / "pakistan_plates" / "annotations"
TARGET_CSV = TARGET_DIR / "ocr_annotation_manifest_verified.csv"
CROPS_DIR = PROJECT_ROOT / "data" / "pakistan_plates" / "verified_crops"

def validate_and_import():
    if not DOWNLOADS_CSV.exists():
        raise FileNotFoundError(f"Downloads manifest not found: {DOWNLOADS_CSV}")

    TARGET_DIR.mkdir(parents=True, exist_ok=True)

    with open(DOWNLOADS_CSV, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        rows = list(reader)

    logger.info(f"Loaded {len(rows)} records from {DOWNLOADS_CSV}")
    assert len(rows) == 50, f"Expected exactly 50 records, found {len(rows)}"

    sample_ids = set()
    source_images = set()
    missing_crops = []
    unverified_count = 0
    overwritten_by_easyocr = 0

    validated_records = []

    for idx, r in enumerate(rows):
        s_id = r["sample_id"].strip()
        assert s_id, f"Row {idx} has empty sample_id"
        assert s_id not in sample_ids, f"Duplicate sample_id found: {s_id}"
        sample_ids.add(s_id)

        crop_rel = r["crop_relative_path"].strip()
        crop_path = PROJECT_ROOT / crop_rel
        if not crop_path.exists():
            missing_crops.append(crop_rel)

        gt_text = r["human_verified_gt_text"].strip()
        pred_text = r["easyocr_predicted_text"].strip()
        v_status = r["verified_status"].strip()

        assert v_status in {"HUMAN_VERIFIED", "FLAGGED_FOR_REVIEW"}, f"Invalid verified_status: {v_status}"
        assert gt_text, f"Sample {s_id} has empty human_verified_gt_text"

        province = r["jurisdiction_province"].strip()
        assert province in {"Punjab", "ICT", "Sindh", "KPK", "Balochistan", "Unknown", "Custom"}, f"Unexpected province: {province}"

        validated_records.append(r)

    logger.info(f"Integrity check passed:")
    logger.info(f"  - Total records: {len(validated_records)}")
    logger.info(f"  - Unique sample IDs: {len(sample_ids)}")
    logger.info(f"  - Missing crop images: {len(missing_crops)}")
    if missing_crops:
        for m in missing_crops[:5]:
            logger.error(f"    Missing: {m}")
        raise FileNotFoundError(f"Found {len(missing_crops)} missing crop images!")

    # Copy verified CSV to target destination preserving original
    shutil.copy2(DOWNLOADS_CSV, TARGET_CSV)
    logger.info(f"Successfully copied manifest to {TARGET_CSV}")

    # Output summary manifest json
    summary_path = TARGET_DIR / "ocr_verification_summary.json"
    provinces = {}
    for r in validated_records:
        p = r["jurisdiction_province"]
        provinces[p] = provinces.get(p, 0) + 1

    summary = {
        "source": str(DOWNLOADS_CSV),
        "target": str(TARGET_CSV),
        "total_verified_samples": len(validated_records),
        "provinces_distribution": provinces,
        "missing_crops_count": len(missing_crops),
        "integrity_verified": True
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    logger.info(f"Wrote summary manifest to {summary_path}")

if __name__ == "__main__":
    validate_and_import()
