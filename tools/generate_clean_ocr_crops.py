"""
Regenerate 50 clean real Pakistani plate crops with polygon-safe coordinates,
execute EasyOCR on each to provide preliminary predictions, and output
both a CSV manifest and an interactive HTML review tool.
"""

import csv
import json
from pathlib import Path
import cv2
import numpy as np
from loguru import logger

from module2_vehicle.plate_ocr import PlateOCR
from tools.recompute_plate_localization import parse_yolo_label, yolo_to_xyxy


def main():
    root = Path(__file__).resolve().parent.parent
    ds_a_valid = root / "data" / "pakistan_plates" / "dataset_a" / "valid"
    ds_a_test = root / "data" / "pakistan_plates" / "dataset_a" / "test"
    crop_dir = root / "data" / "pakistan_plates" / "verified_crops"
    crop_dir.mkdir(parents=True, exist_ok=True)

    manifest_csv = root / "data" / "pakistan_plates" / "ocr_annotation_manifest.csv"

    # Collect images and labels from Dataset A (outdoor road vehicle scenes)
    samples = []
    for split_dir in [ds_a_test, ds_a_valid]:
        img_dir = split_dir / "images"
        lbl_dir = split_dir / "labels"
        if not img_dir.exists():
            continue
        for img_p in sorted(img_dir.glob("*.*")):
            if img_p.suffix.lower() not in [".jpg", ".jpeg", ".png"]:
                continue
            lbl_p = lbl_dir / f"{img_p.stem}.txt"
            if not lbl_p.exists():
                continue
            boxes = []
            for line in lbl_p.read_text(encoding="utf-8").splitlines():
                parsed = parse_yolo_label(line)
                if parsed:
                    boxes.append(parsed)
            if boxes:
                samples.append({
                    "image_path": img_p,
                    "boxes": boxes,
                    "split": split_dir.name
                })

    print(f"Total available annotated scenes in Dataset A test+valid: {len(samples)}")
    subset = samples[:50]

    # Initialize EasyOCR
    ocr = PlateOCR(gpu=True)

    manifest_rows = []
    html_items = []

    for idx, s in enumerate(subset):
        img = cv2.imread(str(s["image_path"]))
        if img is None:
            continue
        h, w = img.shape[:2]

        box = yolo_to_xyxy(s["boxes"][0], w, h)
        # Add 5% padding if within image bounds
        pad_x = int((box[2] - box[0]) * 0.05)
        pad_y = int((box[3] - box[1]) * 0.05)
        x1 = max(0, box[0] - pad_x)
        y1 = max(0, box[1] - pad_y)
        x2 = min(w, box[2] + pad_x)
        y2 = min(h, box[3] + pad_y)

        crop = img[y1:y2, x1:x2]
        if crop.size == 0 or crop.shape[0] < 8 or crop.shape[1] < 8:
            continue

        crop_fname = f"crop_sample_{idx:03d}_{s['image_path'].stem}.jpg"
        crop_path = crop_dir / crop_fname
        cv2.imwrite(str(crop_path), crop)

        res = ocr.read(crop)
        pred_text = res.clean_text or res.raw_text or ""
        conf = round(float(res.confidence), 3)

        manifest_rows.append({
            "sample_id": idx + 1,
            "source_image": s["image_path"].name,
            "crop_relative_path": f"data/pakistan_plates/verified_crops/{crop_fname}",
            "plate_bbox_xyxy": f"[{x1},{y1},{x2},{y2}]",
            "easyocr_predicted_text": pred_text,
            "easyocr_confidence": conf,
            "human_verified_gt_text": "",
            "jurisdiction_province": "",
            "verified_status": "PENDING_HUMAN_REVIEW",
        })

    # Write manifest CSV
    fieldnames = [
        "sample_id", "source_image", "crop_relative_path", "plate_bbox_xyxy",
        "easyocr_predicted_text", "easyocr_confidence", "human_verified_gt_text",
        "jurisdiction_province", "verified_status"
    ]
    with open(manifest_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(manifest_rows)

    print(f"[SUCCESS] Updated {len(manifest_rows)} clean crops in {manifest_csv}")


if __name__ == "__main__":
    main()
