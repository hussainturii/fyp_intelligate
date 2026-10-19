"""
Interactive CLI workflow for human review and transcription of the 50 Pakistani plate crops.
Allows the human reviewer to:
- Inspect crop metadata, bbox, and EasyOCR guess
- Type verified ground-truth text
- Accept or reject EasyOCR suggestion
- Mark sample as AMBIGUOUS or UNREADABLE explicitly
- Set province / jurisdiction
- Save progress anytime to data/pakistan_plates/ocr_annotation_manifest.csv
"""

import csv
import sys
from pathlib import Path


MANIFEST_PATH = Path("data/pakistan_plates/ocr_annotation_manifest.csv")


def load_manifest():
    if not MANIFEST_PATH.exists():
        print(f"Error: {MANIFEST_PATH} not found.")
        sys.exit(1)
    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fieldnames = reader.fieldnames
    return rows, fieldnames


def save_manifest(rows, fieldnames):
    with open(MANIFEST_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"[SAVED] Manifest saved ({len(rows)} entries) -> {MANIFEST_PATH}\n")


def interactive_review():
    rows, fieldnames = load_manifest()
    total = len(rows)
    print("=" * 70)
    print("   INTELLIGATE — HUMAN-IN-THE-LOOP LICENSE PLATE OCR ANNOTATION")
    print("=" * 70)
    print("Commands:")
    print("  [Enter]           - Accept EasyOCR prediction as Ground Truth (if verified)")
    print("  <text>            - Enter custom verified plate text (e.g. LEA-1234)")
    print("  u                 - Mark as UNREADABLE (too blurry / occluded)")
    print("  a                 - Mark as AMBIGUOUS (disputed characters)")
    print("  s                 - Skip this sample (leave pending)")
    print("  q                 - Save & Quit")
    print("=" * 70)

    for i, row in enumerate(rows):
        sample_id = row.get("sample_id")
        crop_path = row.get("crop_relative_path")
        pred_text = row.get("easyocr_predicted_text", "")
        conf = row.get("easyocr_confidence", "")
        curr_gt = row.get("human_verified_gt_text", "")
        status = row.get("verified_status", "PENDING_HUMAN_REVIEW")

        print(f"\n[Sample {sample_id}/{total}] {crop_path}")
        print(f"  EasyOCR Prediction : '{pred_text}' (Confidence: {conf})")
        print(f"  Current Status     : {status} | Current GT: '{curr_gt}'")

        choice = input("  Enter GT plate text (or [Enter] to accept, u=unreadable, a=ambiguous, s=skip, q=quit): ").strip()

        if choice.lower() == 'q':
            save_manifest(rows, fieldnames)
            print("Exiting review tool.")
            return
        elif choice.lower() == 's':
            print("  Skipped.")
            continue
        elif choice.lower() == 'u':
            row["human_verified_gt_text"] = "UNREADABLE"
            row["verified_status"] = "UNREADABLE"
            print("  Marked as UNREADABLE.")
        elif choice.lower() == 'a':
            row["human_verified_gt_text"] = "AMBIGUOUS"
            row["verified_status"] = "AMBIGUOUS"
            print("  Marked as AMBIGUOUS.")
        elif choice == "":
            if pred_text:
                row["human_verified_gt_text"] = pred_text
                row["verified_status"] = "HUMAN_VERIFIED"
                print(f"  Accepted EasyOCR '{pred_text}' as Ground Truth.")
            else:
                row["human_verified_gt_text"] = "UNREADABLE"
                row["verified_status"] = "UNREADABLE"
                print("  Empty text -> Marked as UNREADABLE.")
        else:
            row["human_verified_gt_text"] = choice.upper()
            row["verified_status"] = "HUMAN_VERIFIED"
            print(f"  Saved GT: '{choice.upper()}'.")

        # Optional province query if verified
        if row["verified_status"] == "HUMAN_VERIFIED" and not row.get("jurisdiction_province"):
            prov = input("  Jurisdiction province (Punjab/Sindh/ICT/KP/Balochistan/Enter to skip): ").strip()
            if prov:
                row["jurisdiction_province"] = prov.title()

    save_manifest(rows, fieldnames)
    print("All samples reviewed!")


if __name__ == "__main__":
    interactive_review()
