"""
Generate visual proof of label alignment and polygon-to-bbox conversion.
Renders 8 random annotated sample images from Dataset A and Dataset B with green bounding boxes and label text.
Saves verification images to docs/report_assets/m03b_label_verification/
"""
from pathlib import Path
import random
import cv2
import numpy as np

def parse_box(parts):
    coords = [float(x) for x in parts[1:]]
    if len(coords) == 4:
        xc, yc, w, h = coords
        return xc, yc, w, h, "bbox"
    elif len(coords) >= 6:
        xs = coords[0::2]
        ys = coords[1::2]
        xmin, xmax = min(xs), max(xs)
        ymin, ymax = min(ys), max(ys)
        return (xmin + xmax) / 2, (ymin + ymax) / 2, xmax - xmin, ymax - ymin, "polygon_converted"
    return None

def render_sample_annotations():
    random.seed(42)
    root = Path("D:/MZHR_FYP/INTELLIGATE")
    out_dir = root / "docs" / "report_assets" / "m03b_label_verification"
    out_dir.mkdir(parents=True, exist_ok=True)

    samples = [
        ("dataset_a", "train", 2),
        ("dataset_a", "valid", 2),
        ("dataset_b", "train", 2),
        ("dataset_b", "valid", 2),
    ]

    saved_images = []

    for ds, split, count in samples:
        img_dir = root / "data" / "pakistan_plates" / ds / split / "images"
        lbl_dir = root / "data" / "pakistan_plates" / ds / split / "labels"
        img_files = sorted(list(img_dir.glob("*.jpg")))
        chosen = random.sample(img_files, min(count, len(img_files)))

        for idx, img_path in enumerate(chosen):
            lbl_path = lbl_dir / f"{img_path.stem}.txt"
            if not lbl_path.exists():
                continue
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            h, w = img.shape[:2]

            lines = lbl_path.read_text(encoding="utf-8").strip().splitlines()
            for line in lines:
                parts = line.strip().split()
                if not parts:
                    continue
                res = parse_box(parts)
                if not res:
                    continue
                xc, yc, bw, bh, ltype = res
                x1 = int((xc - bw / 2) * w)
                y1 = int((yc - bh / 2) * h)
                x2 = int((xc + bw / 2) * w)
                y2 = int((yc + bh / 2) * h)

                # Draw green bounding box
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                tag = f"plate ({ltype})"
                cv2.putText(img, tag, (x1, max(15, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            out_name = f"verify_{ds}_{split}_{idx+1}_{img_path.stem[:15]}.jpg"
            out_p = out_dir / out_name
            cv2.imwrite(str(out_p), img)
            saved_images.append(str(out_p))
            print(f"  Saved verification sample: {out_p.name}")

    print(f"\nSuccessfully generated {len(saved_images)} visual label verification samples.")

if __name__ == "__main__":
    render_sample_annotations()
