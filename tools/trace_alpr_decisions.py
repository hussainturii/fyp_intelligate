from pathlib import Path
import sys
root = Path("D:/MZHR_FYP/INTELLIGATE")
if str(root) not in sys.path:
    sys.path.insert(0, str(root))

import cv2
import numpy as np
import torch
from ultralytics import YOLO

from module2_vehicle.plate_detector import PlateDetector
from module2_vehicle.plate_ocr import PlateOCR
from module2_vehicle.database.vehicle_db import VehicleDatabase

def main():
    root = Path("D:/MZHR_FYP/INTELLIGATE")
    test_dir_a = root / "data/pakistan_plates/combined_clean/test_a/images"
    test_dir_b = root / "data/pakistan_plates/combined_clean/test_b/images"
    test_images = sorted(list(test_dir_a.glob("*.*"))) + sorted(list(test_dir_b.glob("*.*")))

    vehicle_model = YOLO("yolov8n.pt")
    yolo_det = PlateDetector(device="cuda:0", model_path="models/plate_detector_yolov8n.pt")
    ocr = PlateOCR(languages=["en"], gpu=True, n_reads=1, min_confidence=0.55)
    vdb = VehicleDatabase(str(root / "logs/cpu_vehicles.h5"), str(root / "logs/cpu_unknown_vehicles.h5"))

    print(f"Total test images: {len(test_images)}")
    allowed_records = []
    review_records = []

    for img_p in test_images:
        img = cv2.imread(str(img_p))
        if img is None:
            continue
        h, w = img.shape[:2]

        veh_res = vehicle_model.predict(img, imgsz=640, conf=0.35, classes=[2, 3, 5, 7], device="cuda:0", verbose=False)[0]
        vehicle_boxes = [tuple(map(int, b.xyxy[0].cpu().tolist())) for b in veh_res.boxes]

        search_crops = []
        if vehicle_boxes:
            for vx1, vy1, vx2, vy2 in vehicle_boxes:
                cvx1, cvy1 = max(0, vx1), max(0, vy1)
                cvx2, cvy2 = min(w, vx2), min(h, vy2)
                if cvx2 > cvx1 and cvy2 > cvy1:
                    search_crops.append(img[cvy1:cvy2, cvx1:cvx2])
        else:
            search_crops.append(img)

        found_plate = None
        for crop in search_crops:
            plates = yolo_det.detect(crop)
            if plates:
                found_plate = plates[0]
                break

        if found_plate is not None and found_plate.crop.size > 0:
            ocr_res = ocr.read(found_plate.crop)
            raw_txt = ocr_res.clean_text.upper()
            clean_plate = "".join(c for c in raw_txt if c.isalnum())
            ocr_conf = float(ocr_res.confidence)

            if clean_plate and ocr_conf >= 0.55:
                match = vdb.lookup_by_plate(clean_plate)
                if match and match.get("method") == "exact_plate":
                    if ocr_conf >= 0.85:
                        allowed_records.append({
                            "file": img_p.name,
                            "plate": clean_plate,
                            "conf": ocr_conf,
                            "owner": match.get("owner_name")
                        })
                    else:
                        review_records.append({
                            "file": img_p.name,
                            "plate": clean_plate,
                            "conf": ocr_conf,
                            "reason": "low_ocr_conf",
                            "owner": match.get("owner_name")
                        })
                elif ocr_conf < 0.80:
                    review_records.append({
                        "file": img_p.name,
                        "plate": clean_plate,
                        "conf": ocr_conf,
                        "reason": "unregistered_uncertain"
                    })

    print(f"\nTotal ALLOWED records: {len(allowed_records)}")
    for r in allowed_records:
        print("  ALLOWED:", r)

    print(f"\nTotal REVIEW records: {len(review_records)}")
    for r in review_records[:10]:
        print("  REVIEW:", r)

if __name__ == "__main__":
    main()
