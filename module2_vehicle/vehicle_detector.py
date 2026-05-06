"""
vehicle_detector.py — Module 2 / FE-1 (detection stage)
=========================================================
YOLOv8-based vehicle detector. Detects cars, trucks, buses,
and motorcycles in real-time camera frames.

Uses ultralytics YOLOv8n (nano) pretrained on COCO — downloads
automatically on first run (~6 MB). For better accuracy swap to
yolov8s or yolov8m at the cost of speed.

COCO vehicle class IDs:
    2  → car
    3  → motorcycle
    5  → bus
    7  → truck

Author: Hussain Raza (SP23-BAI-057)
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import cv2
import numpy as np
from loguru import logger

# Ultralytics YOLOv8
try:
    from ultralytics import YOLO
    _YOLO_OK = True
except ImportError:
    _YOLO_OK = False
    logger.error("[VehicleDetector] ultralytics not installed. Run: pip install ultralytics")


# COCO class IDs we care about
VEHICLE_CLASSES = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}


@dataclass
class DetectedVehicle:
    """One detected vehicle in a frame."""
    bbox:        Tuple[int, int, int, int]   # (x1, y1, x2, y2)
    class_id:    int
    class_name:  str                          # 'car' | 'truck' | 'bus' | 'motorcycle'
    confidence:  float
    crop:        np.ndarray                   # BGR crop of the vehicle
    frame_id:    int = 0
    timestamp:   float = field(default_factory=time.time)
    track_id:    Optional[int] = None        # filled when tracker is active

    @property
    def area(self) -> int:
        x1, y1, x2, y2 = self.bbox
        return (x2 - x1) * (y2 - y1)

    @property
    def center(self) -> Tuple[int, int]:
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) // 2, (y1 + y2) // 2)


class VehicleDetector:
    """
    Real-time vehicle detector using YOLOv8.

    Downloads yolov8n.pt automatically on first use if not present.

    Usage::

        detector = VehicleDetector()
        vehicles = detector.detect(frame_bgr)
        for v in vehicles:
            print(v.class_name, v.confidence, v.bbox)
    """

    # Colours per vehicle class for visualisation
    CLASS_COLORS = {
        "car":        (0, 255, 0),     # green
        "truck":      (0, 165, 255),   # orange
        "bus":        (255, 0, 0),     # blue
        "motorcycle": (255, 0, 255),   # magenta
    }

    def __init__(
        self,
        model_name: str = "yolov8n.pt",
        confidence_threshold: float = 0.40,
        iou_threshold: float = 0.45,
        device: str = "cpu",
        use_tracker: bool = False,
    ) -> None:
        """
        Args:
            model_name            : YOLOv8 weights file (auto-downloaded if missing)
            confidence_threshold  : Minimum detection confidence
            iou_threshold         : NMS IoU threshold
            device                : 'cuda' or 'cpu'
            use_tracker           : Enable ByteTrack for persistent vehicle IDs
        """
        if not _YOLO_OK:
            raise RuntimeError("Install ultralytics: pip install ultralytics")

        self.conf      = confidence_threshold
        self.iou       = iou_threshold
        self.device    = device
        self.use_tracker = use_tracker
        self._frame_id = 0

        logger.info(f"[VehicleDetector] Loading {model_name} on {device}...")
        self.model = YOLO(model_name)
        self.model.to(device)
        logger.success(f"[VehicleDetector] YOLOv8 ready — conf={confidence_threshold}")

    # ─────────────────────────────────────────────────────────────────────────

    def detect(self, frame_bgr: np.ndarray) -> List[DetectedVehicle]:
        """
        Detect all vehicles in a BGR frame.

        Returns list of DetectedVehicle, sorted by area descending
        (largest vehicle first — most likely the gate subject).
        """
        self._frame_id += 1
        if frame_bgr is None or frame_bgr.size == 0:
            return []

        # Run YOLOv8 inference
        if self.use_tracker:
            results = self.model.track(
                frame_bgr, persist=True,
                conf=self.conf, iou=self.iou,
                classes=list(VEHICLE_CLASSES.keys()),
                verbose=False,
            )
        else:
            results = self.model(
                frame_bgr,
                conf=self.conf, iou=self.iou,
                classes=list(VEHICLE_CLASSES.keys()),
                verbose=False,
            )

        vehicles: List[DetectedVehicle] = []
        h, w = frame_bgr.shape[:2]

        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue

            for i, box in enumerate(boxes):
                cls_id = int(box.cls[0])
                if cls_id not in VEHICLE_CLASSES:
                    continue

                conf  = float(box.conf[0])
                x1, y1, x2, y2 = [int(v) for v in box.xyxy[0]]
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)

                if x2 <= x1 or y2 <= y1:
                    continue

                crop = frame_bgr[y1:y2, x1:x2].copy()

                track_id = None
                if self.use_tracker and boxes.id is not None:
                    track_id = int(boxes.id[i])

                vehicles.append(DetectedVehicle(
                    bbox=(x1, y1, x2, y2),
                    class_id=cls_id,
                    class_name=VEHICLE_CLASSES[cls_id],
                    confidence=conf,
                    crop=crop,
                    frame_id=self._frame_id,
                    track_id=track_id,
                ))

        vehicles.sort(key=lambda v: v.area, reverse=True)
        return vehicles

    def draw_detections(
        self,
        frame: np.ndarray,
        vehicles: List[DetectedVehicle],
    ) -> np.ndarray:
        """Draw bounding boxes and labels on frame. Returns annotated copy."""
        vis = frame.copy()
        for v in vehicles:
            color = self.CLASS_COLORS.get(v.class_name, (0, 255, 0))
            x1, y1, x2, y2 = v.bbox
            cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)

            label = f"{v.class_name} {v.confidence:.2f}"
            if v.track_id is not None:
                label += f" #{v.track_id}"

            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
            cv2.rectangle(vis, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)
            cv2.putText(vis, label, (x1 + 2, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
        return vis

    def get_primary_vehicle(
        self, vehicles: List[DetectedVehicle]
    ) -> Optional[DetectedVehicle]:
        """Return the largest (most prominent) vehicle, or None."""
        return vehicles[0] if vehicles else None
