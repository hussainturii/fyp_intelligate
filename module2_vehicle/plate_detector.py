"""
plate_detector.py — Module 2 / FE-1 (plate localisation)
==========================================================
License plate localisation using a YOLOv8 model fine-tuned
specifically on license plates.

Model: keremberke/yolov8n-license-plate-detection
       (auto-downloaded from HuggingFace on first run, ~6 MB)

If the fine-tuned model is unavailable, falls back to a classical
OpenCV approach (edge detection + contour filtering) so the pipeline
never hard-fails.

Author: Hussain Raza (SP23-BAI-057)
"""

from __future__ import annotations

import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
from loguru import logger

try:
    from ultralytics import YOLO
    _YOLO_OK = True
except ImportError:
    _YOLO_OK = False

# ── HuggingFace plate detection model ────────────────────────────────────────
_PLATE_MODEL_URL = (
    "https://github.com/keremberke/awesome-yolov8-models/releases/download/"
    "v1.0.0/license_plate_detector.pt"
)
_PLATE_MODEL_PATH = Path(__file__).parent / "models" / "license_plate_detector.pt"


@dataclass
class DetectedPlate:
    """A single detected license plate region."""
    bbox:       Tuple[int, int, int, int]   # (x1, y1, x2, y2) in full frame coords
    confidence: float
    crop:       np.ndarray                  # BGR crop of plate
    crop_preprocessed: Optional[np.ndarray] = None   # set by PlateOCR
    method:     str = "yolo"               # 'yolo' | 'classical'


class PlateDetector:
    """
    License plate region detector.

    Tries a YOLOv8 plate model first; falls back to classical OpenCV
    contour detection if the model download fails.

    Usage::

        detector = PlateDetector()
        plates = detector.detect(vehicle_crop_bgr)
        for plate in plates:
            print(plate.confidence, plate.bbox)
    """

    def __init__(
        self,
        confidence_threshold: float = 0.40,
        device: str = "cpu",
        model_path: Optional[str] = None,
        allow_download: bool = True,
    ) -> None:
        self.conf   = confidence_threshold
        self.device = device
        self._model: Optional[YOLO] = None

        path = Path(model_path) if model_path else _PLATE_MODEL_PATH

        if _YOLO_OK and (path.exists() or allow_download):
            self._model = self._load_model(path)
        else:
            logger.warning("[PlateDetector] Using classical contour baseline (no local plate model).")

    # ─────────────────────────────────────────────────────────────────────────

    def detect(
        self,
        image_bgr: np.ndarray,
        offset: Tuple[int, int] = (0, 0),
    ) -> List[DetectedPlate]:
        """
        Detect license plates in an image (vehicle crop or full frame).

        Args:
            image_bgr : BGR image to search
            offset    : (x_off, y_off) if image is a crop — adds back to bbox
                        so coordinates are in full-frame space

        Returns:
            List of DetectedPlate sorted by confidence descending.
        """
        if image_bgr is None or image_bgr.size == 0:
            return []

        if self._model is not None:
            return self._detect_yolo(image_bgr, offset)
        else:
            return self._detect_classical(image_bgr, offset)

    # ─────────────────────────────────────────────────────────────────────────
    # YOLO detection
    # ─────────────────────────────────────────────────────────────────────────

    def _detect_yolo(
        self, image_bgr: np.ndarray, offset: Tuple[int, int]
    ) -> List[DetectedPlate]:
        ox, oy = offset
        results = self._model(image_bgr, conf=self.conf, verbose=False)
        plates: List[DetectedPlate] = []

        for result in results:
            if result.boxes is None:
                continue
            for box in result.boxes:
                conf = float(box.conf[0])
                x1, y1, x2, y2 = [int(v) for v in box.xyxy[0]]
                # Clamp to image
                h, w = image_bgr.shape[:2]
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)
                if x2 <= x1 or y2 <= y1:
                    continue
                crop = image_bgr[y1:y2, x1:x2].copy()
                plates.append(DetectedPlate(
                    bbox=(x1 + ox, y1 + oy, x2 + ox, y2 + oy),
                    confidence=conf,
                    crop=crop,
                    method="yolo",
                ))

        plates.sort(key=lambda p: p.confidence, reverse=True)
        return plates

    # ─────────────────────────────────────────────────────────────────────────
    # Classical OpenCV fallback
    # ─────────────────────────────────────────────────────────────────────────

    def _detect_classical(
        self, image_bgr: np.ndarray, offset: Tuple[int, int]
    ) -> List[DetectedPlate]:
        """
        Detect license plates using edge detection + contour filtering.
        Less accurate than YOLO but requires no model file.

        Heuristic: plates are roughly 2:1–5:1 aspect ratio rectangles
        with high edge density.
        """
        ox, oy = offset
        gray   = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        blur   = cv2.GaussianBlur(gray, (5, 5), 0)
        edges  = cv2.Canny(blur, 50, 200)

        # Morphological close to join nearby edges
        kernel  = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 3))
        closed  = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(
            closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        plates: List[DetectedPlate] = []
        h_img, w_img = image_bgr.shape[:2]

        for cnt in contours:
            x, y, w, hh = cv2.boundingRect(cnt)
            if w < 60 or hh < 15:
                continue
            aspect = w / hh
            if not (1.5 <= aspect <= 6.0):
                continue
            area_ratio = (w * hh) / (w_img * h_img)
            if not (0.005 <= area_ratio <= 0.25):
                continue

            x1, y1, x2, y2 = x, y, x + w, y + hh
            crop = image_bgr[y1:y2, x1:x2].copy()
            # Confidence proxy: edge density in ROI
            edge_density = float(np.mean(edges[y1:y2, x1:x2]) / 255.0)

            plates.append(DetectedPlate(
                bbox=(x1 + ox, y1 + oy, x2 + ox, y2 + oy),
                confidence=min(edge_density * 3, 0.95),
                crop=crop,
                method="classical",
            ))

        plates.sort(key=lambda p: p.confidence, reverse=True)
        return plates[:3]   # return top 3 candidates

    # ─────────────────────────────────────────────────────────────────────────
    # Model loading with auto-download
    # ─────────────────────────────────────────────────────────────────────────

    def _load_model(self, path: Path) -> Optional[YOLO]:
        """Load plate YOLO model, downloading if necessary."""
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            logger.info(f"[PlateDetector] Downloading plate detection model...")
            try:
                def _progress(block, bsize, total):
                    if total > 0:
                        print(f"\r  {min(block*bsize/total*100, 100):.1f}%",
                              end="", flush=True)
                urllib.request.urlretrieve(_PLATE_MODEL_URL, str(path), _progress)
                print()
                logger.success(f"[PlateDetector] Model saved to {path}")
            except Exception as e:
                logger.warning(
                    f"[PlateDetector] Download failed ({e}). "
                    "Using classical fallback."
                )
                return None

        try:
            model = YOLO(str(path))
            model.to(self.device)
            logger.success("[PlateDetector] Plate model loaded (YOLO).")
            return model
        except Exception as e:
            logger.warning(f"[PlateDetector] Failed to load model: {e}. Using classical fallback.")
            return None

    def draw_plates(
        self, frame: np.ndarray, plates: List[DetectedPlate]
    ) -> np.ndarray:
        """Draw plate bounding boxes on frame."""
        vis = frame.copy()
        for p in plates:
            x1, y1, x2, y2 = p.bbox
            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 255), 2)
            label = f"Plate {p.confidence:.2f} [{p.method}]"
            cv2.putText(vis, label, (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        return vis
