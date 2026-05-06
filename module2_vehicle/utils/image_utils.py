"""
image_utils.py — Module 2 image preprocessing helpers
=======================================================
Plate preprocessing pipeline to maximise EasyOCR accuracy.

Author: Hussain Raza (SP23-BAI-057)
"""

from __future__ import annotations
import cv2
import numpy as np


def preprocess_plate(plate_bgr: np.ndarray, target_h: int = 64) -> np.ndarray:
    """
    Preprocess a license plate crop for EasyOCR.

    Pipeline:
        Resize (keep aspect) → Grayscale → CLAHE (contrast)
        → Denoise → Adaptive threshold → Deskew → Return

    Args:
        plate_bgr : BGR crop of the license plate
        target_h  : Target height in pixels (width scales proportionally)

    Returns:
        Preprocessed BGR image ready for OCR
    """
    if plate_bgr is None or plate_bgr.size == 0:
        return plate_bgr

    # 1. Resize to fixed height (keeps aspect ratio)
    h, w = plate_bgr.shape[:2]
    scale = target_h / h
    resized = cv2.resize(plate_bgr, (int(w * scale), target_h),
                         interpolation=cv2.INTER_CUBIC)

    # 2. Convert to grayscale
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

    # 3. CLAHE — contrast limited adaptive histogram equalisation
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
    gray = clahe.apply(gray)

    # 4. Gaussian denoise
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    # 5. Deskew — correct slight rotation using moments
    gray = _deskew(gray)

    # 6. Convert back to BGR so EasyOCR (which expects colour) is happy
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def _deskew(gray: np.ndarray) -> np.ndarray:
    """Correct slight skew in plate image using image moments."""
    try:
        coords = np.column_stack(np.where(gray < 128))
        if len(coords) < 5:
            return gray
        angle = cv2.minAreaRect(coords)[-1]
        if angle < -45:
            angle = -(90 + angle)
        else:
            angle = -angle
        if abs(angle) < 1.0:   # skip tiny corrections
            return gray
        h, w = gray.shape
        M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
        return cv2.warpAffine(gray, M, (w, h),
                              flags=cv2.INTER_CUBIC,
                              borderMode=cv2.BORDER_REPLICATE)
    except Exception:
        return gray


def draw_vehicle_overlay(
    frame: np.ndarray,
    bbox: tuple,
    label: str,
    confidence: float,
    color: tuple = (0, 255, 0),
    plate_text: str = "",
) -> np.ndarray:
    """
    Draw a vehicle detection box with label and optional plate text on a frame.
    Returns the frame in-place (also returns it for chaining).
    """
    x1, y1, x2, y2 = [int(v) for v in bbox]
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    # Label background
    text = f"{label} {confidence:.2f}"
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
    cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)
    cv2.putText(frame, text, (x1 + 2, y1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)

    # Plate text below box
    if plate_text:
        plate_label = f"PLATE: {plate_text}"
        cv2.putText(frame, plate_label, (x1, y2 + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)

    return frame
