"""
plate_ocr.py — Module 2 / FE-1 (OCR stage)
============================================
EasyOCR-based license plate text extraction with preprocessing
and post-processing to clean and validate plate strings.

Supports:
    - Pakistani plates  (ABC-1234, ABC-123, LEA-1234 etc.)
    - Generic plates    (any alphanumeric string)
    - Multiple reads + majority voting for higher accuracy

Author: Hussain Raza (SP23-BAI-057)
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import cv2
import numpy as np
from loguru import logger

try:
    import easyocr
    _EASYOCR_OK = True
except ImportError:
    _EASYOCR_OK = False
    logger.error("[PlateOCR] easyocr not installed. Run: pip install easyocr")

from .utils.image_utils import preprocess_plate


@dataclass
class OCRResult:
    """Result from reading one license plate."""
    raw_text:    str            # Raw EasyOCR output
    clean_text:  str            # Cleaned / normalised plate string
    confidence:  float          # EasyOCR confidence 0–1
    is_valid:    bool           # Passed format validation
    bbox:        Optional[Tuple] = None   # Text bbox within plate crop


# ── Pakistani plate format patterns ──────────────────────────────────────────
# Standard:  ABC-1234   (3 letters, dash, 4 digits)
# Old style: ABC-123    (3 letters, dash, 3 digits)
# LEA/Govt:  LEA-1234 / AJK-1234 etc.
_PLATE_PATTERNS = [
    re.compile(r'^[A-Z]{2,4}[-\s]?\d{3,4}$'),   # ABC-1234 / AB-1234
    re.compile(r'^[A-Z]{1,3}\d{2,4}[A-Z]{0,2}$'), # Generic international
]

# Common OCR character confusions on plates
_OCR_FIXES = {
    'O': '0', 'I': '1', 'S': '5', 'B': '8',
    'G': '6', 'Z': '2', 'Q': '0',
}


class PlateOCR:
    """
    License plate text reader using EasyOCR.

    Reads the same plate crop multiple times with different
    preprocessing variants and takes the majority vote for
    maximum accuracy.

    Usage::

        ocr = PlateOCR()
        result = ocr.read(plate_crop_bgr)
        print(result.clean_text, result.confidence)
    """

    def __init__(
        self,
        languages: List[str] = None,
        gpu: bool = False,
        n_reads: int = 3,
        min_confidence: float = 0.30,
    ) -> None:
        """
        Args:
            languages      : EasyOCR language codes (default: English only)
            gpu            : Use GPU for OCR inference
            n_reads        : Number of preprocessing variants to try (voting)
            min_confidence : Minimum EasyOCR confidence to accept a read
        """
        if not _EASYOCR_OK:
            raise RuntimeError("Install easyocr: pip install easyocr")

        self.n_reads        = n_reads
        self.min_confidence = min_confidence
        langs = languages or ["en"]

        logger.info(f"[PlateOCR] Initialising EasyOCR (langs={langs}, gpu={gpu})...")
        self.reader = easyocr.Reader(langs, gpu=gpu, verbose=False)
        logger.success("[PlateOCR] EasyOCR ready.")

    # ─────────────────────────────────────────────────────────────────────────

    def read(self, plate_bgr: np.ndarray) -> OCRResult:
        """
        Read plate text from a BGR plate crop.

        Runs up to n_reads preprocessing variants and returns the
        highest-confidence valid result (or best raw result if none valid).

        Args:
            plate_bgr : BGR crop of the license plate

        Returns:
            OCRResult with clean_text and confidence
        """
        if plate_bgr is None or plate_bgr.size == 0:
            return OCRResult("", "", 0.0, False)

        # Generate preprocessing variants
        variants = self._make_variants(plate_bgr)
        candidates: List[OCRResult] = []

        for variant in variants:
            result = self._read_single(variant)
            if result.confidence >= self.min_confidence:
                candidates.append(result)

        if not candidates:
            return OCRResult("", "", 0.0, False)

        # Prefer valid-format plates; otherwise take highest confidence
        valid = [c for c in candidates if c.is_valid]
        if valid:
            best = max(valid, key=lambda r: r.confidence)
        else:
            best = max(candidates, key=lambda r: r.confidence)

        return best

    def read_with_voting(self, plate_bgr: np.ndarray, n_votes: int = 5) -> OCRResult:
        """
        Read plate text using majority voting across multiple reads.
        More accurate but slower. Use for static images.
        """
        results = [self.read(plate_bgr) for _ in range(n_votes)]
        texts = [r.clean_text for r in results if r.clean_text]
        if not texts:
            return OCRResult("", "", 0.0, False)

        # Most common text
        winner = Counter(texts).most_common(1)[0][0]
        # Average confidence of matching reads
        matching = [r for r in results if r.clean_text == winner]
        avg_conf = sum(r.confidence for r in matching) / len(matching)

        return OCRResult(
            raw_text=winner,
            clean_text=winner,
            confidence=avg_conf,
            is_valid=self._validate(winner),
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Internal helpers
    # ─────────────────────────────────────────────────────────────────────────

    def _read_single(self, image_bgr: np.ndarray) -> OCRResult:
        """Run EasyOCR on one image variant and return best text."""
        try:
            detections = self.reader.readtext(
                image_bgr,
                detail=1,
                paragraph=False,
                allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-",
            )
        except Exception as e:
            logger.debug(f"[PlateOCR] EasyOCR read error: {e}")
            return OCRResult("", "", 0.0, False)

        if not detections:
            return OCRResult("", "", 0.0, False)

        # Concatenate all detected text segments (a plate may read as 2 boxes)
        parts = []
        total_conf = 0.0
        best_bbox  = None

        for (bbox_pts, text, conf) in detections:
            if conf < self.min_confidence:
                continue
            parts.append(text.upper().strip())
            total_conf += conf
            if best_bbox is None:
                best_bbox = bbox_pts

        if not parts:
            return OCRResult("", "", 0.0, False)

        raw   = " ".join(parts)
        clean = self._clean(raw)
        avg_conf = total_conf / len(parts)

        return OCRResult(
            raw_text=raw,
            clean_text=clean,
            confidence=float(avg_conf),
            is_valid=self._validate(clean),
            bbox=best_bbox,
        )

    def _make_variants(self, plate_bgr: np.ndarray) -> List[np.ndarray]:
        """
        Generate multiple preprocessed versions of the plate for voting.
        Different conditions respond to different preprocessing.
        """
        variants = []

        # Variant 1: standard preprocessing
        variants.append(preprocess_plate(plate_bgr, target_h=64))

        # Variant 2: larger resize for small plates
        variants.append(preprocess_plate(plate_bgr, target_h=96))

        # Variant 3: inverted (white-on-dark plates)
        pp = preprocess_plate(plate_bgr, target_h=64)
        variants.append(cv2.bitwise_not(pp))

        return variants[:self.n_reads]

    @staticmethod
    def _clean(raw: str) -> str:
        """
        Normalise raw OCR output into a clean plate string.
            - Uppercase
            - Remove non-alphanumeric except dash
            - Apply common OCR character confusion fixes
            - Insert dash in Pakistani format if missing
        """
        text = raw.upper().strip()
        # Keep only alphanumeric + dash
        text = re.sub(r'[^A-Z0-9\-]', '', text)

        # Fix OCR confusions in digit positions (last 3-4 chars)
        if len(text) >= 4:
            letter_part = text[:-4] if len(text) > 4 else text[:-3]
            digit_part  = text[-4:] if len(text) > 4 else text[-3:]
            digit_part  = ''.join(_OCR_FIXES.get(c, c) for c in digit_part)
            text = letter_part + digit_part

        # Insert dash for Pakistani format: "ABC1234" → "ABC-1234"
        m = re.match(r'^([A-Z]{2,4})(\d{3,4})$', text)
        if m:
            text = m.group(1) + '-' + m.group(2)

        return text

    @staticmethod
    def _validate(text: str) -> bool:
        """Return True if text matches a known plate pattern."""
        if not text or len(text) < 4:
            return False
        return any(p.match(text) for p in _PLATE_PATTERNS)
