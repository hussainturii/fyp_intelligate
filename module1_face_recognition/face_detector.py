"""
face_detector.py — Module 1 / FE-1
====================================
MTCNN-based real-time face detection with alignment.
Outputs aligned, cropped face tensors ready for the ViT recognizer.

Author : Hussain Raza (SP23-BAI-057)
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import cv2
import numpy as np
import torch
from facenet_pytorch import MTCNN
from loguru import logger
from PIL import Image


@dataclass
class DetectedFace:
    """Holds a single detected face and its metadata."""
    bbox: Tuple[int, int, int, int]    # (x1, y1, x2, y2) in pixels
    confidence: float                  # Detection confidence 0-1
    landmarks: np.ndarray              # 5 facial landmarks (eyes, nose, mouth)
    aligned_face: np.ndarray           # RGB numpy array (224, 224, 3) for ViT
    aligned_pil: Image.Image           # PIL version for torchvision transforms
    frame_id: int = 0
    timestamp: float = field(default_factory=time.time)

    @property
    def center(self) -> Tuple[int, int]:
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) // 2, (y1 + y2) // 2)

    @property
    def area(self) -> int:
        x1, y1, x2, y2 = self.bbox
        return (x2 - x1) * (y2 - y1)


class FaceDetector:
    """
    Real-time face detector using MTCNN (Multi-Task Cascaded CNN).

    Pipeline:
        BGR frame → RGB → MTCNN → aligned 224×224 face crops

    Usage::

        detector = FaceDetector(device="cuda")
        faces = detector.detect(bgr_frame)
        for face in faces:
            print(face.confidence, face.bbox)
    """

    def __init__(
        self,
        device: str = "cuda",
        min_face_size: int = 60,
        thresholds: Tuple[float, float, float] = (0.6, 0.7, 0.7),
        scale_factor: float = 0.709,
        margin: int = 20,
        image_size: int = 224,
        keep_all: bool = True,
    ) -> None:
        """
        Args:
            device        : 'cuda' or 'cpu'
            min_face_size : Minimum face pixel size to detect
            thresholds    : MTCNN P/R/O-Net thresholds
            scale_factor  : Image pyramid scale factor
            margin        : Pixel margin added around detected face
            image_size    : Output crop size (height=width) for ViT
            keep_all      : If True, return all faces; else only largest
        """
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.image_size = image_size
        self.margin = margin
        self.keep_all = keep_all
        self._frame_counter = 0

        logger.info(f"[FaceDetector] Initializing MTCNN on {self.device}")

        self.mtcnn = MTCNN(
            image_size=image_size,
            margin=margin,
            min_face_size=min_face_size,
            thresholds=list(thresholds),
            factor=scale_factor,
            keep_all=keep_all,
            device=self.device,
            select_largest=False,   # We handle selection ourselves
        )

        logger.success("[FaceDetector] MTCNN ready.")

    # ─────────────────────────────────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────────────────────────────────

    def detect(self, frame_bgr: np.ndarray) -> List[DetectedFace]:
        """
        Detect all faces in a BGR OpenCV frame.

        Returns a list of DetectedFace objects, sorted by confidence descending.
        Returns [] if no face found.
        """
        self._frame_counter += 1

        if frame_bgr is None or frame_bgr.size == 0:
            logger.warning("[FaceDetector] Received empty frame.")
            return []

        # Convert BGR → RGB (MTCNN expects RGB PIL or numpy)
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(frame_rgb)

        # Run MTCNN detection
        try:
            boxes, probs, landmarks = self.mtcnn.detect(pil_img, landmarks=True)
        except Exception as e:
            logger.error(f"[FaceDetector] MTCNN error: {e}")
            return []

        if boxes is None or len(boxes) == 0:
            return []

        detected: List[DetectedFace] = []
        h, w = frame_bgr.shape[:2]

        for i, (box, prob, lm) in enumerate(zip(boxes, probs, landmarks)):
            if prob is None or prob < 0.5:
                continue

            x1, y1, x2, y2 = [int(v) for v in box]
            # Clamp to frame boundaries
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)

            if x2 <= x1 or y2 <= y1:
                continue

            # Crop and resize for ViT
            face_crop = frame_rgb[y1:y2, x1:x2]
            aligned = cv2.resize(face_crop, (self.image_size, self.image_size))
            aligned_pil = Image.fromarray(aligned)

            detected.append(DetectedFace(
                bbox=(x1, y1, x2, y2),
                confidence=float(prob),
                landmarks=lm.astype(np.int32),
                aligned_face=aligned,
                aligned_pil=aligned_pil,
                frame_id=self._frame_counter,
            ))

        # Sort by confidence descending
        detected.sort(key=lambda f: f.confidence, reverse=True)
        return detected

    def draw_detections(
        self,
        frame_bgr: np.ndarray,
        faces: List[DetectedFace],
        show_landmarks: bool = True,
        show_confidence: bool = True,
    ) -> np.ndarray:
        """
        Draw bounding boxes and landmarks on a BGR frame.
        Returns a copy of the frame with annotations.
        """
        vis = frame_bgr.copy()

        for face in faces:
            x1, y1, x2, y2 = face.bbox
            color = (0, 255, 0)  # Green for detected face

            # Bounding box
            cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)

            if show_confidence:
                label = f"Face: {face.confidence:.2f}"
                cv2.putText(vis, label, (x1, y1 - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1)

            if show_landmarks and face.landmarks is not None:
                # 5 landmarks: left_eye, right_eye, nose, left_mouth, right_mouth
                lm_colors = [
                    (255, 0, 0),    # left eye - blue
                    (0, 0, 255),    # right eye - red
                    (0, 255, 255),  # nose - yellow
                    (255, 0, 255),  # left mouth - magenta
                    (0, 165, 255),  # right mouth - orange
                ]
                for pt, c in zip(face.landmarks, lm_colors):
                    cv2.circle(vis, tuple(pt), 3, c, -1)

        return vis

    def get_largest_face(self, faces: List[DetectedFace]) -> Optional[DetectedFace]:
        """Return the largest (by area) detected face, or None."""
        if not faces:
            return None
        return max(faces, key=lambda f: f.area)

    # ─────────────────────────────────────────────────────────────────────────
    # Properties
    # ─────────────────────────────────────────────────────────────────────────

    @property
    def frame_count(self) -> int:
        return self._frame_counter
