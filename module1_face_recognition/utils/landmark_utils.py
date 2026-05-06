"""
landmark_utils.py — MediaPipe Tasks API wrapper
================================================
Unified face landmark detector that works with MediaPipe 0.10+
(the new Tasks API). Auto-downloads the face_landmarker.task model
on first use.

This replaces the deprecated mp.solutions.face_mesh API.

Author : Hussain Raza (SP23-BAI-057)
"""

from __future__ import annotations

import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
from loguru import logger

# ── MediaPipe Tasks import ────────────────────────────────────────────────────
try:
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision as mp_vision
    _MP_TASKS_OK = True
except Exception as _mp_err:
    _MP_TASKS_OK = False
    logger.error(f"[LandmarkUtils] MediaPipe Tasks API not available: {_mp_err}")

# ── Model URL & local path ───────────────────────────────────────────────────
_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "face_landmarker/face_landmarker/float16/latest/face_landmarker.task"
)
_DEFAULT_MODEL_DIR = Path(__file__).parent.parent / "models"
_DEFAULT_MODEL_PATH = _DEFAULT_MODEL_DIR / "face_landmarker.task"


# ─────────────────────────────────────────────────────────────────────────────
# Result dataclass
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class NormalizedLandmark:
    """Single 3D face landmark, normalized 0–1."""
    x: float
    y: float
    z: float


@dataclass
class LandmarkResult:
    """
    Container for face landmark detection results.
    Mimics the old mp.solutions interface so detector code stays clean.
    """
    landmarks: List[NormalizedLandmark]   # 478 landmarks (468 face + 10 iris)
    width: int
    height: int

    def pixel_xy(self, idx: int) -> Tuple[int, int]:
        """Return (x, y) in pixel coordinates for landmark index."""
        lm = self.landmarks[idx]
        return (int(lm.x * self.width), int(lm.y * self.height))

    def z_value(self, idx: int) -> float:
        return self.landmarks[idx].z

    def z_array(self, indices: List[int]) -> np.ndarray:
        return np.array([self.landmarks[i].z for i in indices])

    def xy_array(self, indices: List[int]) -> List[Tuple[float, float]]:
        return [(self.landmarks[i].x * self.width,
                 self.landmarks[i].y * self.height) for i in indices]


# ─────────────────────────────────────────────────────────────────────────────
# Downloader
# ─────────────────────────────────────────────────────────────────────────────

def _download_model(model_path: Path) -> bool:
    """Download face_landmarker.task if not present. Returns True on success."""
    if model_path.exists():
        return True

    model_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"[LandmarkUtils] Downloading face_landmarker.task model...")
    logger.info(f"[LandmarkUtils] URL: {_MODEL_URL}")

    try:
        def _progress(block_num, block_size, total_size):
            downloaded = block_num * block_size
            if total_size > 0:
                pct = min(downloaded / total_size * 100, 100)
                print(f"\r  Downloading: {pct:.1f}%", end="", flush=True)

        urllib.request.urlretrieve(_MODEL_URL, str(model_path), _progress)
        print()  # newline after progress
        logger.success(f"[LandmarkUtils] Model saved to {model_path}")
        return True

    except Exception as e:
        logger.error(f"[LandmarkUtils] Download failed: {e}")
        if model_path.exists():
            model_path.unlink()
        return False


# ─────────────────────────────────────────────────────────────────────────────
# Main Detector Class
# ─────────────────────────────────────────────────────────────────────────────

class FaceLandmarkDetector:
    """
    MediaPipe Tasks FaceLandmarker wrapper.

    Supports VIDEO mode (stateful, best for real-time) and
    IMAGE mode (stateless, for single-frame analysis).

    Usage::

        detector = FaceLandmarkDetector()                 # VIDEO mode
        result = detector.detect(frame_rgb)               # → LandmarkResult | None
        if result:
            px, py = result.pixel_xy(4)                   # nose tip in pixels
    """

    def __init__(
        self,
        model_path: Optional[Path] = None,
        num_faces: int = 1,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        use_video_mode: bool = True,
    ) -> None:
        """
        Args:
            model_path               : Path to face_landmarker.task (auto-download if None)
            num_faces                : Max number of faces to detect
            min_detection_confidence : Detection confidence threshold
            min_tracking_confidence  : Tracking confidence threshold
            use_video_mode           : True = VIDEO mode (real-time), False = IMAGE mode
        """
        if not _MP_TASKS_OK:
            raise RuntimeError(
                "MediaPipe Tasks API not available. "
                "Install with: pip install mediapipe"
            )

        self._path = Path(model_path) if model_path else _DEFAULT_MODEL_PATH
        self._use_video = use_video_mode
        self._timestamp_ms = 0

        # Download model if needed
        if not _download_model(self._path):
            raise RuntimeError(
                f"Could not download face_landmarker.task model. "
                f"Please download manually from:\n{_MODEL_URL}\n"
                f"and place it at: {self._path}"
            )

        # Build landmarker options
        running_mode = (mp_vision.RunningMode.VIDEO
                        if use_video_mode
                        else mp_vision.RunningMode.IMAGE)

        options = mp_vision.FaceLandmarkerOptions(
            base_options=mp_python.BaseOptions(
                model_asset_path=str(self._path)
            ),
            running_mode=running_mode,
            num_faces=num_faces,
            min_face_detection_confidence=min_detection_confidence,
            min_face_presence_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
        )

        self._landmarker = mp_vision.FaceLandmarker.create_from_options(options)
        logger.success(
            f"[FaceLandmarkDetector] Ready "
            f"({'VIDEO' if use_video_mode else 'IMAGE'} mode)"
        )

    # ─────────────────────────────────────────────────────────────────────────

    def detect(self, frame_rgb: np.ndarray) -> Optional[LandmarkResult]:
        """
        Run face landmark detection on an RGB numpy frame.

        Args:
            frame_rgb : H×W×3 uint8 RGB image

        Returns:
            LandmarkResult with 478 landmarks, or None if no face detected.
        """
        h, w = frame_rgb.shape[:2]
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=frame_rgb.astype(np.uint8)
        )

        if self._use_video:
            self._timestamp_ms += 33   # ~30 fps increment
            result = self._landmarker.detect_for_video(mp_image, self._timestamp_ms)
        else:
            result = self._landmarker.detect(mp_image)

        if not result.face_landmarks:
            return None

        # Convert to our NormalizedLandmark list
        lm_list = [
            NormalizedLandmark(x=lm.x, y=lm.y, z=lm.z)
            for lm in result.face_landmarks[0]
        ]

        return LandmarkResult(landmarks=lm_list, width=w, height=h)

    def close(self) -> None:
        """Release the landmarker."""
        try:
            self._landmarker.close()
        except Exception:
            pass
