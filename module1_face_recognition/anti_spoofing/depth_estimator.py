"""
depth_estimator.py — Module 1 / FE-4
========================================
Depth-based liveness detection using MediaPipe Tasks FaceLandmarker.

A real 3D face has significant depth variation across landmarks
(nose protrudes, eyes are recessed, cheeks curve inward).
A flat 2D photo/screen has near-zero depth variance.

Algorithm:
    1. FaceLandmarkDetector → 478 3D landmarks (x, y, z)
    2. Extract Z-coordinates (depth)
    3. Compute:
       - Z-variance across all landmarks   (low = flat = spoof)
       - Nose tip prominence vs face plane (low = flat = spoof)
       - Bilateral asymmetry score         (perfect symmetry = flat = spoof)
    4. Combine into a depth liveness score 0–1

Author : Hussain Raza (SP23-BAI-057)
"""

from __future__ import annotations

from typing import Optional

import cv2
import numpy as np
from loguru import logger

from ..utils.landmark_utils import FaceLandmarkDetector, LandmarkResult


# ── Key landmark indices ──────────────────────────────────────────────────────
NOSE_TIP_IDX    = 4
CHIN_IDX        = 152
FOREHEAD_IDX    = 10
LEFT_CHEEK_IDX  = 234
RIGHT_CHEEK_IDX = 454

# Indices used for left/right symmetry check
LEFT_SIDE_IDXS  = [234, 93, 132, 58]
RIGHT_SIDE_IDXS = [454, 323, 361, 288]

# All 468 face mesh indices (excludes the 10 iris landmarks at 468–477)
ALL_FACE_IDXS   = list(range(468))


class DepthEstimator:
    """
    3D depth-based liveness estimator using MediaPipe Tasks FaceLandmarker.

    Usage::

        estimator = DepthEstimator()
        result    = estimator.estimate(frame_bgr)
        # result["liveness_score"] → 0.0–1.0
        # result["is_3d"]          → True if face appears 3-dimensional
    """

    def __init__(
        self,
        depth_variance_threshold: float = 0.0003,  # Calibrated to MediaPipe normalized Z
        nose_tip_prominence:      float = 0.030,   # Real nose: ~0.045–0.055
        min_detection_confidence: float = 0.5,
        min_tracking_confidence:  float = 0.5,
        landmark_model_path: Optional[str] = None,
    ) -> None:
        """
        Args:
            depth_variance_threshold : Min Z-variance to classify as 3D face
            nose_tip_prominence      : Min nose Z-depth relative to face plane
            min_detection_confidence : MediaPipe detection confidence
            min_tracking_confidence  : MediaPipe tracking confidence
            landmark_model_path      : Override path to face_landmarker.task
        """
        self.depth_variance_threshold = depth_variance_threshold
        self.nose_tip_prominence      = nose_tip_prominence

        # Shared landmark detector instance (VIDEO mode)
        self._landmark_det = FaceLandmarkDetector(
            model_path=landmark_model_path,
            num_faces=1,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
            use_video_mode=True,
        )

        logger.info(
            f"[DepthEstimator] var_threshold={depth_variance_threshold}, "
            f"nose_prominence={nose_tip_prominence}"
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────────────────────────────────

    def estimate(self, frame_bgr: np.ndarray) -> dict:
        """
        Estimate 3D depth liveness from a BGR frame.

        Returns dict with:
            liveness_score   (float) 0–1, higher = more likely 3D/real
            is_3d            (bool)  True if depth analysis passes
            z_variance       (float) Z-coordinate variance of landmarks
            nose_prominence  (float) Nose tip Z relative to face plane
            depth_map        (ndarray|None)
        """
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        lm = self._landmark_det.detect(frame_rgb)

        if lm is None:
            logger.debug("[DepthEstimator] No face mesh detected.")
            return {
                "liveness_score":  0.0,
                "is_3d":           False,
                "z_variance":      0.0,
                "nose_prominence": 0.0,
                "depth_map":       None,
            }

        # ── Metric 1: Z-Variance ─────────────────────────────────────────────
        z_values  = lm.z_array(ALL_FACE_IDXS)
        z_variance = float(np.var(z_values))
        variance_score = float(np.clip(
            z_variance / (self.depth_variance_threshold * 3), 0.0, 1.0
        ))

        # ── Metric 2: Nose Tip Prominence ────────────────────────────────────
        # In MediaPipe camera space, a more negative Z means closer to camera.
        # Nose tip should protrude toward the camera relative to chin/forehead.
        nose_z      = lm.z_value(NOSE_TIP_IDX)
        chin_z      = lm.z_value(CHIN_IDX)
        fhd_z       = lm.z_value(FOREHEAD_IDX)
        face_plane  = (chin_z + fhd_z) / 2.0
        nose_prom   = float(face_plane - nose_z)   # positive = nose protrudes
        prominence_score = float(np.clip(
            nose_prom / (self.nose_tip_prominence * 3), 0.0, 1.0
        ))

        # ── Metric 3: Bilateral Asymmetry ────────────────────────────────────
        left_z  = lm.z_array(LEFT_SIDE_IDXS)
        right_z = lm.z_array(RIGHT_SIDE_IDXS)
        asymmetry       = float(np.mean(np.abs(left_z - right_z)))
        asymmetry_score = float(np.clip(asymmetry / 0.02, 0.0, 1.0))

        # ── Fused Score ───────────────────────────────────────────────────────
        fused = float(np.clip(
            0.50 * variance_score +
            0.35 * prominence_score +
            0.15 * asymmetry_score,
            0.0, 1.0
        ))

        is_3d = (
            z_variance >= self.depth_variance_threshold and
            nose_prom  >= self.nose_tip_prominence
        )

        logger.debug(
            f"[DepthEstimator] z_var={z_variance:.4f}, "
            f"nose={nose_prom:.4f}, asym={asymmetry:.4f}, "
            f"score={fused:.3f}, is_3d={is_3d}"
        )

        return {
            "liveness_score":  fused,
            "is_3d":           is_3d,
            "z_variance":      z_variance,
            "nose_prominence": nose_prom,
            "depth_map":       self._make_depth_viz(frame_bgr, lm),
        }

    # ─────────────────────────────────────────────────────────────────────────
    # Visualization
    # ─────────────────────────────────────────────────────────────────────────

    def _make_depth_viz(self, frame_bgr: np.ndarray, lm: LandmarkResult) -> np.ndarray:
        """False-colour depth overlay. Blue = far, Red = close."""
        vis = frame_bgr.copy()
        z_vals = lm.z_array(ALL_FACE_IDXS)
        z_min, z_max = z_vals.min(), z_vals.max()
        z_range = z_max - z_min + 1e-8

        for i in ALL_FACE_IDXS:
            px, py = lm.pixel_xy(i)
            norm = 1.0 - (lm.z_value(i) - z_min) / z_range
            r = int(norm * 255)
            b = int((1 - norm) * 255)
            cv2.circle(vis, (px, py), 1, (b, 0, r), -1)

        return vis

    def draw_result(self, frame_bgr: np.ndarray, result: dict) -> np.ndarray:
        """Overlay depth score on frame for debugging."""
        vis   = frame_bgr.copy()
        score = result["liveness_score"]
        color = (0, 255, 0) if result["is_3d"] else (0, 0, 255)
        cv2.putText(vis, f"Depth: {score:.3f}", (10, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        cv2.putText(vis, f"3D: {'PASS' if result['is_3d'] else 'FAIL'}", (10, 150),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        return vis

    def close(self) -> None:
        """Release MediaPipe resources."""
        self._landmark_det.close()
