"""
blink_detector.py — Module 1 / FE-2
======================================
Blink detection for liveness verification using Eye Aspect Ratio (EAR).
A real person blinks naturally; a photo/screen does not.

Algorithm:
    1. FaceLandmarkDetector (MediaPipe Tasks API) → 478 3D landmarks
    2. Extract 6 eye-contour points per eye
    3. Compute EAR = (||p2-p6|| + ||p3-p5||) / (2 * ||p1-p4||)
    4. EAR < threshold → eye closed
    5. Require N consecutive closed frames → 1 blink
    6. Require M total blinks in window → PASS

Reference: Soukupová & Čech, "Real-Time Eye Blink Detection using
           Facial Landmarks", CVWW 2016.

Author : Hussain Raza (SP23-BAI-057)
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Optional, Tuple

import cv2
import numpy as np
from loguru import logger

from ..utils.landmark_utils import FaceLandmarkDetector


# ── MediaPipe FaceMesh landmark indices for eye contours ─────────────────────
# These indices are identical between the old solutions API and new Tasks API.
# Left eye: 6 key points (from medial to lateral canthus)
LEFT_EYE_IDXS  = [362, 385, 387, 263, 373, 380]
# Right eye: 6 key points
RIGHT_EYE_IDXS = [33,  160, 158, 133, 153, 144]


@dataclass
class BlinkState:
    """Tracks blink state across frames for one face."""
    blink_count:      int   = 0
    frames_eye_closed: int  = 0
    ear_history: deque = field(default_factory=lambda: deque(maxlen=90))
    is_passed:   bool  = False
    last_ear:    float = 1.0


class BlinkDetector:
    """
    Real-time blink detector using MediaPipe Tasks FaceLandmarker.

    Usage::

        detector = BlinkDetector()
        state    = BlinkState()

        for frame in video_frames:
            result = detector.process_frame(frame, state)
            if result["passed"]:
                print("Liveness confirmed via blink!")
    """

    def __init__(
        self,
        ear_threshold:     float = 0.25,
        consecutive_frames: int  = 2,
        required_blinks:   int   = 1,
        window_frames:     int   = 90,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence:  float = 0.5,
        landmark_model_path: Optional[str] = None,
    ) -> None:
        """
        Args:
            ear_threshold          : EAR below this → eye closed
            consecutive_frames     : Min frames eye must be closed to count as blink
            required_blinks        : Min blinks needed to pass liveness check
            window_frames          : Max frames to collect blinks in (~3s @ 30fps)
            min_detection_confidence: MediaPipe detection confidence threshold
            min_tracking_confidence : MediaPipe tracking confidence threshold
            landmark_model_path    : Override path to face_landmarker.task model
        """
        self.ear_threshold      = ear_threshold
        self.consecutive_frames = consecutive_frames
        self.required_blinks    = required_blinks
        self.window_frames      = window_frames

        # Use the new Tasks-based landmark detector (auto-downloads model)
        self._landmark_det = FaceLandmarkDetector(
            model_path=landmark_model_path,
            num_faces=1,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
            use_video_mode=True,    # stateful VIDEO mode for real-time
        )

        logger.info(
            f"[BlinkDetector] EAR threshold={ear_threshold}, "
            f"required_blinks={required_blinks}, window={window_frames}f"
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Core Processing
    # ─────────────────────────────────────────────────────────────────────────

    def process_frame(
        self,
        frame_bgr: np.ndarray,
        state: Optional[BlinkState] = None,
    ) -> dict:
        """
        Process one BGR video frame and update blink state.

        Args:
            frame_bgr : BGR OpenCV frame
            state     : BlinkState to accumulate blinks across frames.
                        If None, creates a new state each call (stateless mode).

        Returns:
            dict with keys:
                ear            (float)  Average EAR this frame
                eye_closed     (bool)   Whether eye is currently closed
                blink_count    (int)    Total blinks detected
                passed         (bool)   True if required blinks reached
                liveness_score (float)  0.0–1.0 score for liveness pipeline
        """
        if state is None:
            state = BlinkState()

        # Convert BGR → RGB for MediaPipe
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        lm_result = self._landmark_det.detect(frame_rgb)

        if lm_result is None:
            return self._build_result(state, ear=state.last_ear, eye_closed=False)

        # Get pixel coordinates of eye landmarks
        left_pts  = lm_result.xy_array(LEFT_EYE_IDXS)
        right_pts = lm_result.xy_array(RIGHT_EYE_IDXS)

        left_ear  = self._compute_ear(left_pts)
        right_ear = self._compute_ear(right_pts)
        avg_ear   = (left_ear + right_ear) / 2.0

        state.ear_history.append(avg_ear)
        state.last_ear = avg_ear

        eye_closed = avg_ear < self.ear_threshold

        if eye_closed:
            state.frames_eye_closed += 1
        else:
            # Eye just opened → if it was closed long enough, count as blink
            if state.frames_eye_closed >= self.consecutive_frames:
                state.blink_count += 1
                logger.debug(
                    f"[BlinkDetector] Blink #{state.blink_count} "
                    f"(EAR={avg_ear:.3f})"
                )
            state.frames_eye_closed = 0

        if state.blink_count >= self.required_blinks:
            state.is_passed = True

        return self._build_result(state, avg_ear, eye_closed)

    # ─────────────────────────────────────────────────────────────────────────
    # Helpers
    # ─────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _compute_ear(eye_pts: list) -> float:
        """
        Compute Eye Aspect Ratio from 6 landmark points.

        EAR = (||P2-P6|| + ||P3-P5||) / (2 * ||P1-P4||)

        Points order: [P1=left, P2=top-left, P3=top-right,
                       P4=right, P5=bottom-right, P6=bottom-left]
        """
        p = [np.array(pt) for pt in eye_pts]
        vertical1  = np.linalg.norm(p[1] - p[5])
        vertical2  = np.linalg.norm(p[2] - p[4])
        horizontal = np.linalg.norm(p[0] - p[3])

        if horizontal < 1e-6:
            return 0.30   # safe fallback

        return float((vertical1 + vertical2) / (2.0 * horizontal))

    def _build_result(
        self, state: BlinkState, ear: float, eye_closed: bool
    ) -> dict:
        score = min(state.blink_count / max(self.required_blinks, 1), 1.0)
        return {
            "ear":           ear,
            "eye_closed":    eye_closed,
            "blink_count":   state.blink_count,
            "passed":        state.is_passed,
            "liveness_score": score,
        }

    def reset_state(self) -> BlinkState:
        """Return a fresh BlinkState for a new person / session."""
        return BlinkState()

    def draw_ear_overlay(
        self,
        frame_bgr: np.ndarray,
        ear: float,
        blink_count: int,
        passed: bool,
    ) -> np.ndarray:
        """Draw EAR value and blink count on frame (for debugging/demo)."""
        vis   = frame_bgr.copy()
        color = (0, 255, 0) if passed else (0, 165, 255)
        cv2.putText(vis, f"EAR: {ear:.3f}",    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        cv2.putText(vis, f"Blinks: {blink_count}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        status = "BLINK PASS" if passed else "WAITING..."
        cv2.putText(vis, status, (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        return vis

    def close(self) -> None:
        """Release MediaPipe resources."""
        self._landmark_det.close()
