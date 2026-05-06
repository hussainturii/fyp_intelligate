"""
liveness_detector.py — Module 1 / Anti-Spoofing Pipeline
===========================================================
Combines all four anti-spoofing sub-modules into a single
weighted liveness score and decision.

Business Rule (BR-01):
    Access MUST NOT be granted unless liveness score >= 0.90

Sub-checks and weights (configurable via config.yaml):
    Blink    (FE-2) — 25%   Real person blinks, photos don't
    Texture  (FE-3) — 35%   Skin micro-texture vs paper/screen
    Depth    (FE-4) — 25%   3D face geometry vs flat image
    Mask     (FE-5) — 15%   Detect and reject mask-based spoofs

Decision:
    fused_score = w_blink*blink + w_texture*texture + w_depth*depth + w_mask*mask
    is_live     = fused_score >= LIVENESS_THRESHOLD (default: 0.90)

Author : Hussain Raza (SP23-BAI-057)
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, Optional

import cv2
import numpy as np
from loguru import logger
from PIL import Image

from .anti_spoofing.blink_detector import BlinkDetector, BlinkState
from .anti_spoofing.depth_estimator import DepthEstimator
from .anti_spoofing.mask_detector import MaskDetector
from .anti_spoofing.texture_analyzer import TextureAnalyzer
from .utils.landmark_utils import FaceLandmarkDetector


# ─────────────────────────────────────────────────────────────────────────────
# Liveness Session
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class LivenessSession:
    """
    Maintains state for one person's liveness check session.
    A session spans multiple frames until a decision is made.
    """
    session_id: str = field(default_factory=lambda: str(int(time.time() * 1000)))
    blink_state: BlinkState = field(default_factory=BlinkState)
    frames_processed: int = 0
    started_at: float = field(default_factory=time.time)

    # Running scores (updated each frame)
    blink_score:   float = 0.0
    texture_score: float = 0.0
    depth_score:   float = 0.0
    mask_score:    float = 1.0   # Optimistic start
    fused_score:   float = 0.0

    # Final decision (set once threshold is reached or timeout)
    decision_made: bool = False
    is_live:       bool = False
    deny_reason:   str = ""

    @property
    def elapsed_seconds(self) -> float:
        return time.time() - self.started_at


# ─────────────────────────────────────────────────────────────────────────────
# Liveness Detector
# ─────────────────────────────────────────────────────────────────────────────

class LivenessDetector:
    """
    Unified liveness detection pipeline (FE-2 through FE-5).

    Processes video frames incrementally and maintains a session per person.
    Returns a live/spoof decision once enough evidence is gathered.

    Usage (video loop)::

        detector = LivenessDetector()
        session  = detector.new_session()

        for frame in camera_frames:
            result = detector.process_frame(frame, face_pil, session)
            if result["decision_made"]:
                if result["is_live"]:
                    proceed_to_recognition()
                else:
                    deny_access(result["deny_reason"])
                break
    """

    def __init__(
        self,
        liveness_threshold: float = 0.90,
        weights: Dict[str, float] | None = None,
        device: str = "cuda",
        blink_required: int = 1,
        blink_window_frames: int = 90,
        max_session_seconds: float = 10.0,
        texture_weights_path: str | None = None,
        mask_weights_path: str | None = None,
    ) -> None:
        """
        Args:
            liveness_threshold   : BR-01 threshold (default 0.90)
            weights              : Dict with keys blink, texture, depth, mask
            device               : 'cuda' or 'cpu'
            blink_required       : Minimum blinks to pass blink check
            blink_window_frames  : Frame window for blink collection
            max_session_seconds  : Timeout — deny if no decision in this time
            texture_weights_path : Path to trained TextureCNN weights
            mask_weights_path    : Path to trained MaskClassifier weights
        """
        self.threshold = liveness_threshold
        self.max_seconds = max_session_seconds

        self.weights = weights or {
            "blink":   0.25,
            "texture": 0.35,
            "depth":   0.25,
            "mask":    0.15,
        }
        assert abs(sum(self.weights.values()) - 1.0) < 1e-6, "Weights must sum to 1.0"

        logger.info(f"[LivenessDetector] Initializing sub-modules on {device}...")

        # One shared FaceLandmarkDetector → single model download + single session
        # Both BlinkDetector and DepthEstimator will reuse it via their own instance
        # (each creates its own detector — acceptable; the model file is shared on disk)
        self.blink_det   = BlinkDetector(required_blinks=blink_required,
                                          window_frames=blink_window_frames)
        self.texture_det = TextureAnalyzer(device=device,
                                            weights_path=texture_weights_path)
        self.depth_det   = DepthEstimator()
        self.mask_det    = MaskDetector(device=device,
                                         weights_path=mask_weights_path)

        logger.success(
            f"[LivenessDetector] Ready | threshold={liveness_threshold} | "
            f"weights={self.weights}"
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────────────────────────────────

    def new_session(self) -> LivenessSession:
        """Start a new liveness check session for a new person."""
        return LivenessSession()

    def process_frame(
        self,
        frame_bgr: np.ndarray,
        face_pil: Image.Image,
        session: LivenessSession,
    ) -> dict:
        """
        Process one video frame and update the session's liveness scores.

        Args:
            frame_bgr : Full BGR camera frame (for blink/depth via landmarks)
            face_pil  : Cropped, aligned face PIL image (for texture/mask CNN)
            session   : LivenessSession to accumulate evidence

        Returns:
            dict with:
                fused_score    (float) 0–1
                is_live        (bool)
                decision_made  (bool)  True once threshold reached or timeout
                deny_reason    (str)   Reason string if denied
                sub_scores     (dict)  Individual scores per check
                elapsed        (float) Seconds elapsed in session
        """
        if session.decision_made:
            return self._session_to_result(session)

        session.frames_processed += 1

        # ── Run all four sub-checks ──────────────────────────────────────────

        # FE-2: Blink
        blink_res  = self.blink_det.process_frame(frame_bgr, session.blink_state)
        session.blink_score = blink_res["liveness_score"]

        # FE-3: Texture (run every 3rd frame to save compute)
        if session.frames_processed % 3 == 0:
            texture_res = self.texture_det.analyze(face_pil)
            session.texture_score = texture_res["liveness_score"]

        # FE-4: Depth
        depth_res  = self.depth_det.estimate(frame_bgr)
        session.depth_score = depth_res["liveness_score"]

        # FE-5: Mask (run every 5th frame)
        if session.frames_processed % 5 == 0:
            mask_res = self.mask_det.detect(face_pil)
            session.mask_score = mask_res["liveness_score"]

            # Hard deny: prop mask (active spoof) — immediate decision
            if mask_res["is_spoof"]:
                session.decision_made = True
                session.is_live       = False
                session.deny_reason   = "Prop mask / 3D mask spoofing detected"
                logger.warning(f"[LivenessDetector] SPOOF: {session.deny_reason}")
                return self._session_to_result(session)

        # ── Fuse scores ──────────────────────────────────────────────────────
        session.fused_score = (
            self.weights["blink"]   * session.blink_score +
            self.weights["texture"] * session.texture_score +
            self.weights["depth"]   * session.depth_score +
            self.weights["mask"]    * session.mask_score
        )

        # ── Make decision ────────────────────────────────────────────────────

        # Wait at least 30 frames before deciding (1 second @ 30fps)
        # to give blink detection a chance
        if session.frames_processed < 30:
            return self._session_to_result(session)

        if session.fused_score >= self.threshold:
            session.decision_made = True
            session.is_live       = True
            logger.success(
                f"[LivenessDetector] LIVE CONFIRMED | "
                f"score={session.fused_score:.3f} | "
                f"frames={session.frames_processed}"
            )

        elif session.elapsed_seconds >= self.max_seconds:
            # Timeout
            session.decision_made = True
            session.is_live       = False
            session.deny_reason   = (
                f"Liveness timeout ({self.max_seconds}s). "
                f"Score: {session.fused_score:.2f} < {self.threshold}"
            )
            logger.warning(f"[LivenessDetector] TIMEOUT: {session.deny_reason}")

        return self._session_to_result(session)

    def quick_check(
        self,
        frame_bgr: np.ndarray,
        face_pil: Image.Image,
    ) -> dict:
        """
        Single-frame liveness check (no session, no blink check).
        Used for quick screening — less reliable than full session check.
        Only runs texture, depth, mask checks.
        """
        texture_res = self.texture_det.analyze(face_pil)
        depth_res   = self.depth_det.estimate(frame_bgr)
        mask_res    = self.mask_det.detect(face_pil)

        # Quick fuse without blink
        score = (
            0.50 * texture_res["liveness_score"] +
            0.35 * depth_res["liveness_score"] +
            0.15 * mask_res["liveness_score"]
        )

        return {
            "fused_score": float(score),
            "is_live": score >= self.threshold,
            "texture_score": texture_res["liveness_score"],
            "depth_score": depth_res["liveness_score"],
            "mask_score": mask_res["liveness_score"],
            "is_spoof": mask_res["is_spoof"] or texture_res["liveness_score"] < 0.3,
        }

    # ─────────────────────────────────────────────────────────────────────────
    # Visualization
    # ─────────────────────────────────────────────────────────────────────────

    def draw_liveness_overlay(
        self,
        frame_bgr: np.ndarray,
        session: LivenessSession,
    ) -> np.ndarray:
        """Draw liveness scores and status on a BGR frame."""
        vis = frame_bgr.copy()
        h, w = vis.shape[:2]

        # Background panel
        cv2.rectangle(vis, (w - 230, 10), (w - 10, 200), (0, 0, 0), -1)
        cv2.rectangle(vis, (w - 230, 10), (w - 10, 200), (255, 255, 255), 1)

        def bar(label: str, score: float, y: int, color: tuple) -> None:
            cv2.putText(vis, f"{label}: {score:.2f}", (w - 220, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
            bar_w = int(190 * score)
            cv2.rectangle(vis, (w - 220, y + 5), (w - 220 + bar_w, y + 12), color, -1)

        bar("Blink",   session.blink_score,   30,  (0, 255, 255))
        bar("Texture", session.texture_score, 60,  (255, 165, 0))
        bar("Depth",   session.depth_score,   90,  (255, 0, 255))
        bar("Mask",    session.mask_score,    120, (0, 255, 0))
        bar("FUSED",   session.fused_score,   150, (255, 255, 255))

        # Final status
        if session.decision_made:
            status = "LIVE" if session.is_live else "SPOOF"
            color  = (0, 255, 0) if session.is_live else (0, 0, 255)
        else:
            status = "..."
            color  = (0, 165, 255)

        cv2.putText(vis, status, (w - 220, 185),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        return vis

    # ─────────────────────────────────────────────────────────────────────────
    # Internal
    # ─────────────────────────────────────────────────────────────────────────

    def _session_to_result(self, session: LivenessSession) -> dict:
        return {
            "fused_score":   session.fused_score,
            "is_live":       session.is_live,
            "decision_made": session.decision_made,
            "deny_reason":   session.deny_reason,
            "elapsed":       session.elapsed_seconds,
            "frames":        session.frames_processed,
            "sub_scores": {
                "blink":   session.blink_score,
                "texture": session.texture_score,
                "depth":   session.depth_score,
                "mask":    session.mask_score,
            },
        }
