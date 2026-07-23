"""
mask_detector.py — Module 1 / FE-5
=====================================
Detect and prevent mask-based spoofing attempts.

Two scenarios handled:
    1. Person wearing a facial mask to conceal identity (COVID mask, balaclava)
       → Flag: cannot verify identity, prompt for alternative auth
    2. Person holding a printed face mask / 3D prop mask over their face
       → Flag: active spoofing attempt, deny access + alert

Architecture:
    Lightweight MobileNetV3-Small classification:
    [no_mask, cloth_mask, surgical_mask, n95_mask, 3d_prop_mask]

    Additionally, YCrCb skin-region analysis to detect non-organic mask regions.

Author : Hussain Raza (SP23-BAI-057)
"""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Dict, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger
from PIL import Image
from torchvision import models, transforms


class MaskType(str, Enum):
    NO_MASK       = "no_mask"
    CLOTH_MASK    = "cloth_mask"
    SURGICAL_MASK = "surgical_mask"
    N95_MASK      = "n95_mask"
    PROP_MASK     = "prop_mask"       # 3D prop mask — active spoof
    UNKNOWN       = "unknown"


# Mask types that constitute spoofing attempts (FE-5)
SPOOF_MASK_TYPES = {MaskType.PROP_MASK}

# Mask types that block identity verification (prompt for alt auth)
BLOCKING_MASK_TYPES = {
    MaskType.CLOTH_MASK,
    MaskType.SURGICAL_MASK,
    MaskType.N95_MASK,
}


class MaskClassifier(nn.Module):
    """
    MobileNetV3-Small backbone fine-tuned for 5-class mask detection.
    Lightweight for edge deployment on Raspberry Pi (SRS L6).
    """

    NUM_CLASSES = 5  # no_mask, cloth, surgical, n95, prop

    def __init__(self, pretrained: bool = True) -> None:
        super().__init__()
        self.backbone = models.mobilenet_v3_small(
            weights=models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        )
        # Replace final classifier
        in_features = self.backbone.classifier[-1].in_features
        self.backbone.classifier[-1] = nn.Linear(in_features, self.NUM_CLASSES)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


class MaskDetector:
    """
    Mask-based spoofing detector (FE-5).

    Usage::

        detector = MaskDetector(device="cuda")
        result = detector.detect(face_pil)
        if result["is_spoof"]:
            deny_access("3D prop mask detected")
        elif result["blocks_identity"]:
            prompt_alternative_auth()
    """

    CLASS_NAMES = [
        MaskType.NO_MASK,
        MaskType.CLOTH_MASK,
        MaskType.SURGICAL_MASK,
        MaskType.N95_MASK,
        MaskType.PROP_MASK,
    ]

    def __init__(
        self,
        device: str = "cuda",
        confidence_threshold: float = 0.85,
        weights_path: str | None = None,
    ) -> None:
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.confidence_threshold = confidence_threshold

        self.model = MaskClassifier(pretrained=True).to(self.device)

        if weights_path is None:
            default_weights = Path(__file__).resolve().parent / "weights" / "mask_detection_best.pth"
            if default_weights.exists():
                weights_path = str(default_weights)

        if weights_path:
            try:
                state = torch.load(weights_path, map_location=self.device)
                self.model.load_state_dict(state)
                logger.info(f"[MaskDetector] Loaded weights from {weights_path}")
            except Exception as e:
                logger.warning(f"[MaskDetector] Could not load weights: {e}. Using ImageNet init.")
        else:
            logger.warning(
                "[MaskDetector] No custom weights. Using ImageNet-pretrained MobileNetV3. "
                "Train on mask dataset for production use."
            )

        self.model.eval()

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

        logger.info(f"[MaskDetector] Ready on {self.device}")

    # ─────────────────────────────────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────────────────────────────────

    @torch.no_grad()
    def detect(self, face_pil: Image.Image) -> Dict:
        """
        Detect mask type and determine if it constitutes a spoofing attempt.

        Args:
            face_pil: RGB PIL face crop

        Returns:
            dict with:
                mask_type       (MaskType) classified mask type
                confidence      (float)   classification confidence 0–1
                is_spoof        (bool)    True if prop mask / active spoofing
                blocks_identity (bool)    True if mask hides face (cloth/surgical/n95)
                liveness_score  (float)   0–1 (1.0 = no mask, lower for masks)
                all_probs       (dict)    probabilities for all 5 classes
        """
        # Stage 1: CNN classification
        cnn_result = self._cnn_detect(face_pil)

        # Stage 2: Skin-region check (catches cases where CNN is uncertain)
        skin_check = self._skin_region_check(np.array(face_pil.convert("RGB")))

        mask_type  = cnn_result["mask_type"]
        confidence = cnn_result["confidence"]

        # Override: if skin coverage is very low → likely wearing something
        if skin_check["skin_ratio"] < 0.20 and mask_type == MaskType.NO_MASK:
            mask_type  = MaskType.CLOTH_MASK
            confidence = 0.6

        is_spoof        = mask_type in SPOOF_MASK_TYPES and confidence >= self.confidence_threshold
        blocks_identity = mask_type in BLOCKING_MASK_TYPES

        # Liveness score: 1.0 for no mask, 0.0 for prop spoof, 0.5 for regular mask
        if mask_type == MaskType.NO_MASK:
            liveness_score = 1.0
        elif is_spoof:
            liveness_score = 0.0
        elif blocks_identity:
            liveness_score = 0.5
        else:
            liveness_score = 0.8

        return {
            "mask_type": mask_type,
            "confidence": confidence,
            "is_spoof": is_spoof,
            "blocks_identity": blocks_identity,
            "liveness_score": liveness_score,
            "all_probs": cnn_result["all_probs"],
            "skin_ratio": skin_check["skin_ratio"],
        }

    # ─────────────────────────────────────────────────────────────────────────
    # Internal Stages
    # ─────────────────────────────────────────────────────────────────────────

    @torch.no_grad()
    def _cnn_detect(self, face_pil: Image.Image) -> Dict:
        """Run MobileNetV3 classification."""
        tensor = self.transform(face_pil.convert("RGB")).unsqueeze(0).to(self.device)
        logits = self.model(tensor)                    # (1, 5)
        probs  = F.softmax(logits, dim=1)[0]           # (5,)

        best_idx  = int(probs.argmax().item())
        best_prob = float(probs[best_idx].item())

        all_probs = {
            cls.value: float(probs[i].item())
            for i, cls in enumerate(self.CLASS_NAMES)
        }

        return {
            "mask_type": self.CLASS_NAMES[best_idx],
            "confidence": best_prob,
            "all_probs": all_probs,
        }

    def _skin_region_check(self, face_rgb: np.ndarray) -> Dict:
        """
        Estimate skin coverage in the lower face region using YCrCb.
        Real faces without masks → high skin ratio.
        Cloth/surgical masks → low skin ratio in lower half.
        """
        # Analyse lower half of face (nose + mouth + chin region)
        h, w = face_rgb.shape[:2]
        lower_face = face_rgb[h // 2:, :]

        ycrcb = cv2.cvtColor(lower_face, cv2.COLOR_RGB2YCrCb)

        # Typical skin color range in YCrCb
        lower_skin = np.array([0,   133, 77],  dtype=np.uint8)
        upper_skin = np.array([255, 173, 127], dtype=np.uint8)

        skin_mask  = cv2.inRange(ycrcb, lower_skin, upper_skin)
        skin_ratio = float(np.sum(skin_mask > 0)) / (lower_face.shape[0] * lower_face.shape[1])

        return {"skin_ratio": skin_ratio, "skin_mask": skin_mask}

    # ─────────────────────────────────────────────────────────────────────────
    # Visualization
    # ─────────────────────────────────────────────────────────────────────────

    def draw_result(self, frame_bgr: np.ndarray, result: Dict) -> np.ndarray:
        """Overlay mask detection result on frame."""
        vis = frame_bgr.copy()
        mask_type = result["mask_type"].value
        conf = result["confidence"]

        if result["is_spoof"]:
            color, status = (0, 0, 255), f"SPOOF: {mask_type}"
        elif result["blocks_identity"]:
            color, status = (0, 165, 255), f"MASK: {mask_type}"
        else:
            color, status = (0, 255, 0), "NO MASK"

        cv2.putText(vis, f"{status} ({conf:.2f})", (10, 180),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        return vis
