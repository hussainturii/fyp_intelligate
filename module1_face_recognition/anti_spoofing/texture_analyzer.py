"""
texture_analyzer.py — Module 1 / FE-3
========================================
Texture-based spoof detection to distinguish real faces from
printed photos, video replays, and digital screens.

Two-stage pipeline:
    Stage 1 — LBP (Local Binary Pattern) features:
        Fast, hand-crafted texture descriptor that distinguishes
        smooth paper/screen textures from natural skin micro-texture.

    Stage 2 — CNN Texture Classifier:
        Lightweight CNN trained on anti-spoofing datasets
        (NUAA, CASIA-SURF, MSU-MFSD) for high-accuracy spoofing detection.

Reference: Boulkenafet et al., "Face Anti-Spoofing Using Texture Analysis",
           IEEE TIFS 2017.

Author : Hussain Raza (SP23-BAI-057)
"""

from __future__ import annotations

from typing import Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger
from PIL import Image
from skimage.feature import local_binary_pattern
from torchvision import transforms


# ─────────────────────────────────────────────────────────────────────────────
# CNN Texture Classifier Architecture
# ─────────────────────────────────────────────────────────────────────────────

class TextureCNN(nn.Module):
    """
    Compact CNN for binary spoof / real classification based on texture.

    Input  : 3×128×128 RGB face crop
    Output : 2-class softmax [P(real), P(spoof)]

    Architecture is deliberately lightweight for edge deployment
    (Raspberry Pi constraint from SRS L6).
    """

    def __init__(self) -> None:
        super().__init__()

        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 32, kernel_size=3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2, 2),          # 64×64

            # Block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2, 2),          # 32×32

            # Block 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.MaxPool2d(2, 2),          # 16×16

            # Block 4
            nn.Conv2d(128, 256, kernel_size=3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),  # → 256×4×4
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 2),  # [real, spoof]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


# ─────────────────────────────────────────────────────────────────────────────
# Main Texture Analyzer
# ─────────────────────────────────────────────────────────────────────────────

class TextureAnalyzer:
    """
    Two-stage texture-based anti-spoofing detector.

    Usage::

        analyzer = TextureAnalyzer(device="cuda")
        result = analyzer.analyze(face_pil)
        # result["is_real"]        → True/False
        # result["liveness_score"] → 0.0–1.0
        # result["lbp_score"]      → LBP stage score
        # result["cnn_score"]      → CNN stage score
    """

    def __init__(
        self,
        device: str = "cuda",
        lbp_radius: int = 3,
        lbp_n_points: int = 24,
        spoof_threshold: float = 0.5,
        weights_path: str | None = None,
    ) -> None:
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.spoof_threshold = spoof_threshold

        # LBP params
        self.lbp_radius = lbp_radius
        self.lbp_n_points = lbp_n_points
        self.lbp_method = "uniform"

        # Build CNN
        self.model = TextureCNN().to(self.device)
        self._cnn_trained = False   # track whether real weights are loaded

        if weights_path:
            try:
                state = torch.load(weights_path, map_location=self.device)
                self.model.load_state_dict(state)
                self._cnn_trained = True
                logger.info(f"[TextureAnalyzer] Loaded weights from {weights_path}")
            except Exception as e:
                logger.warning(f"[TextureAnalyzer] Could not load weights: {e}. Using LBP-only mode.")
        else:
            logger.warning(
                "[TextureAnalyzer] No CNN weights — running in LBP-only mode. "
                "Train on NUAA/CASIA-SURF and set texture_weights_path to enable CNN stage."
            )

        self.model.eval()

        self.transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ])

        logger.info(f"[TextureAnalyzer] Ready | threshold={spoof_threshold}")

    # ─────────────────────────────────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────────────────────────────────

    def analyze(self, face_pil: Image.Image) -> dict:
        """
        Full two-stage texture analysis on a face image.

        Returns:
            dict with:
                is_real        (bool)
                liveness_score (float) 0-1 (higher = more likely real)
                lbp_score      (float) 0-1
                cnn_score      (float) 0-1 (P(real) from CNN)
                spoof_type     (str)   hint about spoof type if detected
        """
        face_np = np.array(face_pil.convert("RGB"))

        lbp_score = self._lbp_stage(face_np)

        if self._cnn_trained:
            # Full two-stage fusion — LBP pre-filter + CNN main signal
            cnn_score = self._cnn_stage(face_pil)
            fused_score = 0.3 * lbp_score + 0.7 * cnn_score
        else:
            # LBP-only mode: CNN not trained yet, rely entirely on texture entropy
            cnn_score = lbp_score   # report same value for transparency
            fused_score = lbp_score

        is_real = fused_score >= (1.0 - self.spoof_threshold)

        # Guess spoof type
        spoof_type = "none"
        if not is_real:
            if lbp_score < 0.3:
                spoof_type = "print_attack"    # Very uniform texture → printed photo
            elif lbp_score < 0.5:
                spoof_type = "screen_replay"   # Slight digital texture

        return {
            "is_real": is_real,
            "liveness_score": fused_score,
            "lbp_score": lbp_score,
            "cnn_score": cnn_score,
            "spoof_type": spoof_type,
        }

    # ─────────────────────────────────────────────────────────────────────────
    # Stage 1: LBP
    # ─────────────────────────────────────────────────────────────────────────

    def _lbp_stage(self, face_rgb: np.ndarray) -> float:
        """
        Compute LBP texture score. Real faces have higher variance in
        LBP histograms due to natural skin micro-texture.

        Returns score in [0, 1], higher = more likely real.
        """
        gray = cv2.cvtColor(face_rgb, cv2.COLOR_RGB2GRAY)
        gray = cv2.resize(gray, (128, 128))

        # Compute LBP map
        lbp = local_binary_pattern(
            gray,
            P=self.lbp_n_points,
            R=self.lbp_radius,
            method=self.lbp_method,
        )

        # Histogram of LBP codes
        n_bins = self.lbp_n_points + 2
        hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins), density=True)

        # Entropy of histogram (real faces → higher entropy)
        hist_nonzero = hist[hist > 0]
        entropy = -np.sum(hist_nonzero * np.log2(hist_nonzero))

        # Normalize entropy to [0, 1] (empirically, real ≈ 3–5 bits)
        max_entropy = np.log2(n_bins)
        score = float(np.clip(entropy / max_entropy, 0.0, 1.0))

        return score

    # ─────────────────────────────────────────────────────────────────────────
    # Stage 2: CNN
    # ─────────────────────────────────────────────────────────────────────────

    @torch.no_grad()
    def _cnn_stage(self, face_pil: Image.Image) -> float:
        """
        Run face through TextureCNN, return P(real) from softmax.
        """
        tensor = self.transform(face_pil.convert("RGB")).unsqueeze(0).to(self.device)
        logits = self.model(tensor)                     # (1, 2)
        probs = F.softmax(logits, dim=1)                # (1, 2) [P(real), P(spoof)]
        p_real = float(probs[0, 0].item())
        return p_real

    # ─────────────────────────────────────────────────────────────────────────
    # Visualization
    # ─────────────────────────────────────────────────────────────────────────

    def visualize_lbp(self, face_rgb: np.ndarray) -> np.ndarray:
        """Return LBP image for visualization/debugging."""
        gray = cv2.cvtColor(face_rgb, cv2.COLOR_RGB2GRAY)
        lbp = local_binary_pattern(gray, P=self.lbp_n_points, R=self.lbp_radius, method=self.lbp_method)
        lbp_norm = ((lbp / lbp.max()) * 255).astype(np.uint8)
        return cv2.applyColorMap(lbp_norm, cv2.COLORMAP_JET)
