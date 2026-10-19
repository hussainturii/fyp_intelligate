"""
deep_pad.py — Deep Presentation-Attack Detection (PAD) using MiniFASNetV2
=========================================================================
Integrated in Milestone 3C for IntelliGate Face Biometric Pipeline.

Implements the official MiniFASNetV2 architecture from Minivision
Silent-Face-Anti-Spoofing with 2.7x scale bounding-box expansion,
80x80 BGR input, and 3-class softmax output (Print Spoof, Bona Fide, Screen Spoof).

Security Invariants:
- All PAD outputs are advisory model assessments, NOT proof of life.
- Face recognition and PAD NEVER independently trigger gate opening (ALLOW).
- Fail-closed design: Inconclusive, invalid crops, corrupted inputs, or failed
  initialization degrade to INCONCLUSIVE / NOT_EVALUATED and route to operator review.
- Cryptographic model weight verification against models/trusted_manifest.json.

Author: Hussain Raza (SP23-BAI-057) / IntelliGate Computer Vision Team
"""
from __future__ import annotations

import os
import threading
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger
from torch.nn import (
    BatchNorm1d,
    BatchNorm2d,
    Conv2d,
    Linear,
    Module,
    PReLU,
    Sequential,
)

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CHECKPOINT = ROOT / "models" / "2.7_80x80_MiniFASNetV2.pth"


# ─────────────────────────────────────────────────────────────────────────────
# MiniFASNetV2 Neural Network Architecture
# ─────────────────────────────────────────────────────────────────────────────

class Flatten(Module):
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return input.view(input.size(0), -1)


class Conv_block(Module):
    def __init__(
        self,
        in_c: int,
        out_c: int,
        kernel: Tuple[int, int] = (1, 1),
        stride: Tuple[int, int] = (1, 1),
        padding: Tuple[int, int] = (0, 0),
        groups: int = 1,
    ):
        super().__init__()
        self.conv = Conv2d(
            in_c,
            out_c,
            kernel_size=kernel,
            groups=groups,
            stride=stride,
            padding=padding,
            bias=False,
        )
        self.bn = BatchNorm2d(out_c)
        self.prelu = PReLU(out_c)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.prelu(self.bn(self.conv(x)))


class Linear_block(Module):
    def __init__(
        self,
        in_c: int,
        out_c: int,
        kernel: Tuple[int, int] = (1, 1),
        stride: Tuple[int, int] = (1, 1),
        padding: Tuple[int, int] = (0, 0),
        groups: int = 1,
    ):
        super().__init__()
        self.conv = Conv2d(
            in_c,
            out_channels=out_c,
            kernel_size=kernel,
            groups=groups,
            stride=stride,
            padding=padding,
            bias=False,
        )
        self.bn = BatchNorm2d(out_c)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.bn(self.conv(x))


class Depth_Wise(Module):
    def __init__(
        self,
        c1: Tuple[int, int],
        c2: Tuple[int, int],
        c3: Tuple[int, int],
        residual: bool = False,
        kernel: Tuple[int, int] = (3, 3),
        stride: Tuple[int, int] = (2, 2),
        padding: Tuple[int, int] = (1, 1),
        groups: int = 1,
    ):
        super().__init__()
        c1_in, c1_out = c1
        c2_in, c2_out = c2
        c3_in, c3_out = c3
        self.conv = Conv_block(c1_in, out_c=c1_out, kernel=(1, 1), padding=(0, 0), stride=(1, 1))
        self.conv_dw = Conv_block(c2_in, c2_out, groups=c2_in, kernel=kernel, padding=padding, stride=stride)
        self.project = Linear_block(c3_in, c3_out, kernel=(1, 1), padding=(0, 0), stride=(1, 1))
        self.residual = residual

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        short_cut = x if self.residual else None
        x = self.conv(x)
        x = self.conv_dw(x)
        x = self.project(x)
        return (short_cut + x) if self.residual else x


class Residual(Module):
    def __init__(
        self,
        c1: List[Tuple[int, int]],
        c2: List[Tuple[int, int]],
        c3: List[Tuple[int, int]],
        num_block: int,
        groups: int,
        kernel: Tuple[int, int] = (3, 3),
        stride: Tuple[int, int] = (1, 1),
        padding: Tuple[int, int] = (1, 1),
    ):
        super().__init__()
        modules = []
        for i in range(num_block):
            modules.append(
                Depth_Wise(
                    c1[i],
                    c2[i],
                    c3[i],
                    residual=True,
                    kernel=kernel,
                    padding=padding,
                    stride=stride,
                    groups=groups,
                )
            )
        self.model = Sequential(*modules)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


# Channel scaling configuration for 1.8M_ parameter MiniFASNetV2
_KEEP_DICT_1_8M_ = [
    32, 32, 103, 103, 64, 13, 13, 64, 13, 13, 64, 13,
    13, 64, 13, 13, 64, 231, 231, 128, 231, 231, 128, 52,
    52, 128, 26, 26, 128, 77, 77, 128, 26, 26, 128, 26, 26,
    128, 308, 308, 128, 26, 26, 128, 26, 26, 128, 512, 512
]


class MiniFASNet(Module):
    """
    MiniFASNet backbone for face anti-spoofing.
    Input resolution: 80x80 -> conv6_kernel: (5, 5).
    """

    def __init__(
        self,
        keep: List[int] = _KEEP_DICT_1_8M_,
        embedding_size: int = 128,
        conv6_kernel: Tuple[int, int] = (5, 5),
        drop_p: float = 0.2,
        num_classes: int = 3,
        img_channel: int = 3,
    ):
        super().__init__()
        self.embedding_size = embedding_size
        self.conv1 = Conv_block(img_channel, keep[0], kernel=(3, 3), stride=(2, 2), padding=(1, 1))
        self.conv2_dw = Conv_block(keep[0], keep[1], kernel=(3, 3), stride=(1, 1), padding=(1, 1), groups=keep[1])

        c1 = [(keep[1], keep[2])]
        c2 = [(keep[2], keep[3])]
        c3 = [(keep[3], keep[4])]
        self.conv_23 = Depth_Wise(c1[0], c2[0], c3[0], kernel=(3, 3), stride=(2, 2), padding=(1, 1), groups=keep[3])

        c1 = [(keep[4], keep[5]), (keep[7], keep[8]), (keep[10], keep[11]), (keep[13], keep[14])]
        c2 = [(keep[5], keep[6]), (keep[8], keep[9]), (keep[11], keep[12]), (keep[14], keep[15])]
        c3 = [(keep[6], keep[7]), (keep[9], keep[10]), (keep[12], keep[13]), (keep[15], keep[16])]
        self.conv_3 = Residual(c1, c2, c3, num_block=4, groups=keep[4], kernel=(3, 3), stride=(1, 1), padding=(1, 1))

        c1 = [(keep[16], keep[17])]
        c2 = [(keep[17], keep[18])]
        c3 = [(keep[18], keep[19])]
        self.conv_34 = Depth_Wise(c1[0], c2[0], c3[0], kernel=(3, 3), stride=(2, 2), padding=(1, 1), groups=keep[19])

        c1 = [
            (keep[19], keep[20]), (keep[22], keep[23]), (keep[25], keep[26]),
            (keep[28], keep[29]), (keep[31], keep[32]), (keep[34], keep[35])
        ]
        c2 = [
            (keep[20], keep[21]), (keep[23], keep[24]), (keep[26], keep[27]),
            (keep[29], keep[30]), (keep[32], keep[33]), (keep[35], keep[36])
        ]
        c3 = [
            (keep[21], keep[22]), (keep[24], keep[25]), (keep[27], keep[28]),
            (keep[30], keep[31]), (keep[33], keep[34]), (keep[36], keep[37])
        ]
        self.conv_4 = Residual(c1, c2, c3, num_block=6, groups=keep[19], kernel=(3, 3), stride=(1, 1), padding=(1, 1))

        c1 = [(keep[37], keep[38])]
        c2 = [(keep[38], keep[39])]
        c3 = [(keep[39], keep[40])]
        self.conv_45 = Depth_Wise(c1[0], c2[0], c3[0], kernel=(3, 3), stride=(2, 2), padding=(1, 1), groups=keep[40])

        c1 = [(keep[40], keep[41]), (keep[43], keep[44])]
        c2 = [(keep[41], keep[42]), (keep[44], keep[45])]
        c3 = [(keep[42], keep[43]), (keep[45], keep[46])]
        self.conv_5 = Residual(c1, c2, c3, num_block=2, groups=keep[40], kernel=(3, 3), stride=(1, 1), padding=(1, 1))

        self.conv_6_sep = Conv_block(keep[46], keep[47], kernel=(1, 1), stride=(1, 1), padding=(0, 0))
        self.conv_6_dw = Linear_block(keep[47], keep[48], groups=keep[48], kernel=conv6_kernel, stride=(1, 1), padding=(0, 0))
        self.conv_6_flatten = Flatten()
        self.linear = Linear(512, embedding_size, bias=False)
        self.bn = BatchNorm1d(embedding_size)
        self.drop = nn.Dropout(p=drop_p)
        self.prob = Linear(embedding_size, num_classes, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv1(x)
        out = self.conv2_dw(out)
        out = self.conv_23(out)
        out = self.conv_3(out)
        out = self.conv_34(out)
        out = self.conv_4(out)
        out = self.conv_45(out)
        out = self.conv_5(out)
        out = self.conv_6_sep(out)
        out = self.conv_6_dw(out)
        out = self.conv_6_flatten(out)
        if self.embedding_size != 512:
            out = self.linear(out)
        out = self.bn(out)
        out = self.drop(out)
        out = self.prob(out)
        return out


# ─────────────────────────────────────────────────────────────────────────────
# Presentation-Attack Detector Wrapper
# ─────────────────────────────────────────────────────────────────────────────

class MiniFASNetV2Detector:
    """
    Presentation Attack Detection (PAD) inference wrapper.
    
    Provides:
    - Verified checkpoint loading with safe torch.load(..., weights_only=True).
    - CUDA GPU acceleration with automatic fallback to CPU.
    - Scale 2.7x bounding box expansion with boundary clamping.
    - Standardized 80x80 BGR [0.0, 1.0] tensor preprocessing.
    - Conservative status mapping:
        * BONA_FIDE_CANDIDATE: p(live) >= 0.80 and argmax == 1
        * ATTACK_SUSPECTED:    p(live) < 0.40 or argmax in {0, 2}
        * INCONCLUSIVE:        0.40 <= p(live) < 0.80 or boundary crop
        * NOT_EVALUATED:       Face missing, tiny crop (< 35px), or degraded input
    """

    MODEL_NAME = "MiniFASNetV2-80x80-2.7"
    INPUT_SIZE = (80, 80)
    CROP_SCALE = 2.7
    THRESHOLD_BONA_FIDE = 0.80
    THRESHOLD_ATTACK = 0.40

    def __init__(
        self,
        checkpoint_path: Optional[Union[str, Path]] = None,
        device: Optional[str] = None,
        verify_integrity: bool = True,
    ):
        self.lock = threading.Lock()
        self.checkpoint_path = Path(checkpoint_path or DEFAULT_CHECKPOINT)
        self.verify_integrity = verify_integrity

        # Resolve device
        if device is not None:
            self.device_str = device
        else:
            self.device_str = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(self.device_str)

        self.model: Optional[MiniFASNet] = None
        self.initialized = False
        self._load_model()

    def _load_model(self) -> None:
        """Verify hash and load model weights safely."""
        if not self.checkpoint_path.exists():
            logger.error(f"[PAD] Checkpoint not found at {self.checkpoint_path}")
            return

        if self.verify_integrity:
            from module8_dashboard.backend.model_integrity import verify_model_file
            rel_path = "models/" + self.checkpoint_path.name
            try:
                verify_model_file(rel_path, strict=True)
            except Exception as exc:
                logger.error(f"[PAD] Model integrity check failed for {rel_path}: {exc}")
                raise RuntimeError(f"Corrupted or untrusted PAD checkpoint: {exc}") from exc

        try:
            model = MiniFASNet(
                keep=_KEEP_DICT_1_8M_,
                embedding_size=128,
                conv6_kernel=(5, 5),
                drop_p=0.0,
                num_classes=3,
                img_channel=3,
            )
            raw_state = torch.load(self.checkpoint_path, map_location=self.device, weights_only=True)
            clean_state = OrderedDict()
            for k, v in raw_state.items():
                clean_k = k.replace("module.", "")
                clean_state[clean_k] = v
            model.load_state_dict(clean_state, strict=True)
            model.to(self.device)
            model.eval()
            self.model = model
            self.initialized = True
            logger.info(f"[PAD] Initialized {self.MODEL_NAME} on {self.device_str}")
        except Exception as exc:
            logger.error(f"[PAD] Failed to load {self.checkpoint_path}: {exc}")
            self.model = None
            self.initialized = False

    @staticmethod
    def expand_and_crop(
        frame: np.ndarray,
        bbox: Union[List[int], Tuple[int, ...], np.ndarray],
        scale: float = 2.7,
        out_size: Tuple[int, int] = (80, 80),
    ) -> Tuple[Optional[np.ndarray], str]:
        """
        Expand bounding box by scale factor and crop with boundary clamping.
        
        Args:
            frame: Full BGR image (H x W x 3)
            bbox: (x, y, w, h) or (x1, y1, x2, y2)
            scale: Expansion scale factor (default: 2.7)
            out_size: Output resolution (W, H)
            
        Returns:
            (cropped_and_resized_patch, quality_status)
        """
        if frame is None or frame.size == 0:
            return None, "empty_frame"

        src_h, src_w = frame.shape[:2]
        if len(bbox) < 4:
            return None, "invalid_bbox"

        # Support both [x, y, w, h] and [x1, y1, x2, y2]
        b0, b1, b2, b3 = float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])
        if b2 > b0 and b3 > b1 and (b2 > 1.0 or b3 > 1.0) and b2 > src_w * 0.5 and b0 < src_w * 0.5 and b2 - b0 > 20:
            # Likely [x1, y1, x2, y2] format
            x, y, box_w, box_h = b0, b1, b2 - b0, b3 - b1
        else:
            x, y, box_w, box_h = b0, b1, b2, b3

        if box_w < 5 or box_h < 5:
            return None, "face_too_small"

        # Apply Silent-Face-Anti-Spoofing scale formula
        effective_scale = min((src_h - 1) / max(box_h, 1.0), min((src_w - 1) / max(box_w, 1.0), scale))
        new_w = box_w * effective_scale
        new_h = box_h * effective_scale
        center_x = box_w / 2.0 + x
        center_y = box_h / 2.0 + y

        left_top_x = center_x - new_w / 2.0
        left_top_y = center_y - new_h / 2.0
        right_bottom_x = center_x + new_w / 2.0
        right_bottom_y = center_y + new_h / 2.0

        clamped = False
        if left_top_x < 0:
            right_bottom_x -= left_top_x
            left_top_x = 0
            clamped = True
        if left_top_y < 0:
            right_bottom_y -= left_top_y
            left_top_y = 0
            clamped = True
        if right_bottom_x > src_w - 1:
            left_top_x -= (right_bottom_x - src_w + 1)
            right_bottom_x = src_w - 1
            clamped = True
        if right_bottom_y > src_h - 1:
            left_top_y -= (right_bottom_y - src_h + 1)
            right_bottom_y = src_h - 1
            clamped = True

        x1 = max(0, int(left_top_x))
        y1 = max(0, int(left_top_y))
        x2 = min(src_w, int(right_bottom_x) + 1)
        y2 = min(src_h, int(right_bottom_y) + 1)

        if x2 <= x1 or y2 <= y1:
            return None, "crop_out_of_bounds"

        patch = frame[y1:y2, x1:x2]
        if patch.size == 0:
            return None, "crop_degenerate"

        resized = cv2.resize(patch, out_size, interpolation=cv2.INTER_LINEAR)
        quality = "truncated_boundary" if clamped else "optimal"
        return resized, quality

    def predict_patch(self, patch: np.ndarray) -> Tuple[float, int, np.ndarray, float]:
        """
        Run forward pass on an 80x80 BGR image patch.
        
        Returns:
            (bona_fide_score, argmax_class, class_probabilities, latency_ms)
        """
        if not self.initialized or self.model is None:
            raise RuntimeError("PAD model is not initialized")

        t0 = time.perf_counter()
        # Preprocessing: BGR uint8 [0, 255] -> FloatTensor [0.0, 1.0] (ToTensor)
        # Note: No ImageNet mean/std subtraction per Silent-Face-Anti-Spoofing specification
        tensor = torch.from_numpy(patch.transpose((2, 0, 1))).float().div(255.0).unsqueeze(0).to(self.device)

        with self.lock:
            with torch.no_grad():
                logits = self.model(tensor)
                probs = F.softmax(logits, dim=1).squeeze(0).cpu().numpy()

        latency_ms = (time.perf_counter() - t0) * 1000.0
        pred_class = int(np.argmax(probs))
        bona_fide_score = float(probs[1])  # Class 1 is Bona Fide / Live Face
        return bona_fide_score, pred_class, probs, latency_ms

    def evaluate_face(
        self,
        frame: np.ndarray,
        face_bbox: Union[List[int], Tuple[int, ...], np.ndarray],
    ) -> Dict[str, Any]:
        """
        Evaluate presentation-attack suspicion for a detected face bounding box.
        
        Returns standard metadata dictionary:
            pad_status: 'BONA_FIDE_CANDIDATE' | 'ATTACK_SUSPECTED' | 'INCONCLUSIVE' | 'NOT_EVALUATED'
            pad_model: str
            pad_score: float | None
            pad_threshold: dict
            pad_class_probs: dict | None
            pad_inference_ms: float
            pad_quality_status: str
            device: str
        """
        if not self.initialized or self.model is None:
            return {
                "pad_status": "NOT_EVALUATED",
                "pad_model": self.MODEL_NAME,
                "pad_score": None,
                "pad_threshold": {"bona_fide": self.THRESHOLD_BONA_FIDE, "attack": self.THRESHOLD_ATTACK},
                "pad_class_probs": None,
                "pad_inference_ms": 0.0,
                "pad_quality_status": "uninitialized",
                "device": self.device_str,
            }

        # Check raw face dimensions
        w = face_bbox[2]
        h = face_bbox[3]
        if min(w, h) < 35:
            return {
                "pad_status": "NOT_EVALUATED",
                "pad_model": self.MODEL_NAME,
                "pad_score": None,
                "pad_threshold": {"bona_fide": self.THRESHOLD_BONA_FIDE, "attack": self.THRESHOLD_ATTACK},
                "pad_class_probs": None,
                "pad_inference_ms": 0.0,
                "pad_quality_status": "low_resolution_under_35px",
                "device": self.device_str,
            }

        patch, quality = self.expand_and_crop(frame, face_bbox, scale=self.CROP_SCALE, out_size=self.INPUT_SIZE)
        if patch is None:
            return {
                "pad_status": "NOT_EVALUATED",
                "pad_model": self.MODEL_NAME,
                "pad_score": None,
                "pad_threshold": {"bona_fide": self.THRESHOLD_BONA_FIDE, "attack": self.THRESHOLD_ATTACK},
                "pad_class_probs": None,
                "pad_inference_ms": 0.0,
                "pad_quality_status": quality,
                "device": self.device_str,
            }

        try:
            score, pred_cls, probs, inference_ms = self.predict_patch(patch)
        except Exception as exc:
            logger.error(f"[PAD] Inference failure: {exc}")
            return {
                "pad_status": "INCONCLUSIVE",
                "pad_model": self.MODEL_NAME,
                "pad_score": None,
                "pad_threshold": {"bona_fide": self.THRESHOLD_BONA_FIDE, "attack": self.THRESHOLD_ATTACK},
                "pad_class_probs": None,
                "pad_inference_ms": 0.0,
                "pad_quality_status": f"inference_error: {exc}",
                "device": self.device_str,
            }

        # Classification mapping
        # Class 0: Print Attack Spoof
        # Class 1: Bona Fide (Real / Live)
        # Class 2: Screen Replay Attack Spoof
        class_probs = {
            "print_spoof": round(float(probs[0]), 4),
            "bona_fide": round(float(probs[1]), 4),
            "screen_spoof": round(float(probs[2]), 4),
        }

        # Conservative status assignment
        if quality == "truncated_boundary" and score < self.THRESHOLD_BONA_FIDE:
            pad_status = "INCONCLUSIVE"
        elif pred_cls == 1 and score >= self.THRESHOLD_BONA_FIDE:
            pad_status = "BONA_FIDE_CANDIDATE"
        elif pred_cls in {0, 2} or score < self.THRESHOLD_ATTACK:
            pad_status = "ATTACK_SUSPECTED"
        else:
            pad_status = "INCONCLUSIVE"

        return {
            "pad_status": pad_status,
            "pad_model": self.MODEL_NAME,
            "pad_score": round(score, 4),
            "pad_threshold": {"bona_fide": self.THRESHOLD_BONA_FIDE, "attack": self.THRESHOLD_ATTACK},
            "pad_class_probs": class_probs,
            "pad_inference_ms": round(inference_ms, 2),
            "pad_quality_status": quality,
            "device": self.device_str,
        }
