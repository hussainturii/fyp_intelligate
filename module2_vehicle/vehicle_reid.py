"""
vehicle_reid.py — Module 2 / Deep Metric Vehicle Re-Identification
==================================================================
Deep Vehicle Re-Identification using the genuine Intel Open Model Zoo
vehicle-reid-0001 (OSNet-AIN-x1.0) architecture, pre-trained and
evaluated on the VeRi-776 vehicle benchmark.

Key Properties:
- Architecture: OSNet-AIN (Omni-Scale Network with Adaptive Instance Normalization)
- Parameter Count: 2.183M parameters
- Input Format: RGB whole vehicle body crop, 208×208, letterboxed
- Output Embedding: 512-dimensional float32 L2-normalized metric vector
- Benchmark (VeRi-776 published): Rank-1: 96.31%, mAP: 85.15%
- Execution Engine: OpenCV DNN with automatic hardware target configuration
- Advisory Invariant: Vehicle Re-ID remains advisory-only; it cannot
  independently grant autonomous gate access.

Reference:
- Zhou et al., "Omni-Scale Feature Learning for Person Re-Identification", ICCV 2019.
- Intel Open Model Zoo: vehicle-reid-0001 (MIT License).
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
from loguru import logger
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_PATH = ROOT / "models" / "osnet_ain_x1_0_vehicle_reid.onnx"


# ─────────────────────────────────────────────────────────────────────────────
# Re-ID Result Data Model
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ReIDResult:
    """Result from a vehicle Re-ID appearance query."""
    matched: bool
    vehicle_id: Optional[str]  # DB vehicle ID if matched
    plate: Optional[str]  # Plate from DB record if matched
    similarity: float  # Cosine similarity [0.0, 1.0]
    embedding: np.ndarray  # Raw 512-D L2-normalized embedding
    method: str = "deep_reid"  # 'exact_plate' | 'deep_reid' | 'appearance_histogram_baseline'
    model_id: str = "osnet_ain_x1_0_vehicle_reid"
    embedding_dim: int = 512
    meta: Dict[str, Any] = field(default_factory=dict)


# ─────────────────────────────────────────────────────────────────────────────
# Preprocessing Utilities
# ─────────────────────────────────────────────────────────────────────────────

def letterbox_vehicle(
    img: np.ndarray,
    target_size: Tuple[int, int] = (208, 208),
    pad_color: Tuple[int, int, int] = (124, 116, 104),
) -> np.ndarray:
    """
    Resize image to target_size while strictly preserving original aspect ratio.
    Pads the shorter dimension symmetrically with pad_color (neutral gray).

    Args:
        img: Input image (H, W, C) in BGR or RGB.
        target_size: (width, height) target dimensions.
        pad_color: Tuple of 3 values for padding borders.

    Returns:
        Padded image of shape (target_size[1], target_size[0], C).
    """
    h, w = img.shape[:2]
    target_w, target_h = target_size
    scale = min(target_w / w, target_h / h)
    nw, nh = int(round(w * scale)), int(round(h * scale))

    if nw != w or nh != h:
        resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    else:
        resized = img

    pad_w = target_w - nw
    pad_h = target_h - nh
    top = pad_h // 2
    bottom = pad_h - top
    left = pad_w // 2
    right = pad_w - left

    return cv2.copyMakeBorder(
        resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=pad_color
    )


def color_histogram_baseline(frame: np.ndarray) -> np.ndarray:
    """
    768-D concatenated RGB color histogram baseline.
    Preserved for experimental comparison against deep metric Re-ID.
    """
    if frame is None or frame.size == 0:
        return np.zeros(768, dtype=np.float32)
    vector = np.concatenate([
        cv2.calcHist([frame], [i], None, [256], [0, 256]).ravel()
        for i in range(3)
    ])
    norm = np.linalg.norm(vector)
    return (vector / max(norm, 1e-12)).astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# OSNet Vehicle Re-ID Feature Extractor
# ─────────────────────────────────────────────────────────────────────────────

class OSNetVehicleReID:
    """
    Modular deep metric vehicle feature extractor using the official
    Intel Open Model Zoo OSNet-AIN checkpoint (vehicle-reid-0001).

    Produces 512-dimensional L2-normalized appearance embeddings.
    """

    IMAGENET_RGB_MEAN = np.array([123.675, 116.28, 103.53], dtype=np.float32)
    IMAGENET_RGB_STD = np.array([58.395, 57.12, 57.375], dtype=np.float32)
    INPUT_SIZE = (208, 208)  # (width, height)
    EMBEDDING_DIM = 512
    MODEL_ID = "osnet_ain_x1_0_vehicle_reid"

    def __init__(
        self,
        weights_path: Optional[Union[str, Path]] = None,
        similarity_threshold: float = 0.70,
        device: str = "cpu",
        verify_integrity: bool = True,
        resize_mode: str = "direct",
    ) -> None:
        """
        Initialize the vehicle Re-ID extractor.

        Args:
            weights_path: Path to osnet_ain_x1_0_vehicle_reid.onnx.
            similarity_threshold: Operating cosine threshold for advisory match.
            device: 'cpu' or 'cuda'.
            verify_integrity: If True, validates SHA-256 against trusted manifest.
            resize_mode: 'direct' (canonical VeRi-776 training resize) or 'letterbox'.
        """
        self.weights_path = Path(weights_path) if weights_path else DEFAULT_MODEL_PATH
        self.threshold = similarity_threshold
        self.requested_device = device.lower()
        self.embedding_dim = self.EMBEDDING_DIM
        self.model_id = self.MODEL_ID
        self.resize_mode = resize_mode.lower()

        if not self.weights_path.exists():
            raise FileNotFoundError(
                f"[VehicleReID] Weights not found at {self.weights_path}. "
                f"Ensure osnet_ain_x1_0_vehicle_reid.onnx is downloaded."
            )

        if verify_integrity:
            self._verify_weights_integrity()

        # Load model using OpenCV DNN
        logger.info(f"[VehicleReID] Loading {self.model_id} from {self.weights_path}...")
        try:
            self.net = cv2.dnn.readNetFromONNX(str(self.weights_path))
        except Exception as exc:
            raise RuntimeError(
                f"[VehicleReID] Failed to load ONNX model via OpenCV DNN: {exc}"
            ) from exc

        # Configure execution target
        self.active_backend = "opencv"
        self.active_target = "cpu"
        if self.requested_device in {"cuda", "gpu"}:
            try:
                self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
                self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
                self.active_backend = "cuda"
                self.active_target = "cuda"
                logger.info("[VehicleReID] Configured for CUDA execution.")
            except Exception as e:
                logger.warning(f"[VehicleReID] CUDA backend unavailable ({e}); falling back to CPU.")
                self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
                self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
                self.active_backend = "opencv"
                self.active_target = "cpu"
        else:
            self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
            self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

        # Warm up model once to prime graph execution
        self._warmup()
        logger.success(
            f"[VehicleReID] Ready | dim={self.embedding_dim} | "
            f"target={self.active_target} | threshold={self.threshold:.2f}"
        )

    def _verify_weights_integrity(self) -> None:
        """Verify the weights file SHA-256 against models/trusted_manifest.json."""
        try:
            from module8_dashboard.backend.model_integrity import verify_model_file
            rel_path = "models/osnet_ain_x1_0_vehicle_reid.onnx"
            res = verify_model_file(rel_path, strict=False)
            if not res.get("verified"):
                logger.warning(
                    f"[VehicleReID] Model integrity warning for {rel_path}: {res.get('status')}"
                )
        except Exception as exc:
            logger.debug(f"[VehicleReID] Optional manifest check skipped: {exc}")

    def _warmup(self) -> None:
        """Prime OpenCV DNN graph execution with dummy tensor."""
        dummy = np.zeros((1, 3, self.INPUT_SIZE[1], self.INPUT_SIZE[0]), dtype=np.float32)
        try:
            self.net.setInput(dummy)
            _ = self.net.forward()
        except Exception as exc:
            logger.debug(f"[VehicleReID] Warmup notice: {exc}")

    # ─────────────────────────────────────────────────────────────────────────
    # Preprocessing
    # ─────────────────────────────────────────────────────────────────────────

    def preprocess(
        self,
        crop: Union[np.ndarray, Image.Image],
        resize_mode: Optional[str] = None,
    ) -> np.ndarray:
        """
        Validate, letterbox, normalize, and transpose vehicle crop to NCHW blob.

        Args:
            crop: BGR or RGB numpy array, or PIL Image.

        Returns:
            np.ndarray of shape (1, 3, 208, 208) in float32.
        """
        if crop is None:
            raise ValueError("[VehicleReID] Input crop cannot be None.")

        if isinstance(crop, Image.Image):
            crop_rgb = np.array(crop.convert("RGB"))
        elif isinstance(crop, np.ndarray):
            if crop.size == 0 or crop.shape[0] < 5 or crop.shape[1] < 5:
                raise ValueError(f"[VehicleReID] Invalid crop dimensions: {crop.shape}")
            # Detect whether BGR or already RGB
            if len(crop.shape) == 2:
                crop_rgb = cv2.cvtColor(crop, cv2.COLOR_GRAY2RGB)
            elif crop.shape[2] == 3:
                # In IntelliGate vision pipeline, OpenCV frames are BGR
                crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            elif crop.shape[2] == 4:
                crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGRA2RGB)
            else:
                raise ValueError(f"[VehicleReID] Unsupported channel count: {crop.shape[2]}")
        else:
            raise TypeError(f"[VehicleReID] Unsupported input type: {type(crop)}")

        # Apply resize strategy
        target_mode = (resize_mode or self.resize_mode).lower()
        if target_mode == "letterbox":
            processed_rgb = letterbox_vehicle(crop_rgb, target_size=self.INPUT_SIZE)
        else:
            # Canonical VeRi-776 training and OMZ evaluation standard: direct bilinear resize
            processed_rgb = cv2.resize(crop_rgb, self.INPUT_SIZE, interpolation=cv2.INTER_LINEAR)

        # Normalize with ImageNet mean and standard deviation
        normed = (processed_rgb.astype(np.float32) - self.IMAGENET_RGB_MEAN) / self.IMAGENET_RGB_STD

        # Transpose HWC -> CHW and add batch dimension -> NCHW (1, 3, 208, 208)
        blob = np.transpose(normed, (2, 0, 1))[np.newaxis, ...]
        return blob

    # ─────────────────────────────────────────────────────────────────────────
    # Feature Extraction
    # ─────────────────────────────────────────────────────────────────────────

    def extract_embedding(
        self,
        vehicle_crop: Union[np.ndarray, Image.Image],
        resize_mode: Optional[str] = None,
    ) -> np.ndarray:
        """
        Extract a 512-dimensional, L2-normalized deep metric embedding.

        Args:
            vehicle_crop: Vehicle image crop (BGR numpy array or PIL Image).
            resize_mode: Optional override ('direct' or 'letterbox').

        Returns:
            np.ndarray of shape (512,), float32, unit L2 norm.
        """
        blob = self.preprocess(vehicle_crop, resize_mode=resize_mode)
        self.net.setInput(blob)
        raw_feat = self.net.forward()  # Shape: (1, 512)
        feat = raw_feat[0].astype(np.float32)

        # L2-normalize feature vector
        norm = float(np.linalg.norm(feat))
        if norm > 1e-12:
            feat = feat / norm
        else:
            feat = np.zeros(self.EMBEDDING_DIM, dtype=np.float32)

        return feat

    def extract_batch(self, crops: List[Union[np.ndarray, Image.Image]]) -> np.ndarray:
        """Extract embeddings for a list of crops. Returns (N, 512) array."""
        if not crops:
            return np.empty((0, self.EMBEDDING_DIM), dtype=np.float32)
        embeddings = [self.extract_embedding(c) for c in crops]
        return np.vstack(embeddings)

    def extract_histogram_baseline(self, vehicle_crop: np.ndarray) -> np.ndarray:
        """Extract the legacy 768-D color histogram baseline vector."""
        return color_histogram_baseline(vehicle_crop)

    # ─────────────────────────────────────────────────────────────────────────
    # Comparison & Identification
    # ─────────────────────────────────────────────────────────────────────────

    @staticmethod
    def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        """Cosine similarity between two normalized embeddings."""
        if a.shape != b.shape:
            raise ValueError(
                f"[VehicleReID] Dimension mismatch for cosine similarity: {a.shape} vs {b.shape}"
            )
        return float(np.dot(a, b))

    def compare(self, emb1: np.ndarray, emb2: np.ndarray) -> Dict[str, Any]:
        """
        1:1 verification between two appearance embeddings.
        Returns similarity, distance, and advisory match decision.
        """
        sim = self.cosine_similarity(emb1, emb2)
        return {
            "similarity": round(sim, 4),
            "distance": round(1.0 - sim, 4),
            "is_same": bool(sim >= self.threshold),
            "threshold": self.threshold,
            "dimension": int(emb1.shape[0]),
            "model_id": self.model_id,
            "advisory_only": True,
        }

    def identify(
        self,
        vehicle_crop: Union[np.ndarray, Image.Image],
        vehicle_db: Any,
        top_k: int = 5,
    ) -> ReIDResult:
        """
        Search for matching authorized vehicle appearance in the database.

        Args:
            vehicle_crop: Vehicle image crop.
            vehicle_db: VehicleDatabase instance with FAISS index.
            top_k: Number of nearest candidates to inspect.

        Returns:
            ReIDResult (advisory match).
        """
        try:
            embedding = self.extract_embedding(vehicle_crop)
        except Exception as exc:
            logger.warning(f"[VehicleReID] Failed to extract embedding: {exc}")
            return ReIDResult(
                matched=False,
                vehicle_id=None,
                plate=None,
                similarity=0.0,
                embedding=np.zeros(self.EMBEDDING_DIM, dtype=np.float32),
                method="deep_reid",
                meta={"error": str(exc)},
            )

        matches = vehicle_db.search_by_embedding(embedding, k=top_k)
        if not matches:
            return ReIDResult(
                matched=False,
                vehicle_id=None,
                plate=None,
                similarity=0.0,
                embedding=embedding,
                method="deep_reid",
            )

        best = matches[0]
        sim = best.get("similarity", 0.0)
        is_match = sim >= self.threshold

        return ReIDResult(
            matched=is_match,
            vehicle_id=best.get("vehicle_id") if is_match else None,
            plate=best.get("plate") if is_match else None,
            similarity=float(sim),
            embedding=embedding,
            method="deep_reid",
            meta={
                "owner_name": best.get("owner_name", ""),
                "top_candidates": len(matches),
            },
        )


# Backward-compatible alias
VehicleReID = OSNetVehicleReID
