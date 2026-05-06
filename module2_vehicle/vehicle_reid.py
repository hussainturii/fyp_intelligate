"""
vehicle_reid.py — Module 2 / FE-2, FE-3, FE-4
================================================
Vehicle Re-Identification by visual appearance using Vision Transformers.

Why Re-ID matters (FE-4):
    When a vehicle's license plate is obscured, dirty, or deliberately
    covered, OCR fails. Re-ID matches vehicles by their appearance
    (colour, shape, make, model silhouette) using deep embeddings —
    the same ViT approach used for face recognition in Module 1.

Architecture:
    Vehicle crop (224×224) → ViT-Base/16 → 768-dim L2-norm embedding
    → Cosine similarity vs. VehicleDatabase → Re-ID match

Training note:
    ViT is pretrained on ImageNet-21k (not vehicle-specific).
    For production, fine-tune on VeRi-776 or CityFlow using
    triplet / ArcFace loss to learn vehicle identity embeddings.

    Reference: Liu et al., "VeRi-776: A Large Scale Benchmark
    Dataset for Vehicle Re-Identification", ECCV 2016.
    arXiv: https://arxiv.org/abs/1609.01475

Author: Hussain Raza (SP23-BAI-057)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger
from PIL import Image
from torchvision import transforms


# ─────────────────────────────────────────────────────────────────────────────
# ViT Embedding Backbone
# ─────────────────────────────────────────────────────────────────────────────

class ViTVehicleEmbedder(nn.Module):
    """
    ViT-Base/16 backbone for vehicle appearance embedding.

    Identical architecture to Module 1's face embedder —
    remove classification head, output L2-normalised CLS token.

    For vehicle Re-ID, the embedding captures:
        - Body colour and finish
        - Silhouette shape (sedan vs SUV vs truck)
        - Front grille pattern, headlight shape
        - Distinguishing markings / stickers
    """

    def __init__(
        self,
        model_name: str = "vit_base_patch16_224",
        pretrained: bool = True,
    ) -> None:
        super().__init__()
        self.backbone = timm.create_model(
            model_name, pretrained=pretrained, num_classes=0
        )
        self.embedding_dim = self.backbone.num_features  # 768

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        emb = self.backbone(x)                    # (B, 768)
        return F.normalize(emb, p=2, dim=1)       # L2 normalise


# ─────────────────────────────────────────────────────────────────────────────
# Re-ID Result
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ReIDResult:
    """Result from a vehicle Re-ID query."""
    matched:      bool
    vehicle_id:   Optional[str]   # DB vehicle ID if matched
    plate:        Optional[str]   # Plate from DB record if matched
    similarity:   float           # Cosine similarity 0–1
    embedding:    np.ndarray      # Raw embedding (for DB storage)
    method:       str = "reid"    # 'plate' | 'reid' | 'combined'


# ─────────────────────────────────────────────────────────────────────────────
# Main Re-ID Class
# ─────────────────────────────────────────────────────────────────────────────

class VehicleReID:
    """
    Vehicle Re-Identification pipeline.

    Extracts a 768-dim appearance embedding from a vehicle crop and
    matches it against the VehicleDatabase using cosine similarity.

    Used as the fallback when OCR fails (obscured plate) — FE-4.

    Usage::

        reid = VehicleReID(device="cuda")
        emb  = reid.extract_embedding(vehicle_crop_pil)
        result = reid.identify(vehicle_crop_pil, vehicle_db)
        if result.matched:
            print("Known vehicle:", result.plate)
    """

    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD  = [0.229, 0.224, 0.225]

    def __init__(
        self,
        model_name: str = "vit_base_patch16_224",
        similarity_threshold: float = 0.82,
        device: str = "cuda",
        weights_path: Optional[Path] = None,
    ) -> None:
        """
        Args:
            model_name            : timm model identifier
            similarity_threshold  : Cosine sim >= this → Re-ID match
                                    (higher than face threshold — vehicles are
                                     more visually similar to each other)
            device                : 'cuda' or 'cpu'
            weights_path          : Fine-tuned weights (VeRi-776) or None
        """
        self.device    = torch.device(device if torch.cuda.is_available() else "cpu")
        self.threshold = similarity_threshold

        logger.info(f"[VehicleReID] Loading {model_name} on {self.device}...")
        self.model = ViTVehicleEmbedder(model_name=model_name, pretrained=True)

        if weights_path and Path(weights_path).exists():
            state = torch.load(weights_path, map_location=self.device)
            self.model.load_state_dict(state)
            logger.info(f"[VehicleReID] Loaded fine-tuned weights: {weights_path}")
        else:
            logger.warning(
                "[VehicleReID] No fine-tuned weights (VeRi-776). "
                "Using ImageNet-pretrained ViT — works for same-camera matching, "
                "weaker across different camera angles."
            )

        self.model.to(self.device).eval()
        self.embedding_dim = self.model.embedding_dim

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.IMAGENET_MEAN, std=self.IMAGENET_STD),
        ])

        logger.success(
            f"[VehicleReID] Ready | threshold={similarity_threshold} | "
            f"device={self.device}"
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Embedding Extraction
    # ─────────────────────────────────────────────────────────────────────────

    @torch.no_grad()
    def extract_embedding(self, vehicle_pil: Image.Image) -> np.ndarray:
        """
        Extract 768-dim L2-normalised appearance embedding.

        Args:
            vehicle_pil : RGB PIL image of the vehicle crop

        Returns:
            numpy array shape (768,)
        """
        if vehicle_pil.mode != "RGB":
            vehicle_pil = vehicle_pil.convert("RGB")
        tensor = self.transform(vehicle_pil).unsqueeze(0).to(self.device)
        emb = self.model(tensor)
        return emb.squeeze(0).cpu().numpy()

    @torch.no_grad()
    def extract_batch(self, vehicles: List[Image.Image]) -> np.ndarray:
        """Batch extract embeddings. Returns (N, 768) array."""
        tensors = torch.stack([
            self.transform(v.convert("RGB")) for v in vehicles
        ]).to(self.device)
        return self.model(tensors).cpu().numpy()

    # ─────────────────────────────────────────────────────────────────────────
    # Identification
    # ─────────────────────────────────────────────────────────────────────────

    def identify(
        self,
        vehicle_pil: Image.Image,
        vehicle_db: "VehicleDatabase",   # noqa: F821
        top_k: int = 5,
    ) -> ReIDResult:
        """
        Identify a vehicle by appearance against the database.

        Args:
            vehicle_pil : RGB PIL image of the vehicle
            vehicle_db  : VehicleDatabase instance
            top_k       : FAISS top-K candidates

        Returns:
            ReIDResult
        """
        embedding = self.extract_embedding(vehicle_pil)
        matches   = vehicle_db.search_by_embedding(embedding, k=top_k)

        if not matches:
            return ReIDResult(
                matched=False, vehicle_id=None, plate=None,
                similarity=0.0, embedding=embedding,
            )

        best = matches[0]
        sim  = best["similarity"]

        return ReIDResult(
            matched=sim >= self.threshold,
            vehicle_id=best["vehicle_id"] if sim >= self.threshold else None,
            plate=best.get("plate") if sim >= self.threshold else None,
            similarity=sim,
            embedding=embedding,
            method="reid",
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Utilities
    # ─────────────────────────────────────────────────────────────────────────

    @staticmethod
    def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        """Cosine similarity between two L2-normalised embeddings."""
        return float(np.dot(a, b))

    def compare(
        self,
        emb1: np.ndarray,
        emb2: np.ndarray,
    ) -> Dict:
        """1:1 verification between two embeddings."""
        sim = self.cosine_similarity(emb1, emb2)
        return {
            "similarity": sim,
            "distance":   1.0 - sim,
            "is_same":    sim >= self.threshold,
        }
