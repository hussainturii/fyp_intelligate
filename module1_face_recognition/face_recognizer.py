"""
face_recognizer.py — Module 1 / FE-1
=======================================
ViT-based face recognition using deep embedding extraction.
Matches live face embeddings against the authorized face database
via cosine similarity (FAISS-accelerated).

Architecture:
    Aligned Face (224×224) → ViT-Base/16 → 768-dim Embedding
    → Cosine Similarity vs. DB → Authorized / Unknown

Author : Hussain Raza (SP23-BAI-057)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger
from PIL import Image
from torchvision import transforms


@dataclass
class RecognitionResult:
    """Result returned by FaceRecognizer.recognize()"""
    identity: str               # 'AUTHORIZED:<name>' | 'UNKNOWN'
    confidence: float           # Cosine similarity score 0–1
    is_authorized: bool
    user_id: Optional[str]      # Database user ID if authorized
    embedding: np.ndarray       # Raw 768-dim embedding (for DB storage)
    match_rank: int = 1         # Rank among top-K candidates


class ViTFaceEmbedder(nn.Module):
    """
    ViT-Base/16 backbone fine-tuned for face embedding extraction.

    Removes the classification head and outputs an L2-normalized
    768-dimensional embedding vector per face.

    We use timm's 'vit_base_patch16_224' pre-trained on ImageNet-21k
    as the backbone. For production, this would be fine-tuned on a
    face dataset (MS-Celeb, VGGFace2, etc.).
    """

    def __init__(self, model_name: str = "vit_base_patch16_224", pretrained: bool = True) -> None:
        super().__init__()
        logger.info(f"[ViTFaceEmbedder] Loading {model_name} (pretrained={pretrained})")

        # Load ViT backbone — remove the classification head
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,       # Remove classifier → output raw CLS token
        )

        self.embedding_dim = self.backbone.num_features  # 768 for ViT-Base
        logger.success(f"[ViTFaceEmbedder] Embedding dim: {self.embedding_dim}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 3, 224, 224) normalized face image tensor
        Returns:
            (B, 768) L2-normalized embedding
        """
        embedding = self.backbone(x)              # (B, 768)
        embedding = F.normalize(embedding, p=2, dim=1)  # L2 normalize
        return embedding


class FaceRecognizer:
    """
    Complete face recognition pipeline.

    Flow:
        PIL Image → preprocess → ViTFaceEmbedder → embedding
        embedding → cosine similarity vs. FaceDatabase → RecognitionResult

    Usage::

        recognizer = FaceRecognizer(device="cuda")
        result = recognizer.recognize(face_pil, face_db)
        if result.is_authorized:
            open_gate()
    """

    # ImageNet mean/std for ViT preprocessing
    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD  = [0.229, 0.224, 0.225]

    def __init__(
        self,
        model_name: str = "vit_base_patch16_224",
        similarity_threshold: float = 0.75,
        device: str = "cuda",
        pretrained_weights: Optional[Path] = None,
    ) -> None:
        """
        Args:
            model_name            : timm model identifier
            similarity_threshold  : Cosine similarity score >= this → authorized (BR-01 analog)
            device                : 'cuda' or 'cpu'
            pretrained_weights    : Path to custom fine-tuned weights (.pth), or None for ImageNet init
        """
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.similarity_threshold = similarity_threshold

        # Build model
        self.model = ViTFaceEmbedder(model_name=model_name, pretrained=True)

        if pretrained_weights and Path(pretrained_weights).exists():
            state = torch.load(pretrained_weights, map_location=self.device)
            self.model.load_state_dict(state)
            logger.info(f"[FaceRecognizer] Loaded fine-tuned weights from {pretrained_weights}")

        self.model.to(self.device)
        self.model.eval()

        self.embedding_dim = self.model.embedding_dim

        # Preprocessing transform for ViT
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.IMAGENET_MEAN, std=self.IMAGENET_STD),
        ])

        logger.success(f"[FaceRecognizer] Ready on {self.device} | threshold={similarity_threshold}")

    # ─────────────────────────────────────────────────────────────────────────
    # Embedding Extraction
    # ─────────────────────────────────────────────────────────────────────────

    @torch.no_grad()
    def extract_embedding(self, face_pil: Image.Image) -> np.ndarray:
        """
        Extract a 768-dim L2-normalized embedding from a face PIL image.

        Args:
            face_pil: RGB PIL image of the detected face

        Returns:
            numpy array of shape (768,)
        """
        if face_pil.mode != "RGB":
            face_pil = face_pil.convert("RGB")

        tensor = self.transform(face_pil).unsqueeze(0).to(self.device)  # (1, 3, 224, 224)
        embedding = self.model(tensor)                                    # (1, 768)
        return embedding.squeeze(0).cpu().numpy()

    @torch.no_grad()
    def extract_batch(self, faces: List[Image.Image]) -> np.ndarray:
        """
        Batch extract embeddings for multiple face images.

        Args:
            faces: List of RGB PIL images

        Returns:
            numpy array of shape (N, 768)
        """
        tensors = torch.stack([
            self.transform(f.convert("RGB")) for f in faces
        ]).to(self.device)                              # (N, 3, 224, 224)

        embeddings = self.model(tensors)                # (N, 768)
        return embeddings.cpu().numpy()

    # ─────────────────────────────────────────────────────────────────────────
    # Recognition
    # ─────────────────────────────────────────────────────────────────────────

    def recognize(
        self,
        face_pil: Image.Image,
        face_db: "FaceDatabase",  # noqa: F821 — forward ref
        top_k: int = 5,
    ) -> RecognitionResult:
        """
        Recognize a face against the authorized face database.

        Args:
            face_pil : Cropped, aligned face image (RGB PIL)
            face_db  : FaceDatabase instance with authorized embeddings
            top_k    : Number of nearest neighbors to retrieve from FAISS

        Returns:
            RecognitionResult with identity, confidence, authorization status
        """
        embedding = self.extract_embedding(face_pil)

        # Query database for top-K nearest neighbors
        matches = face_db.search(embedding, k=top_k)

        if not matches:
            return RecognitionResult(
                identity="UNKNOWN",
                confidence=0.0,
                is_authorized=False,
                user_id=None,
                embedding=embedding,
            )

        # Best match is first result (highest similarity)
        best = matches[0]
        best_sim = best["similarity"]
        best_name = best["name"]
        best_uid = best["user_id"]

        is_authorized = best_sim >= self.similarity_threshold

        return RecognitionResult(
            identity=f"AUTHORIZED:{best_name}" if is_authorized else "UNKNOWN",
            confidence=best_sim,
            is_authorized=is_authorized,
            user_id=best_uid if is_authorized else None,
            embedding=embedding,
            match_rank=1,
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Similarity Utilities
    # ─────────────────────────────────────────────────────────────────────────

    @staticmethod
    def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        """Compute cosine similarity between two L2-normalized embeddings."""
        return float(np.dot(a, b))  # Both are L2-normalized → dot == cosine

    @staticmethod
    def compare_embeddings(
        emb1: np.ndarray,
        emb2: np.ndarray,
        threshold: float = 0.75,
    ) -> Dict:
        """
        Compare two embeddings directly (useful for 1:1 verification).

        Returns dict with similarity, is_same_person, distance.
        """
        sim = float(np.dot(emb1, emb2))
        return {
            "similarity": sim,
            "distance": 1.0 - sim,
            "is_same_person": sim >= threshold,
        }
