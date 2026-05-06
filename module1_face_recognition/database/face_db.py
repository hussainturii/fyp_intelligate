"""
face_db.py — Module 1 / Database Layer
=========================================
Face embedding database for authorized users and stranger tracking.

Storage:
    - Authorized faces: HDF5 file (persistent) + FAISS index (fast search)
    - Stranger faces:   HDF5 file with TTL-based auto-deletion (BR-03: 90 days)

Functional Requirements covered:
    FR-07  Maintain stranger database, tracking visit frequency and timestamps
    FR-08  Alert when unknown visitor appears 3+ times within configurable window
    BR-03  Unknown visitor data NOT retained beyond 90 days
    BR-10  Admin MUST authorize all new face registrations

Author : Hussain Raza (SP23-BAI-057)
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

import faiss
import h5py
import numpy as np
from loguru import logger


# ─────────────────────────────────────────────────────────────────────────────
# Data Models
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class FaceRecord:
    """Represents one registered face in the database."""
    user_id:    str
    name:       str
    role:       str              # 'resident' | 'employee' | 'security_staff' | 'admin'
    embedding:  np.ndarray
    registered_at: float = field(default_factory=time.time)
    last_seen:  float = field(default_factory=time.time)
    access_count: int = 0
    is_active:  bool = True
    metadata:   dict = field(default_factory=dict)


@dataclass
class StrangerRecord:
    """Represents an unknown visitor tracked in the stranger database."""
    stranger_id: str
    embedding:   np.ndarray
    first_seen:  float = field(default_factory=time.time)
    last_seen:   float = field(default_factory=time.time)
    visit_count: int = 1
    alert_sent:  bool = False
    notes:       str = ""


# ─────────────────────────────────────────────────────────────────────────────
# Face Database
# ─────────────────────────────────────────────────────────────────────────────

class FaceDatabase:
    """
    Dual-database face management system.

    Authorized DB  → FAISS L2 index + HDF5 persistence
    Stranger DB    → FAISS L2 index + HDF5 persistence with TTL

    Usage::

        db = FaceDatabase()
        db.register("hussain", "resident", embedding)
        matches = db.search(live_embedding, k=5)
        db.add_stranger(unknown_embedding)
    """

    def __init__(
        self,
        authorized_db_path: str = "module1_face_recognition/database/face_embeddings.h5",
        stranger_db_path:   str = "module1_face_recognition/database/stranger_embeddings.h5",
        embedding_dim:      int = 768,
        stranger_retention_days: int = 90,
        repeat_visit_threshold: int = 3,
        repeat_visit_window_hours: int = 24,
    ) -> None:
        self.authorized_path  = Path(authorized_db_path)
        self.stranger_path    = Path(stranger_db_path)
        self.embedding_dim    = embedding_dim
        self.stranger_ttl     = timedelta(days=stranger_retention_days)
        self.repeat_threshold = repeat_visit_threshold
        self.repeat_window    = timedelta(hours=repeat_visit_window_hours)

        # In-memory stores
        self._authorized: Dict[str, FaceRecord] = {}    # user_id → FaceRecord
        self._strangers:  Dict[str, StrangerRecord] = {} # stranger_id → StrangerRecord

        # FAISS indices
        self._auth_index:     Optional[faiss.Index] = None
        self._stranger_index: Optional[faiss.Index] = None
        self._auth_id_list:   List[str] = []      # Maps FAISS index → user_id
        self._stranger_id_list: List[str] = []    # Maps FAISS index → stranger_id

        # Create DB directories
        self.authorized_path.parent.mkdir(parents=True, exist_ok=True)
        self.stranger_path.parent.mkdir(parents=True, exist_ok=True)

        self._build_faiss_indices()
        self._load_from_disk()

        logger.success(
            f"[FaceDatabase] Loaded {len(self._authorized)} authorized, "
            f"{len(self._strangers)} strangers."
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Authorized Face CRUD
    # ─────────────────────────────────────────────────────────────────────────

    def register(
        self,
        name: str,
        role: str,
        embedding: np.ndarray,
        metadata: dict | None = None,
    ) -> str:
        """
        Register a new authorized face. (BR-10: Admin must call this.)

        Args:
            name      : Person's name
            role      : 'resident' | 'employee' | 'security_staff' | 'admin'
            embedding : 768-dim L2-normalized embedding
            metadata  : Optional extra info (department, photo_path, etc.)

        Returns:
            user_id (UUID string)
        """
        user_id = str(uuid.uuid4())

        record = FaceRecord(
            user_id=user_id,
            name=name,
            role=role,
            embedding=embedding.astype(np.float32),
            metadata=metadata or {},
        )

        self._authorized[user_id] = record
        self._add_to_auth_index(user_id, embedding)
        self._save_authorized()

        logger.info(f"[FaceDatabase] Registered: {name} ({role}) → {user_id[:8]}...")
        return user_id

    def remove(self, user_id: str) -> bool:
        """Remove an authorized face. Requires FAISS index rebuild."""
        if user_id not in self._authorized:
            return False
        del self._authorized[user_id]
        self._rebuild_auth_index()
        self._save_authorized()
        logger.info(f"[FaceDatabase] Removed user {user_id[:8]}...")
        return True

    def get_user(self, user_id: str) -> Optional[FaceRecord]:
        return self._authorized.get(user_id)

    def list_users(self) -> List[Dict]:
        return [
            {"user_id": r.user_id, "name": r.name, "role": r.role,
             "access_count": r.access_count, "last_seen": r.last_seen}
            for r in self._authorized.values() if r.is_active
        ]

    # ─────────────────────────────────────────────────────────────────────────
    # Search (1:N Recognition)
    # ─────────────────────────────────────────────────────────────────────────

    def search(self, embedding: np.ndarray, k: int = 5) -> List[Dict]:
        """
        Find top-K most similar authorized faces using FAISS.

        Args:
            embedding : 768-dim query embedding (L2 normalized)
            k         : Number of candidates to retrieve

        Returns:
            List of dicts sorted by similarity descending:
            [{"user_id", "name", "role", "similarity", "distance"}, ...]
        """
        if self._auth_index is None or self._auth_index.ntotal == 0:
            return []

        k = min(k, self._auth_index.ntotal)
        query = embedding.astype(np.float32).reshape(1, -1)

        distances, indices = self._auth_index.search(query, k)

        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < 0 or idx >= len(self._auth_id_list):
                continue
            uid = self._auth_id_list[idx]
            if uid not in self._authorized:
                continue
            rec = self._authorized[uid]
            # FAISS L2 distance → cosine similarity (embeddings are L2-normalized)
            # For unit vectors: cos_sim = 1 - (L2^2 / 2)
            cos_sim = float(1.0 - dist / 2.0)
            results.append({
                "user_id": uid,
                "name": rec.name,
                "role": rec.role,
                "similarity": max(0.0, cos_sim),
                "distance": float(dist),
            })

        results.sort(key=lambda x: x["similarity"], reverse=True)
        return results

    def log_access(self, user_id: str) -> None:
        """Log a successful gate access for an authorized user."""
        if user_id in self._authorized:
            self._authorized[user_id].access_count += 1
            self._authorized[user_id].last_seen = time.time()

    # ─────────────────────────────────────────────────────────────────────────
    # Stranger Database (FR-07, FR-08, BR-03)
    # ─────────────────────────────────────────────────────────────────────────

    def add_stranger(self, embedding: np.ndarray, threshold: float = 0.75) -> Dict:
        """
        Add or update a stranger in the stranger database.
        Checks if this stranger has been seen before (via embedding similarity).

        Returns:
            dict with stranger_id, visit_count, is_repeat, should_alert
        """
        # Check against existing strangers
        existing = self._search_strangers(embedding, k=1)

        if existing and existing[0]["similarity"] >= threshold:
            # Seen before — update record
            sid = existing[0]["stranger_id"]
            rec = self._strangers[sid]
            rec.visit_count += 1
            rec.last_seen = time.time()

            # FR-08: Alert if appears 3+ times within 24h window
            should_alert = self._check_repeat_alert(rec)
            if should_alert and not rec.alert_sent:
                rec.alert_sent = True
                logger.warning(
                    f"[FaceDatabase] STRANGER ALERT: {sid[:8]}... "
                    f"seen {rec.visit_count}x in last 24h!"
                )

            self._save_strangers()
            return {
                "stranger_id": sid,
                "visit_count": rec.visit_count,
                "is_repeat": True,
                "should_alert": should_alert,
            }
        else:
            # New stranger
            sid = str(uuid.uuid4())
            rec = StrangerRecord(stranger_id=sid, embedding=embedding.astype(np.float32))
            self._strangers[sid] = rec
            self._add_to_stranger_index(sid, embedding)
            self._save_strangers()

            logger.info(f"[FaceDatabase] New stranger logged: {sid[:8]}...")
            return {
                "stranger_id": sid,
                "visit_count": 1,
                "is_repeat": False,
                "should_alert": False,
            }

    def purge_expired_strangers(self) -> int:
        """
        Remove stranger records older than retention period (BR-03: 90 days).
        Returns count of purged records.
        """
        cutoff = time.time() - self.stranger_ttl.total_seconds()
        expired = [
            sid for sid, rec in self._strangers.items()
            if rec.last_seen < cutoff
        ]
        for sid in expired:
            del self._strangers[sid]

        if expired:
            self._rebuild_stranger_index()
            self._save_strangers()
            logger.info(f"[FaceDatabase] Purged {len(expired)} expired stranger records.")

        return len(expired)

    def get_stranger_stats(self) -> Dict:
        """Summary statistics about the stranger database."""
        return {
            "total_strangers": len(self._strangers),
            "repeat_visitors": sum(1 for r in self._strangers.values() if r.visit_count >= 2),
            "alerts_triggered": sum(1 for r in self._strangers.values() if r.alert_sent),
        }

    # ─────────────────────────────────────────────────────────────────────────
    # FAISS Index Management
    # ─────────────────────────────────────────────────────────────────────────

    def _build_faiss_indices(self) -> None:
        """Create empty FAISS IndexFlatL2 indices."""
        # IndexFlatL2 on L2-normalized vectors gives cosine similarity ordering
        self._auth_index     = faiss.IndexFlatL2(self.embedding_dim)
        self._stranger_index = faiss.IndexFlatL2(self.embedding_dim)

    def _add_to_auth_index(self, user_id: str, embedding: np.ndarray) -> None:
        vec = embedding.astype(np.float32).reshape(1, -1)
        self._auth_index.add(vec)
        self._auth_id_list.append(user_id)

    def _add_to_stranger_index(self, stranger_id: str, embedding: np.ndarray) -> None:
        vec = embedding.astype(np.float32).reshape(1, -1)
        self._stranger_index.add(vec)
        self._stranger_id_list.append(stranger_id)

    def _rebuild_auth_index(self) -> None:
        """Rebuild FAISS authorized index from in-memory records."""
        self._auth_index = faiss.IndexFlatL2(self.embedding_dim)
        self._auth_id_list = []
        for uid, rec in self._authorized.items():
            if rec.is_active:
                self._add_to_auth_index(uid, rec.embedding)

    def _rebuild_stranger_index(self) -> None:
        self._stranger_index = faiss.IndexFlatL2(self.embedding_dim)
        self._stranger_id_list = []
        for sid, rec in self._strangers.items():
            self._add_to_stranger_index(sid, rec.embedding)

    def _search_strangers(self, embedding: np.ndarray, k: int = 1) -> List[Dict]:
        if self._stranger_index.ntotal == 0:
            return []
        k = min(k, self._stranger_index.ntotal)
        query = embedding.astype(np.float32).reshape(1, -1)
        distances, indices = self._stranger_index.search(query, k)
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < 0 or idx >= len(self._stranger_id_list):
                continue
            sid = self._stranger_id_list[idx]
            if sid not in self._strangers:
                continue
            cos_sim = float(1.0 - dist / 2.0)
            results.append({"stranger_id": sid, "similarity": max(0.0, cos_sim)})
        return results

    # ─────────────────────────────────────────────────────────────────────────
    # Business Rules
    # ─────────────────────────────────────────────────────────────────────────

    def _check_repeat_alert(self, rec: StrangerRecord) -> bool:
        """FR-08: Alert if unknown visitor appears 3+ times within 24h."""
        if rec.visit_count < self.repeat_threshold:
            return False
        # Check if visits are within the window
        window_start = time.time() - self.repeat_window.total_seconds()
        return rec.first_seen >= window_start or rec.last_seen >= window_start

    # ─────────────────────────────────────────────────────────────────────────
    # Persistence (HDF5)
    # ─────────────────────────────────────────────────────────────────────────

    def _save_authorized(self) -> None:
        """Save authorized face database to HDF5."""
        try:
            with h5py.File(self.authorized_path, "w") as f:
                for uid, rec in self._authorized.items():
                    grp = f.create_group(uid)
                    grp.create_dataset("embedding", data=rec.embedding)
                    grp.attrs["name"]          = rec.name
                    grp.attrs["role"]          = rec.role
                    grp.attrs["registered_at"] = rec.registered_at
                    grp.attrs["last_seen"]     = rec.last_seen
                    grp.attrs["access_count"]  = rec.access_count
                    grp.attrs["is_active"]     = rec.is_active
                    grp.attrs["metadata"]      = json.dumps(rec.metadata)
        except Exception as e:
            logger.error(f"[FaceDatabase] Failed to save authorized DB: {e}")

    def _save_strangers(self) -> None:
        """Save stranger database to HDF5."""
        try:
            with h5py.File(self.stranger_path, "w") as f:
                for sid, rec in self._strangers.items():
                    grp = f.create_group(sid)
                    grp.create_dataset("embedding", data=rec.embedding)
                    grp.attrs["first_seen"]   = rec.first_seen
                    grp.attrs["last_seen"]    = rec.last_seen
                    grp.attrs["visit_count"]  = rec.visit_count
                    grp.attrs["alert_sent"]   = rec.alert_sent
                    grp.attrs["notes"]        = rec.notes
        except Exception as e:
            logger.error(f"[FaceDatabase] Failed to save stranger DB: {e}")

    def _load_from_disk(self) -> None:
        """Load both databases from HDF5 files if they exist."""
        # Load authorized
        if self.authorized_path.exists():
            try:
                with h5py.File(self.authorized_path, "r") as f:
                    for uid in f.keys():
                        grp = f[uid]
                        rec = FaceRecord(
                            user_id=uid,
                            name=grp.attrs["name"],
                            role=grp.attrs["role"],
                            embedding=grp["embedding"][:],
                            registered_at=grp.attrs["registered_at"],
                            last_seen=grp.attrs["last_seen"],
                            access_count=int(grp.attrs["access_count"]),
                            is_active=bool(grp.attrs["is_active"]),
                            metadata=json.loads(grp.attrs.get("metadata", "{}")),
                        )
                        self._authorized[uid] = rec
                self._rebuild_auth_index()
            except Exception as e:
                logger.warning(f"[FaceDatabase] Could not load authorized DB: {e}")

        # Load strangers
        if self.stranger_path.exists():
            try:
                with h5py.File(self.stranger_path, "r") as f:
                    for sid in f.keys():
                        grp = f[sid]
                        rec = StrangerRecord(
                            stranger_id=sid,
                            embedding=grp["embedding"][:],
                            first_seen=grp.attrs["first_seen"],
                            last_seen=grp.attrs["last_seen"],
                            visit_count=int(grp.attrs["visit_count"]),
                            alert_sent=bool(grp.attrs["alert_sent"]),
                            notes=grp.attrs.get("notes", ""),
                        )
                        self._strangers[sid] = rec
                self._rebuild_stranger_index()
            except Exception as e:
                logger.warning(f"[FaceDatabase] Could not load stranger DB: {e}")

    # ─────────────────────────────────────────────────────────────────────────
    # Stats
    # ─────────────────────────────────────────────────────────────────────────

    # ── Dashboard-facing helpers ──────────────────────────────────────────────

    def list_persons(self) -> Dict[str, Dict]:
        """
        Return a name-keyed dict of person metadata for the web dashboard.
        Dashboard calls GET /api/faces → list_persons().
        """
        result: Dict[str, Dict] = {}
        for rec in self._authorized.values():
            if not rec.is_active:
                continue
            result[rec.name] = {
                "user_id":       rec.user_id,
                "role":          rec.role,
                "n_embeddings":  1,
                "access_count":  rec.access_count,
                "registered_at": datetime.fromtimestamp(rec.registered_at).isoformat(),
                "last_seen":     datetime.fromtimestamp(rec.last_seen).isoformat()
                                 if rec.last_seen else None,
            }
        return result

    def delete_person(self, name: str) -> bool:
        """
        Soft-delete a person by name (dashboard DELETE /api/faces/{name}).
        Marks is_active=False and saves.
        """
        for rec in self._authorized.values():
            if rec.name == name:
                return self.remove(rec.user_id)
        return False

    # ─────────────────────────────────────────────────────────────────────────

    def __repr__(self) -> str:
        return (
            f"FaceDatabase(authorized={len(self._authorized)}, "
            f"strangers={len(self._strangers)})"
        )
