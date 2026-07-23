"""
vehicle_db.py — Module 2 / FE-5
==================================
Vehicle database — stores authorized vehicles (plate + Re-ID embedding)
and maintains access history and visit patterns.

Storage:
    Authorized  : HDF5 + FAISS index (plate string + appearance embedding)
    Unknown     : HDF5 + FAISS index (Re-ID embeddings only, no plate)

Lookup priority:
    1. Exact plate match (fastest, most reliable)
    2. Fuzzy plate match (OCR errors: "ABC-123" vs "A8C-123")
    3. Re-ID embedding match (plate obscured/missing)

Author: Hussain Raza (SP23-BAI-057)
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime
from difflib import SequenceMatcher
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
class VehicleRecord:
    """One authorized vehicle in the database."""
    vehicle_id:    str
    plate:         str            # Normalized plate string e.g. "ABC-1234"
    owner_name:    str
    owner_id:      str            # Links to face DB user_id (optional)
    make:          str = ""       # e.g. "Toyota"
    model:         str = ""       # e.g. "Camry"
    color:         str = ""       # e.g. "Silver"
    embedding:     Optional[np.ndarray] = None   # 768-dim Re-ID embedding
    registered_at: float = field(default_factory=time.time)
    last_seen:     float = field(default_factory=time.time)
    access_count:  int = 0
    is_active:     bool = True
    notes:         str = ""

    @property
    def display_name(self) -> str:
        parts = [p for p in [self.make, self.model, self.color] if p]
        return f"{self.plate} ({' '.join(parts)})" if parts else self.plate


@dataclass
class UnknownVehicleRecord:
    """An unknown vehicle tracked in the system."""
    unknown_id:  str
    embedding:   np.ndarray
    first_seen:  float = field(default_factory=time.time)
    last_seen:   float = field(default_factory=time.time)
    visit_count: int = 1
    alert_sent:  bool = False
    last_plate:  str = ""    # last attempted OCR (may be partial/empty)


# ─────────────────────────────────────────────────────────────────────────────
# Database
# ─────────────────────────────────────────────────────────────────────────────

class VehicleDatabase:
    """
    Dual-database vehicle management system.

    Authorized DB  : FAISS + HDF5
    Unknown DB     : FAISS + HDF5 (temporary, no plate match)

    Lookup strategy:
        1. Exact plate string match
        2. Fuzzy plate match (handles 1-char OCR errors)
        3. Re-ID embedding cosine similarity

    Usage::

        db = VehicleDatabase()
        db.register("ABC-1234", "Hussain", make="Toyota", color="Silver")
        result = db.lookup(plate="ABC-1234")
        result = db.search_by_embedding(embedding, k=5)
    """

    def __init__(
        self,
        db_path:          str = "module2_vehicle/database/vehicles.h5",
        unknown_db_path:  str = "module2_vehicle/database/unknown_vehicles.h5",
        embedding_dim:    int = 768,
        reid_threshold:   float = 0.82,
        fuzzy_threshold:  float = 0.80,
        repeat_alert_count: int = 3,
    ) -> None:
        self.db_path         = Path(db_path)
        self.unknown_path    = Path(unknown_db_path)
        self.embedding_dim   = embedding_dim
        self.reid_threshold  = reid_threshold
        self.fuzzy_threshold = fuzzy_threshold
        self.repeat_alert    = repeat_alert_count

        self._vehicles:  Dict[str, VehicleRecord]        = {}
        self._unknowns:  Dict[str, UnknownVehicleRecord] = {}
        self._plate_map: Dict[str, str] = {}   # plate → vehicle_id (fast lookup)

        self._auth_index:     Optional[faiss.Index] = None
        self._unknown_index:  Optional[faiss.Index] = None
        self._auth_id_list:   List[str] = []
        self._unknown_id_list: List[str] = []

        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.unknown_path.parent.mkdir(parents=True, exist_ok=True)

        self._build_indices()
        self._load_from_disk()

        logger.success(
            f"[VehicleDatabase] Loaded {len(self._vehicles)} authorized, "
            f"{len(self._unknowns)} unknown vehicles."
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Registration (Authorized)
    # ─────────────────────────────────────────────────────────────────────────

    def register(
        self,
        plate:      str,
        owner_name: str,
        owner_id:   str = "",
        make:       str = "",
        model:      str = "",
        color:      str = "",
        embedding:  Optional[np.ndarray] = None,
        notes:      str = "",
    ) -> str:
        """
        Register an authorized vehicle.

        Args:
            plate      : Normalized plate string (e.g. "ABC-1234")
            owner_name : Owner's name
            owner_id   : Links to face DB user_id (for combined face+vehicle auth)
            make/model/color : Vehicle metadata
            embedding  : Re-ID embedding (optional — for Re-ID-only matching)
            notes      : Any extra notes

        Returns:
            vehicle_id (UUID string)
        """
        plate = plate.upper().strip()
        vid   = str(uuid.uuid4())

        rec = VehicleRecord(
            vehicle_id=vid,
            plate=plate,
            owner_name=owner_name,
            owner_id=owner_id,
            make=make, model=model, color=color,
            embedding=embedding.astype(np.float32) if embedding is not None else None,
            notes=notes,
        )

        self._vehicles[vid] = rec
        self._plate_map[plate] = vid

        if embedding is not None:
            self._add_to_auth_index(vid, embedding)

        self._save_authorized()
        logger.info(f"[VehicleDatabase] Registered: {plate} ({owner_name}) → {vid[:8]}...")
        return vid

    def remove(self, vehicle_id: str) -> bool:
        if vehicle_id not in self._vehicles:
            return False
        plate = self._vehicles[vehicle_id].plate
        del self._vehicles[vehicle_id]
        self._plate_map.pop(plate, None)
        self._rebuild_auth_index()
        self._save_authorized()
        return True

    # ─────────────────────────────────────────────────────────────────────────
    # Lookup Methods
    # ─────────────────────────────────────────────────────────────────────────

    def lookup_by_plate(self, plate: str) -> Optional[Dict]:
        """
        Look up vehicle by plate string.

        Tries exact match first, then fuzzy match
        (catches 1-character OCR errors like '0' vs 'O').

        Returns vehicle dict or None.
        """
        plate = plate.upper().strip()

        # 1. Exact match
        vid = self._plate_map.get(plate)
        if vid and vid in self._vehicles:
            rec = self._vehicles[vid]
            return self._record_to_dict(rec, method="exact_plate")

        # 2. Fuzzy match (SequenceMatcher ratio)
        best_ratio = 0.0
        best_vid   = None
        for stored_plate, stored_vid in self._plate_map.items():
            ratio = SequenceMatcher(None, plate, stored_plate).ratio()
            if ratio > best_ratio:
                best_ratio = ratio
                best_vid   = stored_vid

        if best_ratio >= self.fuzzy_threshold and best_vid:
            rec = self._vehicles[best_vid]
            logger.debug(
                f"[VehicleDatabase] Fuzzy plate match: "
                f"'{plate}' ≈ '{rec.plate}' (ratio={best_ratio:.2f})"
            )
            return self._record_to_dict(rec, method="fuzzy_plate",
                                         extra={"match_ratio": best_ratio})

        return None

    def search_by_embedding(
        self, embedding: np.ndarray, k: int = 5
    ) -> List[Dict]:
        """
        Find top-K visually similar authorized vehicles using FAISS.

        Used when OCR fails (obscured/dirty plate) — FE-4.

        Returns list of dicts sorted by similarity descending.
        """
        if self._auth_index is None or self._auth_index.ntotal == 0:
            return []

        k     = min(k, self._auth_index.ntotal)
        query = embedding.astype(np.float32).reshape(1, -1)
        distances, indices = self._auth_index.search(query, k)

        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < 0 or idx >= len(self._auth_id_list):
                continue
            vid = self._auth_id_list[idx]
            if vid not in self._vehicles:
                continue
            rec = self._vehicles[vid]
            sim = float(max(0.0, 1.0 - dist / 2.0))
            results.append({
                **self._record_to_dict(rec, method="reid"),
                "similarity": sim,
            })

        results.sort(key=lambda x: x["similarity"], reverse=True)
        return results

    def combined_lookup(
        self,
        plate:     Optional[str],
        embedding: Optional[np.ndarray],
    ) -> Optional[Dict]:
        """
        Combined lookup — plate first, then Re-ID fallback.

        This is the main entry point used by the pipeline.

        Returns:
            dict with vehicle info + 'method' key, or None if not found.
        """
        # 1. Plate lookup (fast)
        if plate:
            result = self.lookup_by_plate(plate)
            if result:
                return result

        # 2. Re-ID fallback (when plate failed or obscured)
        if embedding is not None:
            matches = self.search_by_embedding(embedding, k=1)
            if matches and matches[0]["similarity"] >= self.reid_threshold:
                return matches[0]

        return None

    # ─────────────────────────────────────────────────────────────────────────
    # Unknown Vehicle Tracking
    # ─────────────────────────────────────────────────────────────────────────

    def log_unknown(
        self,
        embedding: np.ndarray,
        plate:     str = "",
        reid_threshold: float = 0.82,
    ) -> Dict:
        """
        Log an unknown vehicle. Detects if it's a repeat visitor.

        Returns:
            dict with unknown_id, visit_count, is_repeat, should_alert
        """
        # Check if seen before
        existing = self._search_unknowns(embedding, k=1)

        if existing and existing[0]["similarity"] >= reid_threshold:
            uid = existing[0]["unknown_id"]
            rec = self._unknowns[uid]
            rec.visit_count += 1
            rec.last_seen    = time.time()
            if plate:
                rec.last_plate = plate

            should_alert = (
                rec.visit_count >= self.repeat_alert and not rec.alert_sent
            )
            if should_alert:
                rec.alert_sent = True
                logger.warning(
                    f"[VehicleDatabase] REPEAT UNKNOWN VEHICLE: "
                    f"{uid[:8]}... seen {rec.visit_count}× "
                    f"(plate: '{plate or 'N/A'}')"
                )

            self._save_unknowns()
            return {
                "unknown_id": uid, "visit_count": rec.visit_count,
                "is_repeat": True, "should_alert": should_alert,
                "last_plate": rec.last_plate,
            }

        # New unknown
        uid = str(uuid.uuid4())
        rec = UnknownVehicleRecord(
            unknown_id=uid,
            embedding=embedding.astype(np.float32),
            last_plate=plate,
        )
        self._unknowns[uid] = rec
        self._add_to_unknown_index(uid, embedding)
        self._save_unknowns()

        logger.info(f"[VehicleDatabase] New unknown vehicle: {uid[:8]}...")
        return {
            "unknown_id": uid, "visit_count": 1,
            "is_repeat": False, "should_alert": False,
            "last_plate": plate,
        }

    def log_access(self, vehicle_id: str) -> None:
        """Increment access counter for an authorized vehicle."""
        if vehicle_id in self._vehicles:
            self._vehicles[vehicle_id].access_count += 1
            self._vehicles[vehicle_id].last_seen = time.time()

    def list_vehicles(self) -> List[Dict]:
        """Return summary list of all authorized vehicles."""
        return [
            {
                "vehicle_id":   r.vehicle_id,
                "plate":        r.plate,
                "owner_name":   r.owner_name,
                "make":         r.make,
                "model":        r.model,
                "color":        r.color,
                "access_count": r.access_count,
                "last_seen":    r.last_seen,
            }
            for r in self._vehicles.values() if r.is_active
        ]

    def get_stats(self) -> Dict:
        return {
            "authorized_vehicles": len(self._vehicles),
            "unknown_vehicles":    len(self._unknowns),
            "repeat_unknowns":     sum(
                1 for r in self._unknowns.values() if r.visit_count >= 2
            ),
            "alerts_triggered":    sum(
                1 for r in self._unknowns.values() if r.alert_sent
            ),
        }

    # ─────────────────────────────────────────────────────────────────────────
    # FAISS
    # ─────────────────────────────────────────────────────────────────────────

    def _build_indices(self) -> None:
        self._auth_index    = faiss.IndexFlatL2(self.embedding_dim)
        self._unknown_index = faiss.IndexFlatL2(self.embedding_dim)

    def _add_to_auth_index(self, vid: str, emb: np.ndarray) -> None:
        self._auth_index.add(emb.astype(np.float32).reshape(1, -1))
        self._auth_id_list.append(vid)

    def _add_to_unknown_index(self, uid: str, emb: np.ndarray) -> None:
        self._unknown_index.add(emb.astype(np.float32).reshape(1, -1))
        self._unknown_id_list.append(uid)

    def _rebuild_auth_index(self) -> None:
        self._auth_index = faiss.IndexFlatL2(self.embedding_dim)
        self._auth_id_list = []
        for vid, rec in self._vehicles.items():
            if rec.is_active and rec.embedding is not None:
                self._add_to_auth_index(vid, rec.embedding)

    def _search_unknowns(self, emb: np.ndarray, k: int = 1) -> List[Dict]:
        if self._unknown_index.ntotal == 0:
            return []
        k     = min(k, self._unknown_index.ntotal)
        query = emb.astype(np.float32).reshape(1, -1)
        dists, idxs = self._unknown_index.search(query, k)
        results = []
        for dist, idx in zip(dists[0], idxs[0]):
            if idx < 0 or idx >= len(self._unknown_id_list):
                continue
            uid = self._unknown_id_list[idx]
            sim = float(max(0.0, 1.0 - dist / 2.0))
            results.append({"unknown_id": uid, "similarity": sim})
        return results

    # ─────────────────────────────────────────────────────────────────────────
    # Persistence
    # ─────────────────────────────────────────────────────────────────────────

    def _save_authorized(self) -> None:
        try:
            with h5py.File(self.db_path, "w") as f:
                for vid, rec in self._vehicles.items():
                    grp = f.create_group(vid)
                    if rec.embedding is not None:
                        grp.create_dataset("embedding", data=rec.embedding)
                    for attr in ["plate", "owner_name", "owner_id",
                                 "make", "model", "color", "notes"]:
                        grp.attrs[attr] = getattr(rec, attr)
                    grp.attrs["registered_at"] = rec.registered_at
                    grp.attrs["last_seen"]      = rec.last_seen
                    grp.attrs["access_count"]   = rec.access_count
                    grp.attrs["is_active"]      = rec.is_active
        except Exception as e:
            logger.error(f"[VehicleDatabase] Save error: {e}")

    def _save_unknowns(self) -> None:
        try:
            with h5py.File(self.unknown_path, "w") as f:
                for uid, rec in self._unknowns.items():
                    grp = f.create_group(uid)
                    grp.create_dataset("embedding", data=rec.embedding)
                    grp.attrs["first_seen"]   = rec.first_seen
                    grp.attrs["last_seen"]    = rec.last_seen
                    grp.attrs["visit_count"]  = rec.visit_count
                    grp.attrs["alert_sent"]   = rec.alert_sent
                    grp.attrs["last_plate"]   = rec.last_plate
        except Exception as e:
            logger.error(f"[VehicleDatabase] Unknown save error: {e}")

    def _load_from_disk(self) -> None:
        if self.db_path.exists():
            try:
                with h5py.File(self.db_path, "r") as f:
                    for vid in f.keys():
                        grp = f[vid]
                        emb = grp["embedding"][:] if "embedding" in grp else None
                        rec = VehicleRecord(
                            vehicle_id=vid,
                            plate=grp.attrs["plate"],
                            owner_name=grp.attrs["owner_name"],
                            owner_id=grp.attrs.get("owner_id", ""),
                            make=grp.attrs.get("make", ""),
                            model=grp.attrs.get("model", ""),
                            color=grp.attrs.get("color", ""),
                            embedding=emb,
                            registered_at=grp.attrs["registered_at"],
                            last_seen=grp.attrs["last_seen"],
                            access_count=int(grp.attrs["access_count"]),
                            is_active=bool(grp.attrs["is_active"]),
                            notes=grp.attrs.get("notes", ""),
                        )
                        self._vehicles[vid] = rec
                        self._plate_map[rec.plate] = vid
                self._rebuild_auth_index()
            except Exception as e:
                logger.warning(f"[VehicleDatabase] Load error: {e}")

        if self.unknown_path.exists():
            try:
                with h5py.File(self.unknown_path, "r") as f:
                    for uid in f.keys():
                        grp = f[uid]
                        rec = UnknownVehicleRecord(
                            unknown_id=uid,
                            embedding=grp["embedding"][:],
                            first_seen=grp.attrs["first_seen"],
                            last_seen=grp.attrs["last_seen"],
                            visit_count=int(grp.attrs["visit_count"]),
                            alert_sent=bool(grp.attrs["alert_sent"]),
                            last_plate=grp.attrs.get("last_plate", ""),
                        )
                        self._unknowns[uid] = rec
                # Rebuild unknown FAISS index
                self._unknown_index = faiss.IndexFlatL2(self.embedding_dim)
                self._unknown_id_list = []
                for uid, rec in self._unknowns.items():
                    self._add_to_unknown_index(uid, rec.embedding)
            except Exception as e:
                logger.warning(f"[VehicleDatabase] Unknown load error: {e}")

    # ─────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _record_to_dict(
        rec: VehicleRecord,
        method: str = "exact_plate",
        extra: Dict = None,
    ) -> Dict:
        d = {
            "vehicle_id":   rec.vehicle_id,
            "plate":        rec.plate,
            "owner_name":   rec.owner_name,
            "owner_id":     rec.owner_id,
            "make":         rec.make,
            "model":        rec.model,
            "color":        rec.color,
            "access_count": rec.access_count,
            "last_seen":    rec.last_seen,
            "method":       method,
            "similarity":   1.0 if "plate" in method else 0.0,
        }
        if extra:
            d.update(extra)
        return d

    # ── Dashboard-facing helpers ──────────────────────────────────────────────

    def list_vehicles_dict(self) -> Dict[str, Dict]:
        """
        Return a plate-keyed dict of vehicle metadata for the web dashboard.
        Dashboard calls GET /api/vehicles → list_vehicles_dict().
        """
        result: Dict[str, Dict] = {}
        for rec in self._vehicles.values():
            if not rec.is_active:
                continue
            result[rec.plate] = {
                "vehicle_id":    rec.vehicle_id,
                "owner":         rec.owner_name,
                "make":          rec.make,
                "model":         rec.model,
                "color":         rec.color,
                "category":      "authorized",
                "access_count":  rec.access_count,
                "visit_count":   rec.access_count,
                "registered_at": datetime.fromtimestamp(rec.registered_at).isoformat()
                                 if rec.registered_at else None,
                "last_seen":     datetime.fromtimestamp(rec.last_seen).isoformat()
                                 if rec.last_seen else None,
            }
        return result

    def delete_vehicle(self, plate: str) -> bool:
        """
        Remove a vehicle by plate number (dashboard DELETE /api/vehicles/{plate}).
        """
        for rec in self._vehicles.values():
            if rec.plate == plate:
                return self.remove(rec.vehicle_id)
        return False

    # ─────────────────────────────────────────────────────────────────────────

    def __repr__(self) -> str:
        return (
            f"VehicleDatabase(authorized={len(self._vehicles)}, "
            f"unknown={len(self._unknowns)})"
        )
