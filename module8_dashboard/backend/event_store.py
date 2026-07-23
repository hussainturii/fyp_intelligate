"""
event_store.py — SQLite Event & Alert Store
============================================
Lightweight SQLite database for persisting all gate events and alerts.
Each recognition event (face or vehicle) is stored here for the dashboard
to query, filter, and export.

Schema:
    events  — one row per recognition attempt
    alerts  — security alerts (unknown repeat visitors, spoofing, etc.)
    system  — periodic system health snapshots

Author: Hussain Raza (SP23-BAI-057)
"""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional

from loguru import logger


# ─────────────────────────────────────────────────────────────────────────────
# Schema SQL
# ─────────────────────────────────────────────────────────────────────────────

_SCHEMA = """
CREATE TABLE IF NOT EXISTS events (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp   TEXT    NOT NULL DEFAULT (datetime('now', 'localtime')),
    module      TEXT    NOT NULL,   -- 'face' | 'vehicle'
    event_type  TEXT    NOT NULL,   -- 'authorized' | 'stranger' | 'spoof' | 'known' | 'unknown'
    identity    TEXT,               -- person name or plate number
    confidence  REAL,               -- similarity score 0.0–1.0
    method      TEXT,               -- 'face_recognition' | 'plate' | 'reid' | 'combined'
    camera_id   INTEGER DEFAULT 0,
    image_path  TEXT,               -- path to snapshot (optional)
    meta        TEXT                -- JSON blob for extra fields
);

CREATE TABLE IF NOT EXISTS alerts (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp   TEXT    NOT NULL DEFAULT (datetime('now', 'localtime')),
    severity    TEXT    NOT NULL,   -- 'LOW' | 'MEDIUM' | 'HIGH' | 'CRITICAL'
    category    TEXT    NOT NULL,   -- 'repeat_visitor' | 'spoof_attempt' | 'forced_entry' | 'system'
    title       TEXT    NOT NULL,
    description TEXT,
    resolved    INTEGER DEFAULT 0,  -- 0=open, 1=resolved
    resolved_at TEXT,
    event_id    INTEGER REFERENCES events(id)
);

CREATE TABLE IF NOT EXISTS system_health (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp   TEXT    NOT NULL DEFAULT (datetime('now', 'localtime')),
    cpu_pct     REAL,
    mem_pct     REAL,
    gpu_pct     REAL,
    fps         REAL,
    cam_online  INTEGER DEFAULT 1
);

CREATE INDEX IF NOT EXISTS idx_events_timestamp ON events(timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_events_module    ON events(module);
CREATE INDEX IF NOT EXISTS idx_alerts_resolved  ON alerts(resolved);
"""


# ─────────────────────────────────────────────────────────────────────────────
# Event Store Class
# ─────────────────────────────────────────────────────────────────────────────

class EventStore:
    """
    SQLite-backed store for IntelliGate events, alerts, and health data.

    Thread-safe: uses `check_same_thread=False` and a connection-per-call
    pattern (calls are typically async-thread-dispatched by FastAPI).

    Usage::

        store = EventStore("logs/events.db")
        event_id = store.log_event("face", "authorized", identity="Hussain", confidence=0.91)
        store.log_alert("HIGH", "repeat_visitor", "Unknown face seen 3× in 24h")
        events = store.get_events(module="face", limit=50)
    """

    def __init__(self, db_path: str = "logs/events.db") -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()
        logger.info(f"[EventStore] SQLite at {self.db_path}")

    # ── Connection helper ─────────────────────────────────────────────────────

    @contextmanager
    def _conn(self) -> Generator[sqlite3.Connection, None, None]:
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _init_schema(self) -> None:
        with self._conn() as conn:
            conn.executescript(_SCHEMA)

    # ── Event logging ─────────────────────────────────────────────────────────

    def log_event(
        self,
        module: str,
        event_type: str,
        identity: Optional[str] = None,
        confidence: Optional[float] = None,
        method: Optional[str] = None,
        camera_id: int = 0,
        image_path: Optional[str] = None,
        meta: Optional[str] = None,
    ) -> int:
        """Insert a recognition event. Returns the new row ID."""
        with self._conn() as conn:
            cur = conn.execute(
                """INSERT INTO events
                   (module, event_type, identity, confidence, method, camera_id, image_path, meta)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (module, event_type, identity, confidence, method, camera_id, image_path, meta),
            )
            return cur.lastrowid

    def get_events(
        self,
        module: Optional[str] = None,
        event_type: Optional[str] = None,
        since_hours: Optional[int] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[Dict[str, Any]]:
        """Fetch events with optional filters."""
        conditions = []
        params: List[Any] = []

        if module:
            conditions.append("module = ?")
            params.append(module)
        if event_type:
            conditions.append("event_type = ?")
            params.append(event_type)
        if since_hours is not None:
            cutoff = (datetime.now() - timedelta(hours=since_hours)).strftime("%Y-%m-%d %H:%M:%S")
            conditions.append("timestamp >= ?")
            params.append(cutoff)

        where = ("WHERE " + " AND ".join(conditions)) if conditions else ""
        params += [limit, offset]

        with self._conn() as conn:
            rows = conn.execute(
                f"SELECT * FROM events {where} ORDER BY timestamp DESC LIMIT ? OFFSET ?",
                params,
            ).fetchall()
        return [dict(r) for r in rows]

    def count_events(
        self,
        module: Optional[str] = None,
        event_type: Optional[str] = None,
        since_hours: int = 24,
    ) -> int:
        """Count events for statistics."""
        conditions = []
        params: List[Any] = []
        cutoff = (datetime.now() - timedelta(hours=since_hours)).strftime("%Y-%m-%d %H:%M:%S")
        conditions.append("timestamp >= ?")
        params.append(cutoff)
        if module:
            conditions.append("module = ?")
            params.append(module)
        if event_type:
            conditions.append("event_type = ?")
            params.append(event_type)
        where = "WHERE " + " AND ".join(conditions)
        with self._conn() as conn:
            return conn.execute(
                f"SELECT COUNT(*) FROM events {where}", params
            ).fetchone()[0]

    # ── Alert management ──────────────────────────────────────────────────────

    def log_alert(
        self,
        severity: str,
        category: str,
        title: str,
        description: Optional[str] = None,
        event_id: Optional[int] = None,
    ) -> int:
        with self._conn() as conn:
            cur = conn.execute(
                """INSERT INTO alerts (severity, category, title, description, event_id)
                   VALUES (?, ?, ?, ?, ?)""",
                (severity, category, title, description, event_id),
            )
            return cur.lastrowid

    def get_alerts(
        self,
        resolved: Optional[bool] = False,
        limit: int = 50,
    ) -> List[Dict[str, Any]]:
        params: List[Any] = []
        where = ""
        if resolved is not None:
            where = "WHERE resolved = ?"
            params.append(1 if resolved else 0)
        params.append(limit)
        with self._conn() as conn:
            rows = conn.execute(
                f"SELECT * FROM alerts {where} ORDER BY timestamp DESC LIMIT ?",
                params,
            ).fetchall()
        return [dict(r) for r in rows]

    def resolve_alert(self, alert_id: int) -> bool:
        with self._conn() as conn:
            conn.execute(
                "UPDATE alerts SET resolved=1, resolved_at=datetime('now','localtime') WHERE id=?",
                (alert_id,),
            )
        return True

    def count_open_alerts(self) -> int:
        with self._conn() as conn:
            return conn.execute(
                "SELECT COUNT(*) FROM alerts WHERE resolved=0"
            ).fetchone()[0]

    # ── System health ─────────────────────────────────────────────────────────

    def log_health(
        self,
        cpu_pct: float,
        mem_pct: float,
        gpu_pct: float = 0.0,
        fps: float = 0.0,
        cam_online: bool = True,
    ) -> None:
        with self._conn() as conn:
            conn.execute(
                """INSERT INTO system_health (cpu_pct, mem_pct, gpu_pct, fps, cam_online)
                   VALUES (?, ?, ?, ?, ?)""",
                (cpu_pct, mem_pct, gpu_pct, fps, int(cam_online)),
            )

    def get_recent_health(self, limit: int = 60) -> List[Dict[str, Any]]:
        """Last `limit` health snapshots (for sparklines)."""
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM system_health ORDER BY timestamp DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [dict(r) for r in reversed(rows)]

    # ── Dashboard statistics ──────────────────────────────────────────────────

    def get_stats(self) -> Dict[str, Any]:
        """Summary counters for the dashboard Overview card."""
        return {
            "faces_authorized_24h":  self.count_events("face",    "authorized",  24),
            "faces_stranger_24h":    self.count_events("face",    "stranger",    24),
            "faces_spoof_24h":       self.count_events("face",    "spoof",       24),
            "vehicles_known_24h":    self.count_events("vehicle", "known",       24),
            "vehicles_unknown_24h":  self.count_events("vehicle", "unknown",     24),
            "open_alerts":           self.count_open_alerts(),
            "total_events_24h":      self.count_events(since_hours=24),
        }
