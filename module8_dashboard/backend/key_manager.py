"""
key_manager.py — Cryptographic Key Management & Biometric Protection
=====================================================================
Manages AES-256-GCM encryption keys for biometric embeddings.
Supports environment injection (INTELLIGATE_BIOMETRIC_KEY), protected key files,
and backward-compatible local key fallback.

Security Invariants:
1. Never silently overwrite or regenerate an encryption key if existing encrypted
   records are present in the database.
2. Never print or log raw key material.
3. Fail securely with clear diagnostics when a required key is missing or mismatched.
4. Cryptographically authenticate the key against existing records upon startup.
"""
from __future__ import annotations

import base64
import os
from pathlib import Path
from typing import Optional

import numpy as np
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from loguru import logger

ENV_KEY_VAR = "INTELLIGATE_BIOMETRIC_KEY"
ENV_KEY_FILE_VAR = "INTELLIGATE_BIOMETRIC_KEY_FILE"
KEY_SIZE_BYTES = 32  # 256 bits


def _parse_key_bytes(raw_val: str | bytes) -> bytes:
    """Safely decode key material from hex, base64, or raw bytes."""
    if isinstance(raw_val, bytes):
        if len(raw_val) == KEY_SIZE_BYTES:
            return raw_val
        raise ValueError(f"Raw key must be exactly {KEY_SIZE_BYTES} bytes, got {len(raw_val)}")

    val = raw_val.strip()
    # Try 64-character hex string
    if len(val) == 64:
        try:
            parsed = bytes.fromhex(val)
            if len(parsed) == KEY_SIZE_BYTES:
                return parsed
        except ValueError:
            pass

    # Try 44-character base64 string
    if len(val) == 44:
        try:
            parsed = base64.b64decode(val)
            if len(parsed) == KEY_SIZE_BYTES:
                return parsed
        except Exception:
            pass

    # Try raw 32-character string
    if len(val.encode("utf-8")) == KEY_SIZE_BYTES:
        return val.encode("utf-8")

    raise ValueError(
        f"Invalid biometric encryption key format: expected {KEY_SIZE_BYTES} bytes "
        f"(or 64-char hex / 44-char base64). Provided length: {len(val)}"
    )


def resolve_biometric_key(store, key_path: Optional[Path] = None) -> bytes:
    """
    Resolve the 256-bit AES-GCM biometric key with defensive integrity checks.

    Priority:
    1. Environment variable INTELLIGATE_BIOMETRIC_KEY (hex, base64, or raw)
    2. Environment variable INTELLIGATE_BIOMETRIC_KEY_FILE (file path)
    3. File path: key_path or store.db_path.parent / 'biometric.key'

    Invariants:
    - If the database contains existing encrypted records, a missing or invalid key
      MUST raise RuntimeError rather than generating a replacement key.
    """
    default_key_path = key_path or (store.db_path.parent / "biometric.key")
    key_bytes: Optional[bytes] = None
    key_source: str = "none"

    # 1. Check environment variable injection
    if os.environ.get(ENV_KEY_VAR):
        try:
            key_bytes = _parse_key_bytes(os.environ[ENV_KEY_VAR])
            key_source = f"environment variable ({ENV_KEY_VAR})"
        except Exception as exc:
            raise RuntimeError(f"Failed to load key from {ENV_KEY_VAR}: {exc}") from exc

    # 2. Check external key file environment variable
    elif os.environ.get(ENV_KEY_FILE_VAR):
        file_path = Path(os.environ[ENV_KEY_FILE_VAR])
        if not file_path.exists():
            raise RuntimeError(f"Configured key file not found at {file_path}")
        try:
            key_bytes = _parse_key_bytes(file_path.read_bytes())
            key_source = f"configured key file ({file_path})"
        except Exception as exc:
            raise RuntimeError(f"Failed to load key from {file_path}: {exc}") from exc

    # 3. Check local default file fallback
    elif default_key_path.exists():
        try:
            key_bytes = _parse_key_bytes(default_key_path.read_bytes())
            key_source = f"local file ({default_key_path.name})"
        except Exception as exc:
            raise RuntimeError(f"Failed to load key from {default_key_path}: {exc}") from exc

    # 4. Check if existing encrypted records exist in the database
    has_records = False
    with store._conn() as db:
        # Check if table exists
        row = db.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='cpu_faces'").fetchone()
        if row:
            count = db.execute("SELECT COUNT(*) FROM cpu_faces").fetchone()[0]
            has_records = count > 0

    # 5. Handle missing key
    if key_bytes is None:
        if has_records:
            msg = (
                "Critical Security Error: The database contains existing encrypted biometric records, "
                "but no valid encryption key was provided. Refusing to generate a new key to prevent "
                f"unrecoverable biometric data loss! Please set {ENV_KEY_VAR} or restore biometric.key."
            )
            logger.critical(f"[KeyManager] {msg}")
            raise RuntimeError(msg)

        # Fresh database / no records: generate a new key safely
        key_bytes = AESGCM.generate_key(bit_length=256)
        key_source = "freshly generated key"
        try:
            fd = os.open(str(default_key_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
            with os.fdopen(fd, "wb") as f:
                f.write(key_bytes)
            logger.info(f"[KeyManager] Initialized fresh biometric encryption key at {default_key_path.name}")
        except FileExistsError:
            # Another process wrote it first, re-read
            key_bytes = _parse_key_bytes(default_key_path.read_bytes())
        except Exception as exc:
            logger.warning(f"[KeyManager] Could not persist key to {default_key_path}: {exc}")

    # 6. Verify key matches existing encrypted database records
    if has_records:
        with store._conn() as db:
            test_row = db.execute("SELECT id, embedding FROM cpu_faces WHERE embedding IS NOT NULL LIMIT 1").fetchone()
            if test_row:
                blob = test_row["embedding"]
                identity = test_row["id"]
                try:
                    cipher = AESGCM(key_bytes)
                    cipher.decrypt(blob[:12], blob[12:], identity.encode("utf-8"))
                    logger.info("[KeyManager] Key successfully authenticated against existing encrypted records.")
                except Exception as exc:
                    msg = (
                        "Critical Security Error: The provided biometric encryption key failed authentication "
                        "against existing database records! The key does not match this database. "
                        "Startup aborted to protect biometric store."
                    )
                    logger.critical(f"[KeyManager] {msg}")
                    raise RuntimeError(msg) from exc

    logger.info(f"[KeyManager] Biometric key loaded from {key_source} (256-bit AES-GCM).")
    return key_bytes
