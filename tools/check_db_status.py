import sqlite3
import hashlib
from pathlib import Path

db = Path("logs/events.db")
conn = sqlite3.connect(str(db))
rows = conn.execute("SELECT username, password, role FROM accounts").fetchall()
candidates = ["admin-password-123", "staff-password-123", "viewer-password-123", "password123", "admin12345"]
for user, pw_str, role in rows:
    salt, val = pw_str.split(":")
    matched = None
    for cand in candidates:
        test_val = hashlib.scrypt(cand.encode(), salt=bytes.fromhex(salt), n=16384, r=8, p=1).hex()
        if test_val == val:
            matched = cand
            break
    print(f"User: {user} | Role: {role} | Password: {matched}")
