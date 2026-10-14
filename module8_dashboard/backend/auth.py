"""Local account bootstrap, scrypt passwords, expiring HttpOnly cookie sessions."""
import hashlib
import hmac
import secrets
import time

from fastapi import HTTPException


class Auth:
    def __init__(self, store):
        self.store = store
        with store._conn() as db:
            db.executescript('''
                CREATE TABLE IF NOT EXISTS accounts (username TEXT PRIMARY KEY, password TEXT, role TEXT);
                CREATE TABLE IF NOT EXISTS sessions (token_hash TEXT PRIMARY KEY, username TEXT, expires REAL);
                CREATE TABLE IF NOT EXISTS login_failures (key TEXT PRIMARY KEY, count INTEGER, until REAL);
            ''')

    def ready(self):
        with self.store._conn() as db:
            return db.execute('SELECT COUNT(*) FROM accounts').fetchone()[0] > 0

    @staticmethod
    def password_hash(password, salt=None):
        salt = salt or secrets.token_hex(16)
        value = hashlib.scrypt(password.encode(), salt=bytes.fromhex(salt), n=16384, r=8, p=1).hex()
        return salt + ':' + value

    def create(self, username, password, role, bootstrap=False):
        if role not in {'admin', 'security_staff', 'viewer'}:
            raise HTTPException(422, 'Invalid role')
        if not username.strip() or len(username) > 64 or not 10 <= len(password) <= 128:
            raise HTTPException(422, 'Username required; password must contain 10-128 characters')
        with self.store._conn() as db:
            db.execute('BEGIN IMMEDIATE')
            if bootstrap and db.execute('SELECT COUNT(*) FROM accounts').fetchone()[0]:
                raise HTTPException(409, 'Setup already completed')
            if db.execute('SELECT 1 FROM accounts WHERE username=?', (username,)).fetchone():
                raise HTTPException(409, 'Username already exists')
            db.execute('INSERT INTO accounts VALUES (?,?,?)', (username, self.password_hash(password), role))
        return {'username': username, 'role': role}

    def login(self, username, password, client):
        key = client + ':' + username
        now = time.time()
        with self.store._conn() as db:
            failure = db.execute('SELECT count,until FROM login_failures WHERE key=?', (key,)).fetchone()
            if failure and failure['count'] >= 5 and failure['until'] > now:
                raise HTTPException(429, 'Too many attempts; try again in 5 minutes')
            account = db.execute('SELECT * FROM accounts WHERE username=?', (username,)).fetchone()
        expected = account['password'] if account else self.password_hash('invalid-account-password')
        actual = self.password_hash(password, expected.split(':')[0])
        if not account or not hmac.compare_digest(expected, actual):
            count = failure['count'] + 1 if failure and failure['until'] > now else 1
            with self.store._conn() as db:
                db.execute('INSERT OR REPLACE INTO login_failures VALUES (?,?,?)', (key, count, now + 300))
            raise HTTPException(401, 'Invalid credentials')
        token = secrets.token_urlsafe(32)
        with self.store._conn() as db:
            db.execute('DELETE FROM login_failures WHERE key=?', (key,))
            db.execute('DELETE FROM sessions WHERE expires<?', (now,))
            db.execute('INSERT INTO sessions VALUES (?,?,?)', (hashlib.sha256(token.encode()).hexdigest(), username, now + 8 * 3600))
        return token

    def user(self, token):
        if not token:
            return None
        with self.store._conn() as db:
            row = db.execute('SELECT a.username,a.role FROM sessions s JOIN accounts a ON a.username=s.username WHERE token_hash=? AND expires>?',
                             (hashlib.sha256(token.encode()).hexdigest(), time.time())).fetchone()
        return dict(row) if row else None

    def logout(self, token):
        with self.store._conn() as db:
            db.execute('DELETE FROM sessions WHERE token_hash=?', (hashlib.sha256((token or '').encode()).hexdigest(),))
