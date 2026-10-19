"""Persistent software access control. All gate commands are simulations."""
from __future__ import annotations

import json
import threading
import time
import uuid
from datetime import datetime, timezone


class SoftwareRuntime:
    def __init__(self, store, clock=time.time):
        self.store, self.clock = store, clock
        self.lock = threading.RLock()
        with store._conn() as db:
            db.executescript("""
                CREATE TABLE IF NOT EXISTS runtime_state (key TEXT PRIMARY KEY, value TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS gate_commands (id TEXT PRIMARY KEY, command TEXT, result TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS review_requests (
                    id TEXT PRIMARY KEY, event_id INTEGER, identity TEXT, status TEXT,
                    created REAL, expires REAL, actor TEXT);
                CREATE TABLE IF NOT EXISTS visitor_visits (
                    id INTEGER PRIMARY KEY, identity TEXT, source TEXT, session TEXT, seen REAL,
                    UNIQUE(identity, source, session));
                CREATE INDEX IF NOT EXISTS idx_review_requests_lookup ON review_requests(identity, status, expires);
                CREATE INDEX IF NOT EXISTS idx_visitor_visits_seen ON visitor_visits(seen);
                CREATE INDEX IF NOT EXISTS idx_visitor_visits_identity_seen ON visitor_visits(identity, seen);
            """)
        self._last_visitor_prune = 0.0
        # Motion never resumes automatically after a process restart.
        state = self.gate()
        if state['state'] in {'OPENING', 'CLOSING', 'OPEN'}:
            state.update(state='STOPPED', due=None, reason='Restart requires operator review')
            self.save('gate', state)

    def load(self, key, default):
        with self.store._conn() as db:
            row = db.execute('SELECT value FROM runtime_state WHERE key=?', (key,)).fetchone()
        return json.loads(row[0]) if row else default

    def save(self, key, value):
        with self.store._conn() as db:
            db.execute('INSERT OR REPLACE INTO runtime_state VALUES (?,?)', (key, json.dumps(value)))

    def gate(self):
        return self.load('gate', {'state': 'CLOSED', 'obstruction': False, 'lockdown': False,
                                 'due': None, 'simulated': True, 'reason': 'Virtual gate ready'})

    def emit(self, module, kind, identity=None, **meta):
        return self.store.log_event(module, kind, identity=identity, method='software',
                                    meta=json.dumps({'simulated': True, **meta}))

    def command(self, command, actor, command_id=None, expires=None):
        command_id = command_id or str(uuid.uuid4())
        with self.lock:
            with self.store._conn() as db:
                previous = db.execute('SELECT command,result FROM gate_commands WHERE id=?', (command_id,)).fetchone()
            if previous:
                if previous[0] != command:
                    raise ValueError('Command ID was already used for a different command')
                return json.loads(previous[1])
            gate = self.gate()
            if command != 'STOP' and expires is not None and expires < self.clock():
                raise ValueError('Command expired')
            if command in {'OPEN', 'CLOSE'} and (gate['obstruction'] or gate['lockdown']):
                raise ValueError('Motion blocked by obstruction or lockdown')
            if command == 'OPEN':
                gate.update(state='OPENING', due=self.clock() + 0.7)
            elif command == 'CLOSE':
                gate.update(state='CLOSING', due=self.clock() + 0.7)
            elif command == 'STOP':
                gate.update(state='STOPPED', due=None)
            elif command == 'LOCKDOWN':
                gate.update(state='LOCKDOWN', lockdown=True, due=None)
            elif command == 'RESET':
                if gate['obstruction']:
                    raise ValueError('Clear obstruction before resetting')
                gate.update(state='CLOSED', lockdown=False, due=None)
            else:
                raise ValueError('Unsupported command')
            gate.update(reason=f'{command} by {actor}', command_id=command_id)
            # State, command journal and event commit together.
            with self.store._conn() as db:
                db.execute('INSERT OR REPLACE INTO runtime_state VALUES (?,?)', ('gate', json.dumps(gate)))
                db.execute('INSERT INTO gate_commands VALUES (?,?,?)', (command_id, command, json.dumps(gate)))
                db.execute('INSERT INTO events(module,event_type,identity,method,meta) VALUES (?,?,?,?,?)',
                           ('gate', command.lower(), actor, 'virtual_gate', json.dumps(gate)))
            return gate

    def obstruction(self, present, source='operator'):
        with self.lock:
            gate = self.gate()
            sources = self.load('obstruction_sources', {})
            sources[source] = present
            self.save('obstruction_sources', sources)
            active = any(sources.values())
            changed = gate['obstruction'] != active
            gate['obstruction'] = active
            if active:
                gate.update(state='STOPPED', due=None, reason=f'Obstruction from {source}')
            self.save('gate', gate)
            if changed:
                eid = self.emit('safety', 'obstruction' if active else 'obstruction_cleared', source)
                if active:
                    self.store.log_alert('CRITICAL', 'obstruction', 'Virtual gate stopped: obstruction', event_id=eid)
            return gate

    def tick(self):
        with self.lock:
            gate = self.gate()
            if gate.get('due') and self.clock() >= gate['due']:
                if gate['state'] == 'OPENING':
                    gate.update(state='OPEN', due=self.clock() + 5)
                elif gate['state'] == 'OPEN' and not gate['obstruction'] and not gate['lockdown']:
                    gate.update(state='CLOSING', due=self.clock() + 0.7)
                elif gate['state'] == 'CLOSING':
                    gate.update(state='CLOSED', due=None)
                else:
                    gate['due'] = None
                self.save('gate', gate)
                self.emit('gate', 'state_changed', state=gate)
            with self.store._conn() as db:
                db.execute("UPDATE review_requests SET status='EXPIRED' WHERE status='PENDING' AND expires<?", (self.clock(),))
            return gate

    def observe(self, event, trusted: bool | None = None):
        """Observations are advisory; only explicit validated facts from trusted internal perception may grant access."""
        with self.lock:
            meta = dict(event.get('meta') or {})
            kind, module = event['event_type'], event['module']
            identity = event.get('identity')
            decision, reason = 'REVIEW_REQUIRED', 'Identity or liveness requires operator review'

            # Strict server-controlled trust boundary:
            # Automatic ALLOW is strictly prohibited unless the caller explicitly passes trusted=True
            # AND internal perception provenance is established with strict boolean True.
            # All external, unknown, missing (trusted=None), and untrusted contexts fail closed.
            is_trusted_perception = (
                trusted is True
                and meta.get('provenance') == 'perception_internal'
                and meta.get('trusted_perception') is True
            )

            if kind in {'spoof', 'spoof_detected'}:
                decision, reason = 'DENY', 'Liveness checks rejected the input'
            elif kind in {'authorized', 'known', 'access_granted'}:
                if is_trusted_perception:
                    if module == 'vehicle' and event.get('method') == 'exact_plate' and meta.get('ocr_confidence', 0) >= 0.85:
                        decision, reason = 'ALLOW', 'Registered exact plate with sufficient OCR confidence'
                elif meta.get('provenance') == 'operator_annotation':
                    decision, reason = 'REVIEW_REQUIRED', 'Operator-submitted event requires manual verification'
                else:
                    decision, reason = 'REVIEW_REQUIRED', 'Unverified event provenance requires review'
            if self.gate()['obstruction'] or self.gate()['lockdown']:
                decision, reason = 'DENY', 'Gate safety interlock is active'
            meta.update(decision=decision, reason=reason, simulated=True)
            stored_kind = 'recognition_candidate' if kind in {'authorized', 'known', 'access_granted'} and decision != 'ALLOW' else kind
            eid = self.store.log_event(module, stored_kind, identity=identity, confidence=event.get('confidence'),
                                      method=event.get('method'), camera_id=event.get('camera_id', 0), meta=json.dumps(meta))
            if decision == 'ALLOW':
                self.command('OPEN', 'access-policy', expires=self.clock() + 5)
            elif decision == 'REVIEW_REQUIRED':
                with self.store._conn() as db:
                    pending = db.execute("SELECT 1 FROM review_requests WHERE identity IS ? AND status='PENDING' AND expires>?", (identity, self.clock())).fetchone()
                    if not pending:
                        db.execute('INSERT INTO review_requests VALUES (?,?,?,?,?,?,?)',
                                   (str(uuid.uuid4()), eid, identity, 'PENDING', self.clock(), self.clock() + 120, None))
            elif kind in {'spoof', 'spoof_detected'}:
                alert_desc = 'Operator reported spoof attempt' if meta.get('provenance') == 'operator_annotation' else 'Liveness rejection'
                self.store.log_alert('HIGH', 'spoof_attempt', alert_desc, event_id=eid)
            if meta.get('stranger_id'):
                self.visit(meta['stranger_id'], str(event.get('camera_id', 0)), meta.get('session_id', 'default'), self.clock())
            return {'event_id': eid, 'event_type': stored_kind, 'decision': decision, 'reason': reason, 'gate': self.gate(), 'simulated': True}

    def reviews(self):
        self.tick()
        with self.store._conn() as db:
            return [dict(row) for row in db.execute('SELECT * FROM review_requests ORDER BY created DESC LIMIT 100')]

    def review(self, request_id, approve, actor):
        with self.lock:
            self.tick()
            with self.store._conn() as db:
                row = db.execute('SELECT * FROM review_requests WHERE id=?', (request_id,)).fetchone()
            if not row or row['status'] != 'PENDING':
                raise ValueError('Review is missing, expired or already resolved')
            if approve:
                self.command('OPEN', actor, command_id=f'review-{request_id}', expires=row['expires'])
            status = 'APPROVED' if approve else 'DENIED'
            with self.store._conn() as db:
                db.execute('UPDATE review_requests SET status=?,actor=? WHERE id=?', (status, actor, request_id))
            self.emit('access', status.lower(), row['identity'], actor=actor, request_id=request_id, event_id=row['event_id'])
            return {'status': status, 'gate': self.gate()}

    def visit(self, identity, source, session, seen):
        with self.lock:
            cutoff = seen - 86400
            with self.store._conn() as db:
                if seen - self._last_visitor_prune > 3600:
                    db.execute('DELETE FROM visitor_visits WHERE seen<?', (seen - 90 * 86400,))
                    self._last_visitor_prune = seen
                db.execute('INSERT OR IGNORE INTO visitor_visits(identity,source,session,seen) VALUES (?,?,?,?)',
                           (identity, source, session, seen))
                count = db.execute('SELECT COUNT(*) FROM visitor_visits WHERE identity=? AND seen>=? AND seen<=?',
                                   (identity, cutoff, seen)).fetchone()[0]
            marker = f'repeat:{identity}'
            last_alert = self.load(marker, None)
            if count >= 3 and (last_alert is None or seen - last_alert >= 86400):
                eid = self.emit('visitors', 'repeat_visitor', identity, visit_count=count)
                self.store.log_alert('HIGH', 'repeat_visitor', f'Repeat visitor: {count} distinct visits', event_id=eid)
                self.save(marker, seen)
            return count

    def visitors(self):
        with self.store._conn() as db:
            return [dict(row) for row in db.execute('SELECT identity,COUNT(*) AS visits,MIN(seen) AS first_seen,MAX(seen) AS last_seen FROM visitor_visits GROUP BY identity ORDER BY last_seen DESC LIMIT 100')]

    def report(self, hours=24):
        # Aggregate all matching rows; the evidence preview is deliberately bounded.
        with self.store._conn() as db:
            rows = db.execute("SELECT module,event_type,COUNT(*) AS count FROM events WHERE timestamp>=datetime('now', ?) GROUP BY module,event_type", (f'-{hours} hours',)).fetchall()
        counts = [dict(row) for row in rows]
        total = sum(row['count'] for row in counts)
        text = f'{total} recorded events in the last {hours} hours. '
        text += ' '.join(f"{row['count']} {row['module']} / {row['event_type']} events." for row in counts)
        return {'generated_at': datetime.now(timezone.utc).isoformat(), 'hours': hours, 'total': total,
                'counts': counts, 'summary': text, 'method': 'event-template (not a VLM)',
                'evidence': self.store.get_events(since_hours=hours, limit=100)}

    def search(self, query):
        query = query.strip().lower()
        words = set(query.split())
        aliases = {'faces': 'face', 'face': 'face', 'vehicles': 'vehicle', 'vehicle': 'vehicle',
                   'gate': 'gate', 'safety': 'safety', 'queue': 'queue', 'visitors': 'visitors'}
        module = next((aliases[word] for word in words if word in aliases), None)
        event_type = next((word for word in ['spoof_detected', 'obstruction', 'access_granted', 'repeat_visitor', 'congestion'] if word in words), None)
        if 'spoof' in words:
            event_type = 'spoof_detected'
        if not module and not event_type and query not in {'today', 'all', 'events', 'events today'}:
            raise ValueError('Supported filters: face, vehicle, gate, safety, queue, visitors, spoof, obstruction, today')
        return {'query': query, 'method': 'limited keyword parser', 'events': self.store.get_events(module=module, event_type=event_type, since_hours=24 if 'today' in words else None, limit=200)}
