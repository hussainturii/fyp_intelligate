"""CPU replay pipelines: real detections, baseline appearance and explicit limits."""
from __future__ import annotations

import hashlib
import json
import threading
import time
import uuid
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
INFERENCE_LOCK = threading.RLock()


def enhance(frame):
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    lab[:, :, 0] = cv2.createCLAHE(clipLimit=2, tileGridSize=(8, 8)).apply(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def appearance(frame):
    """768-D colour histogram baseline, not a trained vehicle Re-ID network."""
    vector = np.concatenate([cv2.calcHist([frame], [i], None, [256], [0, 256]).ravel() for i in range(3)])
    return (vector / max(np.linalg.norm(vector), 1e-12)).astype(np.float32)


def iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x2-x1) * max(0, y2-y1)
    return inter / max(1e-9, (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter)


def crop_fingerprint(crop):
    """Cheap continuity cue; an abrupt face change requires fresh identity matching."""
    if not crop.size:
        return None
    grey=cv2.resize(cv2.cvtColor(crop,cv2.COLOR_BGR2GRAY),(32,32)).astype(np.float32)
    if float(grey.std()) < 5:
        return None
    coefficients=cv2.dct(grey)[:8,:8].ravel()[1:]
    return coefficients > np.median(coefficients)


class Tracker:
    """Short-lived, one-to-one IoU association for ROI counts; not ByteTrack."""
    def __init__(self):
        self.tracks, self.counter = {}, 0

    def update(self, detections, now):
        self.tracks = {k: v for k, v in self.tracks.items() if now - v['seen'] < 2}
        # Assign globally rather than letting the first detection steal a track.
        candidates = []
        ambiguous = set()
        for index, (box, label, _) in enumerate(detections):
            scores = sorted([(iou(box, state['box']), key) for key, state in self.tracks.items() if state['label'] == label], reverse=True)
            if len(scores) > 1 and scores[1][0] >= .25 and scores[0][0] - scores[1][0] < .12:
                ambiguous.add(index)
            candidates.extend((score, index, key) for score, key in scores if score >= .25)
        assignment, used = {}, set()
        for _, index, key in sorted(candidates, reverse=True):
            if index not in assignment and key not in used and index not in ambiguous:
                assignment[index] = key
                used.add(key)
        result = []
        for index, (box, label, confidence) in enumerate(detections):
            tid = assignment.get(index)
            if tid is None:
                self.counter += 1
                tid = self.counter
                self.tracks[tid] = {'first': now}
            self.tracks[tid].update(box=box, label=label, seen=now, confidence=confidence)
            result.append({'track_id': tid, 'ambiguous': index in ambiguous, **self.tracks[tid]})
        return result


class PresenceLedger:
    """One event per identity while continuously visible, including short occlusions."""
    def __init__(self, grace_seconds=2):
        self.grace = grace_seconds
        self.present = {}

    def touch(self, identity, now):
        previous = self.present.get(identity)
        fresh = previous is None or now - previous['seen'] >= self.grace
        if fresh:
            self.present[identity] = {'seen': now, 'visit_id': str(uuid.uuid4())}
        else:
            previous['seen'] = now
        return fresh, self.present[identity]['visit_id']

    def prune(self, now):
        self.present = {key: state for key, state in self.present.items() if now - state['seen'] < self.grace}


class FaceEngine:
    def __init__(self, store):
        from .model_integrity import verify_model_file
        from .key_manager import resolve_biometric_key
        from cryptography.hazmat.primitives.ciphers.aead import AESGCM

        model_dir = ROOT / 'models'
        paths = [model_dir / 'face_detection_yunet_2023mar.onnx', model_dir / 'face_recognition_sface_2021dec.onnx']
        verify_model_file('models/face_detection_yunet_2023mar.onnx')
        verify_model_file('models/face_recognition_sface_2021dec.onnx')
        self.detector = cv2.FaceDetectorYN.create(str(paths[0]), '', (640, 360), 0.8, 0.3, 5000)
        self.recognizer = cv2.FaceRecognizerSF.create(str(paths[1]), '')
        self.store = store
        self.gallery_revision = 0
        self._gallery_cache_rev = -1
        self._gallery_cache = []
        self._stranger_cache = []
        self._last_stranger_prune = 0.0
        self.lock = INFERENCE_LOCK

        key_bytes = resolve_biometric_key(store)
        self.cipher = AESGCM(key_bytes)
        with store._conn() as db:
            db.execute('CREATE TABLE IF NOT EXISTS cpu_faces (id TEXT PRIMARY KEY, name TEXT UNIQUE, role TEXT, embedding BLOB, created REAL)')
            db.execute('CREATE TABLE IF NOT EXISTS cpu_strangers (id TEXT PRIMARY KEY, embedding BLOB, seen REAL)')
            db.execute('CREATE INDEX IF NOT EXISTS idx_cpu_strangers_seen ON cpu_strangers(seen)')
            cutoff = time.time() - 90 * 86400
            db.execute('DELETE FROM cpu_strangers WHERE seen<?', (cutoff,))
            rows = db.execute('SELECT id, embedding, seen FROM cpu_strangers ORDER BY seen DESC LIMIT 1000').fetchall()
            for r in rows:
                try:
                    dec = self.open(r['embedding'], r['id'])
                    self._stranger_cache.append((r['id'], dec, r['seen']))
                except Exception:
                    pass
        self._last_stranger_prune = time.time()

    def detect(self, frame):
        self.detector.setInputSize((frame.shape[1], frame.shape[0]))
        _, rows = self.detector.detect(frame)
        return [] if rows is None else list(rows)

    def feature(self, frame, face):
        aligned = self.recognizer.alignCrop(frame, face)
        embedding = self.recognizer.feature(aligned).ravel().astype(np.float32)
        return embedding / max(np.linalg.norm(embedding), 1e-12)

    def seal(self, embedding, identity):
        import os
        nonce = os.urandom(12)
        return nonce + self.cipher.encrypt(nonce, embedding.tobytes(), identity.encode())

    def open(self, blob, identity):
        return np.frombuffer(self.cipher.decrypt(blob[:12], blob[12:], identity.encode()), dtype=np.float32)

    def register(self, name, role, frame):
        with self.lock:
            faces = self.detect(frame)
            if len(faces) != 1:
                raise ValueError('Upload an image containing exactly one clear face')
            identity = str(uuid.uuid4())
            embedding = self.feature(frame, faces[0])
            with self.store._conn() as db:
                if db.execute('SELECT 1 FROM cpu_faces WHERE name=?', (name,)).fetchone():
                    raise ValueError('Name already registered; remove the previous enrolment first')
                db.execute('INSERT INTO cpu_faces VALUES (?,?,?,?,?)', (identity, name, role, self.seal(embedding, identity), time.time()))
            self.gallery_revision += 1
            return identity

    def list(self):
        with self.store._conn() as db:
            rows = db.execute('SELECT id,name,role,created FROM cpu_faces').fetchall()
        return {r['name']: {'user_id': r['id'], 'role': r['role'], 'registered_at': r['created'] * 1000,
                            'n_embeddings': 1, 'model': 'SFace CPU'} for r in rows}

    def delete(self, name):
        with self.store._conn() as db:
            removed = db.execute('DELETE FROM cpu_faces WHERE name=?', (name,)).rowcount > 0
        if removed:
            self.gallery_revision += 1
        return removed

    def match(self, embedding):
        if self._gallery_cache_rev != self.gallery_revision:
            with self.store._conn() as db:
                rows = db.execute('SELECT id,name,embedding FROM cpu_faces').fetchall()
            cached = []
            for r in rows:
                try:
                    dec = self.open(r['embedding'], r['id'])
                    cached.append((dec, r['name']))
                except Exception:
                    pass
            self._gallery_cache = cached
            self._gallery_cache_rev = self.gallery_revision

        if not self._gallery_cache:
            return (0.0, None)
        matches = [(float(np.dot(embedding, emb)), name) for emb, name in self._gallery_cache]
        return max(matches, default=(0.0, None))

    def stranger(self, embedding):
        now = time.time()
        if now - self._last_stranger_prune > 3600:
            with self.store._conn() as db:
                db.execute('DELETE FROM cpu_strangers WHERE seen<?', (now - 90 * 86400,))
            self._stranger_cache = [item for item in self._stranger_cache if item[2] >= now - 90 * 86400]
            self._last_stranger_prune = now

        best_score = 0.0
        best_id = None
        best_idx = -1
        for idx, (s_id, s_emb, _) in enumerate(self._stranger_cache):
            score = float(np.dot(embedding, s_emb))
            if score > best_score:
                best_score = score
                best_id = s_id
                best_idx = idx

        if best_score < 0.45 or best_id is None:
            identity = str(uuid.uuid4())
            sealed = self.seal(embedding, identity)
            with self.store._conn() as db:
                db.execute('INSERT INTO cpu_strangers VALUES (?,?,?)', (identity, sealed, now))
            self._stranger_cache.append((identity, embedding, now))
            return identity
        else:
            self._stranger_cache[best_idx] = (best_id, self._stranger_cache[best_idx][1], now)
            with self.store._conn() as db:
                db.execute('UPDATE cpu_strangers SET seen=? WHERE id=?', (now, best_id))
            return best_id


class ReplayPipeline:
    def __init__(self, mode, runtime, callback, face_engine=None, vehicle_db=None, config=None, clock=None, device=None):
        self.mode, self.runtime, self.callback = mode, runtime, callback
        self.faces, self.vehicle_db = face_engine, vehicle_db
        self.config = config or {}
        self.clock = clock or time.monotonic
        self.tracker = Tracker()
        self.session_id = str(uuid.uuid4())
        self.last_emit, self.alerted = {}, set()
        self.last_count = None
        self.last_profile = None
        self.model, self.ocr, self.plate_detector = None, None, None
        self.source_id = 0
        self.replayed = True
        self.active = True
        self.face_states, self.vehicle_states = {}, {}
        self.presence = PresenceLedger()
        self.overlay_lock = threading.Lock()
        self.overlays = []
        self.scene_overlay = None
        # Device selection: explicit override or automatic CUDA detection with clean CPU fallback
        if device is not None:
            self.device = device
        else:
            try:
                import torch
                self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
            except Exception:
                self.device = 'cpu'

        self.pad_detector = None
        self.metrics = {'frames': 0, 'latency_ms': 0, 'mode': mode, 'device': self.device, 'last_error': None,
                        'recognition_runs': 0, 'recognition_cache_hits': 0, 'visible_tracks': 0,
                        'faces_detected':0,'recognized_faces':0,'unknown_faces':0,'pending_faces':0,
                        'pad_bona_fide':0, 'pad_attack_suspected':0, 'pad_inconclusive':0, 'pad_not_evaluated':0,
                        'status':'starting','face_results':[], 'stage_latencies': {}}
        if mode == 'face':
            try:
                from module1_face_recognition.anti_spoofing.deep_pad import MiniFASNetV2Detector
                self.pad_detector = MiniFASNetV2Detector(device=self.device)
            except Exception as exc:
                self.metrics['last_error'] = f"Failed to initialize PAD detector: {exc}"
                self.pad_detector = None
        if mode in {'vehicle', 'scene', 'thermal'}:
            from .model_integrity import verify_model_file
            verify_model_file('yolov8n.pt')
            from ultralytics import YOLO
            path = ROOT / 'yolov8n.pt'
            self.model = YOLO(str(path))
        self.vehicle_reid = None
        if mode == 'vehicle':
            from module2_vehicle.plate_detector import PlateDetector
            from module2_vehicle.plate_ocr import PlateOCR
            use_gpu = self.device.startswith('cuda')
            self.plate_detector = PlateDetector(device=self.device, allow_download=False)
            self.ocr = PlateOCR(languages=['en'], gpu=use_gpu, n_reads=1, min_confidence=0.55)
            try:
                from module2_vehicle.vehicle_reid import VehicleReID
                self.vehicle_reid = VehicleReID()
            except Exception as exc:
                self.vehicle_reid = None

    def send(self, kind, module, identity=None, confidence=None, method=None, meta=None, key=None, cooldown=10):
        if not self.active:
            return
        key = key or (kind, identity)
        now = time.time()
        if cooldown > 0 and now - self.last_emit.get(key, 0) < cooldown:
            return
        if cooldown > 0:
            self.last_emit = {key: value for key,value in self.last_emit.items() if now-value < 300}
            self.last_emit[key] = now
        self.callback({'module': module, 'event_type': kind, 'identity': identity, 'confidence': confidence,
                       'method': method, 'camera_id': self.source_id,
                       'meta': {'session_id': self.session_id, 'replayed': self.replayed, **(meta or {})}})

    def publish_overlays(self, objects):
        with self.overlay_lock:
            self.overlays = objects

    def overlay_frame(self, frame):
        """Paint cached labels without inference; capture thread stays responsive."""
        with self.overlay_lock:
            objects = list(self.overlays)
        now = self.clock()
        for obj in objects:
            if now - obj['seen'] > 1:
                continue
            x1, y1, x2, y2 = map(int, obj['box'])
            color = obj['color']
            cv2.rectangle(frame, (x1,y1), (x2,y2), color, 2)
            label = obj['text']
            (width, height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, .55, 1)
            y = max(height + 4, y1)
            cv2.rectangle(frame, (max(0,x1),y-height-4), (max(0,x1)+width+6,y), color, -1)
            cv2.putText(frame,label,(max(0,x1)+3,y-3),cv2.FONT_HERSHEY_SIMPLEX,.55,(255,255,255),1)
        if self.scene_overlay is not None:
            height,width=frame.shape[:2]
            for key,color in [('queue_roi',(255,150,0)),('gate_roi',(0,0,255))]:
                region=self.config.get(key,[0,0,1,1] if key=='queue_roi' else [.4,.65,.6,1])
                cv2.rectangle(frame,(int(region[0]*width),int(region[1]*height)),(int(region[2]*width),int(region[3]*height)),color,1)
            cv2.putText(frame,f"ROI count {self.scene_overlay['count']} | virtual obstruction {self.scene_overlay['obstruction']}",(10,50),0,.5,(255,255,255),1)
        return frame

    def invalidate_vehicle_gallery(self):
        self.vehicle_states.clear()

    def _faces(self, frame, start_time=None):
        t0 = start_time or time.perf_counter()
        now = self.clock()
        t_det_0 = time.perf_counter()
        detections = self.faces.detect(frame) if self.faces is not None else []
        t_det_1 = time.perf_counter()
        self.metrics.update(faces_detected=len(detections), recognized_faces=0, unknown_faces=0, pending_faces=0,
                            pad_bona_fide=0, pad_attack_suspected=0, pad_inconclusive=0, pad_not_evaluated=0,
                            face_results=[])
        boxes = [((float(f[0]), float(f[1]), float(f[0] + f[2]), float(f[1] + f[3])), 'face', float(f[-1])) for f in detections]
        t_track_0 = time.perf_counter()
        tracks = self.tracker.update(boxes, now)
        t_track_1 = time.perf_counter()
        alive = set(self.tracker.tracks)
        self.face_states = {key: state for key, state in self.face_states.items() if key in alive}
        self.presence.prune(now)
        objects = []
        total_pad_ms = 0.0
        total_rec_ms = 0.0

        for face, track in zip(detections, tracks):
            tid = track['track_id']
            state = self.face_states.get(tid)
            x, y, w, h = map(int, face[:4])
            crop = frame[max(0, y):min(frame.shape[0], y + h), max(0, x):min(frame.shape[1], x + w)]
            fingerprint = crop_fingerprint(crop)
            attempts = state.get('attempts', 1) if state else 0
            if state and fingerprint is not None and state.get('fingerprint') is not None and float(np.mean(fingerprint != state['fingerprint'])) > .38:
                state = None
            if state and not state['known'] and attempts < 3 and face[2] >= state['crop_width'] * 1.5:
                state = None  # An unknown face is materially clearer/closer; try again.
            revision = getattr(self.faces, 'gallery_revision', 0) if self.faces is not None else 0
            if state is not None and state['revision'] != revision:
                state = None

            if state is None:
                # Wait for a usable crop, rather than caching a tiny/blurred face.
                quality = crop.size > 0 and min(w, h) >= 35
                if not quality:
                    self.metrics['pending_faces'] += 1
                    self.metrics['pad_not_evaluated'] += 1
                    objects.append({'box': track['box'], 'text': f'Face #{tid}: move closer', 'color': (0, 170, 255), 'seen': now})
                    continue

                # Presentation Attack Detection (MiniFASNetV2)
                t_pad_0 = time.perf_counter()
                if self.pad_detector is not None:
                    pad_result = self.pad_detector.evaluate_face(frame, (x, y, w, h))
                else:
                    pad_result = {
                        'pad_status': 'NOT_EVALUATED',
                        'pad_model': 'none',
                        'pad_score': None,
                        'pad_threshold': {'bona_fide': 0.80, 'attack': 0.40},
                        'pad_inference_ms': 0.0,
                        'pad_quality_status': 'no_detector_loaded',
                        'device': self.device
                    }
                pad_inf_time = (time.perf_counter() - t_pad_0) * 1000.0
                total_pad_ms += pad_result.get('pad_inference_ms', pad_inf_time)

                # SFace Feature Extraction & Matching
                t_rec_0 = time.perf_counter()
                embedding = self.faces.feature(frame, face) if self.faces is not None else np.zeros(128, dtype=np.float32)
                score, name = self.faces.match(embedding) if self.faces is not None else (0.0, None)
                total_rec_ms += (time.perf_counter() - t_rec_0) * 1000.0

                known = score >= .45 and name is not None
                identity = name if known else (self.faces.stranger(embedding) if self.faces is not None else str(uuid.uuid4()))
                state = {
                    'name': name if known else 'Unknown',
                    'identity': identity,
                    'known': known,
                    'score': score,
                    'revision': revision,
                    'crop_width': float(face[2]),
                    'attempts': attempts + 1,
                    'pad_status': pad_result['pad_status'],
                    'pad_score': pad_result['pad_score'],
                    'pad_model': pad_result.get('pad_model', 'MiniFASNetV2-80x80-2.7'),
                    'pad_quality_status': pad_result.get('pad_quality_status', 'optimal'),
                    'pad_inference_ms': pad_result.get('pad_inference_ms', round(pad_inf_time, 2)),
                    'pad_threshold': pad_result.get('pad_threshold', {'bona_fide': 0.80, 'attack': 0.40})
                }
                self.face_states[tid] = state
                self.metrics['recognition_runs'] += 1
            else:
                self.metrics['recognition_cache_hits'] += 1

            # Update metrics per PAD outcome
            pad_st = state.get('pad_status', 'NOT_EVALUATED')
            if pad_st == 'BONA_FIDE_CANDIDATE':
                self.metrics['pad_bona_fide'] += 1
            elif pad_st == 'ATTACK_SUSPECTED':
                self.metrics['pad_attack_suspected'] += 1
            elif pad_st == 'INCONCLUSIVE':
                self.metrics['pad_inconclusive'] += 1
            else:
                self.metrics['pad_not_evaluated'] += 1

            state['fingerprint'] = fingerprint
            self.metrics['recognized_faces' if state['known'] else 'unknown_faces'] += 1
            self.metrics['face_results'].append({
                'track_id': tid,
                'label': state['name'],
                'known': state['known'],
                'similarity': round(float(state['score']), 4),
                'threshold': .45,
                'box': [round(v, 1) for v in track['box']],
                'pad_status': state['pad_status'],
                'pad_score': state['pad_score'],
                'pad_model': state['pad_model'],
                'pad_threshold': state['pad_threshold'],
                'pad_inference_ms': state['pad_inference_ms'],
                'pad_quality_status': state['pad_quality_status']
            })

            fresh, visit_id = self.presence.touch(('face', state['identity']), now)
            if fresh:
                if state['pad_status'] == 'ATTACK_SUSPECTED':
                    self.send('spoof_detected', 'face', state['identity'], state['pad_score'], state['pad_model'],
                              {'validated_liveness': False, 'stranger_id': None if state['known'] else state['identity'],
                               'track_id': tid, 'session_id': f'{self.session_id}:{visit_id}',
                               'liveness_status': 'ATTACK_SUSPECTED: rejected by deep PAD (MiniFASNetV2)',
                               'pad_status': state['pad_status'], 'pad_score': state['pad_score'],
                               'pad_model': state['pad_model'], 'pad_quality_status': state['pad_quality_status'],
                               'similarity': round(float(state['score']), 4), 'threshold': .45,
                               'provenance': 'perception_internal', 'trusted_perception': True},
                              key=('face_visit', visit_id), cooldown=0)
                else:
                    self.send('access_granted' if state['known'] else 'stranger_alert', 'face', state['identity'], round(float(state['score']), 4), 'SFace',
                              {'validated_liveness': False, 'stranger_id': None if state['known'] else state['identity'],
                               'track_id': tid, 'session_id': f'{self.session_id}:{visit_id}',
                               'liveness_status': f"unvalidated ({state['pad_status'].lower()}): manual approval required",
                               'pad_status': state['pad_status'], 'pad_score': state['pad_score'],
                               'pad_model': state['pad_model'], 'pad_quality_status': state['pad_quality_status'],
                               'similarity': round(float(state['score']), 4), 'threshold': .45,
                               'provenance': 'perception_internal', 'trusted_perception': True},
                              key=('face_visit', visit_id), cooldown=0)

            # Box overlay styling based on PAD outcome and identity
            if state['pad_status'] == 'ATTACK_SUSPECTED':
                box_color = (0, 0, 230)
                box_text = f"SPOOF #{tid} ({state['name']})"
            elif state['known']:
                box_color = (0, 170, 0)
                box_text = f"{state['name']}  #{tid}"
            else:
                box_color = (0, 0, 230)
                box_text = f"{state['name']}  #{tid}"

            objects.append({'box': track['box'], 'text': box_text, 'color': box_color, 'seen': now})

        self.metrics['visible_tracks'] = len(tracks)
        self.publish_overlays(objects)
        t_faces_end = time.perf_counter()
        self.metrics['stage_latencies'] = {
            'detection_ms': round((t_det_1 - t_det_0) * 1000, 2),
            'pad_ms': round(total_pad_ms, 2),
            'recognition_ms': round(total_rec_ms, 2),
            'tracking_ms': round((t_track_1 - t_track_0) * 1000, 2),
            'total_cycle_ms': round((t_faces_end - t0) * 1000, 2)
        }
        return {'annotated_frame': self.overlay_frame(frame), 'faces_detected': len(detections), 'tracks': objects}

    def process_frame(self, frame):
        start = time.perf_counter()
        with INFERENCE_LOCK:
            if not self.active:
                return {'annotated_frame':frame}
            try:
                result = self._process(frame, start_time=start)
                self.metrics['last_error'] = None
                self.metrics['status']='running'
                return result
            except Exception as exc:
                self.metrics['last_error'] = str(exc)
                self.metrics['status']='error'
                raise
            finally:
                self.metrics['latency_ms'] = round((time.perf_counter() - start) * 1000, 1)
                self.metrics['frames'] += 1

    def _process(self, frame, start_time=None):
        t0 = start_time or time.perf_counter()
        frame = frame.copy()
        luminance = float(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).mean())
        profile = 'infrared_replay' if self.mode == 'thermal' else ('low_light' if luminance < 35 else 'day' if luminance > 55 else self.last_profile or 'day')
        if profile != self.last_profile:
            self.runtime.emit('night', 'profile_changed', profile, luminance=luminance, source_id=self.source_id)
            self.last_profile = profile
        if self.mode == 'low_light':
            adjusted = enhance(frame)
            cv2.putText(adjusted, 'CLAHE enhancement - RGB, not thermal', (10, 50), 0, 0.5, (0, 255, 255), 1)
            return {'annotated_frame': adjusted, 'luminance': luminance, 'profile': 'low_light'}
        if self.mode == 'face':
            return self._faces(frame, start_time=t0)
        if self.model is None:
            return {'annotated_frame': frame}
        if self.mode == 'scene' and profile == 'low_light':
            frame = enhance(frame)
        t_det_0 = time.perf_counter()
        prediction = self.model.predict(frame, imgsz=416, conf=0.4, classes=[0, 2, 3, 5, 7], device=self.device, verbose=False)[0]
        t_det_1 = time.perf_counter()
        detections = [(tuple(box.xyxy[0].cpu().tolist()), self.model.names[int(box.cls[0])], float(box.conf[0])) for box in prediction.boxes]
        t_track_0 = time.perf_counter()
        tracks = self.tracker.update(detections, self.clock())
        t_track_1 = time.perf_counter()
        self.metrics['visible_tracks'] = len(tracks)
        self.presence.prune(self.clock())
        self.vehicle_states = {key: state for key,state in self.vehicle_states.items() if key in self.tracker.tracks}
        self.alerted.intersection_update(self.tracker.tracks)
        height, width = frame.shape[:2]
        roi = self.config.get('queue_roi', [0, 0, 1, 1])
        gate_roi = self.config.get('gate_roi', [0.4, 0.65, 0.6, 1])
        def in_roi(box, region):
            cx, cy = (box[0]+box[2])/2/width, (box[1]+box[3])/2/height
            return region[0] <= cx <= region[2] and region[1] <= cy <= region[3]
        count = sum(in_roi(t['box'], roi) for t in tracks)
        obstruction = any(iou(t['box'], (gate_roi[0]*width, gate_roi[1]*height, gate_roi[2]*width, gate_roi[3]*height)) > 0 for t in tracks)
        self.scene_overlay = {'count':count,'obstruction':obstruction}
        if self.active and self.mode in {'scene', 'thermal'}:
            self.runtime.obstruction(obstruction, f'replay-{self.source_id}')
            state = {'count': count, 'obstruction': obstruction, 'source_id': self.source_id, 'luminance': luminance,
                     'profile': profile,
                     'mode': self.mode, 'simulated': True, 'tracker': 'IoU baseline',
                     'modality_warning': 'RGB-trained detector, infrared performance unvalidated' if self.mode == 'thermal' else None}
            self.runtime.save('scene', state)
            if count != self.last_count:
                self.runtime.emit('queue', 'count_changed', count=count, source_id=self.source_id)
                self.last_count = count
            if count >= self.config.get('queue_threshold', 5):
                self.send('congestion', 'queue', meta={'count': count}, cooldown=30)
        unannotated = frame.copy()
        objects = []
        for track in tracks:
            x1, y1, x2, y2 = map(int, track['box'])
            text, color = f"{track['label']} #{track['track_id']}", (0,170,255)
            if self.clock() - track['first'] >= self.config.get('loiter_seconds', 30) and track['track_id'] not in self.alerted:
                self.alerted.add(track['track_id'])
                self.send('loitering', 'visitors', str(track['track_id']), meta={'rule': 'continuous track dwell', 'seconds': self.config.get('loiter_seconds', 30)})
            if self.mode == 'vehicle' and track['label'] != 'person':
                crop = unannotated[max(0,y1):min(height,y2), max(0,x1):min(width,x2)]
                tid = track['track_id']
                if crop.size:
                    state = self.vehicle_states.get(tid)
                    attempts = state.get('attempts',1) if state else 0
                    if state and state['kind']=='stranger_alert' and attempts<3 and crop.shape[1]>=state['crop_width']*1.5:
                        state = None
                    if state is None:
                        state = self._vehicle(crop, tid)
                        state.update(crop_width=crop.shape[1],attempts=attempts+1)
                        self.vehicle_states[tid] = state
                        self.metrics['recognition_runs'] += 1
                    else:
                        self.metrics['recognition_cache_hits'] += 1
                    fresh, visit_id = self.presence.touch(('vehicle',state['identity']),self.clock())
                    if fresh:
                        self.send(state['kind'],'vehicle',state['identity'],state['confidence'],state['method'],
                                  {**state['meta'],'session_id':f'{self.session_id}:{visit_id}'},
                                  key=('vehicle_visit',visit_id),cooldown=0)
                    text, color = state['text'], state['color']
            objects.append({'box':track['box'],'text':text,'color':color,'seen':self.clock()})
        self.publish_overlays(objects)
        t_proc_end = time.perf_counter()
        self.metrics['stage_latencies'] = {
            'preprocessing_ms': round((t_det_0 - t0) * 1000, 2),
            'detector_inference_ms': round((t_det_1 - t_det_0) * 1000, 2),
            'tracking_ms': round((t_track_1 - t_track_0) * 1000, 2),
            'total_cycle_ms': round((t_proc_end - t0) * 1000, 2)
        }
        frame = self.overlay_frame(frame)
        for region, color in [(roi,(255,150,0)), (gate_roi,(0,0,255))]:
            cv2.rectangle(frame, (int(region[0]*width),int(region[1]*height)), (int(region[2]*width),int(region[3]*height)), color, 1)
        cv2.putText(frame, f'{"REPLAY" if self.replayed else "LIVE"} | ROI count {count} | virtual obstruction {obstruction}', (10,50), 0, 0.5, (255,255,255), 1)
        return {'annotated_frame': frame, 'count': count, 'obstruction': obstruction}

    def _vehicle(self, crop, track_id):
        plate, confidence = '', 0
        plates = self.plate_detector.detect(crop) if self.plate_detector else []
        if plates:
            result = self.ocr.read(plates[0].crop)
            plate, confidence = ''.join(c for c in result.clean_text.upper() if c.isalnum()), result.confidence
        match = self.vehicle_db.lookup_by_plate(plate) if (plate and self.vehicle_db) else None
        if match and match.get('method') != 'exact_plate':
            match = None

        method = 'exact_plate' if match else 'appearance_histogram_baseline'
        appearance_model = 'colour histogram baseline'
        embedding_dim = 768

        # Advisory appearance Re-ID fallback when OCR confidence is below authorization threshold (< 0.85)
        if not match and confidence < 0.85 and self.vehicle_db:
            # 1. Attempt deep metric vehicle Re-ID (OSNet-AIN 512-D)
            if self.vehicle_reid is not None:
                try:
                    deep_emb = self.vehicle_reid.extract_embedding(crop)
                    candidates = self.vehicle_db.search_by_embedding(deep_emb, k=1)
                    reid_thresh = getattr(self.vehicle_reid, 'threshold', 0.70)
                    if candidates and candidates[0].get('similarity', 0) >= reid_thresh:
                        match = candidates[0]
                        method = 'deep_reid'
                        appearance_model = 'osnet_ain_x1_0_vehicle_reid'
                        embedding_dim = 512
                except Exception:
                    match = None

            # 2. Fall back to color histogram baseline (768-D) if deep Re-ID yielded no match
            if not match:
                hist_emb = appearance(crop)
                candidates = self.vehicle_db.search_by_embedding(hist_emb, k=1)
                if candidates and candidates[0].get('similarity', 0) >= 0.92:
                    match = candidates[0]
                    method = 'appearance_histogram_baseline'
                    appearance_model = 'colour histogram baseline'
                    embedding_dim = 768

        identity = (match.get('plate') or match.get('owner_name')) if match else f'vehicle-{self.session_id[:8]}-{track_id}'
        name = f"{match['owner_name']} / {match.get('plate','')}" if match else f"Unknown vehicle #{track_id}" + (f' / {plate}' if plate else '')
        return {
            'kind': 'access_granted' if match else 'stranger_alert',
            'identity': identity,
            'confidence': confidence if method == 'exact_plate' else (match.get('similarity', 0) if match else 0),
            'method': method,
            'text': name,
            'color': (0, 170, 0) if match and method == 'exact_plate' and confidence >= 0.85 else (0, 170, 255) if match else (0, 0, 230),
            'meta': {
                'plate': plate,
                'ocr_confidence': confidence,
                'track_id': track_id,
                'appearance_model': appearance_model,
                'embedding_dim': embedding_dim,
                'advisory_match': (method != 'exact_plate')
            }
        }
