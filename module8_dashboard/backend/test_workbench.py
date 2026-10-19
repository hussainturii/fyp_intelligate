"""Isolated component checks and evidence-based tests of user-supplied media."""
from __future__ import annotations
__test__ = False

import json
import os
import shutil
import sqlite3
import tempfile
import threading
import time
import uuid
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

from .auth import Auth
from .event_store import EventStore
from .software_runtime import SoftwareRuntime
from .vision import FaceEngine, INFERENCE_LOCK, ReplayPipeline, Tracker, enhance


def component_checks(root):
    """Narrow functional checks, never a declaration of full SRS compliance."""
    checks=[]
    def check(module,name,action):
        started=time.perf_counter()
        try:
            detail=action()
            status='PASS'
        except Exception as exc:
            detail=f'{type(exc).__name__}: {exc}'
            status='FAIL'
        checks.append({'module':module,'name':name,'status':status,'detail':detail,
                       'duration_ms':round((time.perf_counter()-started)*1000,1)})
    def require(condition,message):
        if not condition:
            raise AssertionError(message)
    with tempfile.TemporaryDirectory(prefix='intelligate-check-') as temp:
        store=EventStore(str(Path(temp)/'checks.db'))
        clock=[time.time()]
        runtime=SoftwareRuntime(store,lambda:clock[0])
        def face_models():
            engine=FaceEngine(store)
            sample=Path(root)/'logs/demo_data/opencv_lena.jpg'
            if not sample.exists():
                raise FileNotFoundError('Prepare optional demo data to run the real-model smoke check')
            frame=cv2.imread(str(sample))
            rows=engine.detect(frame)
            require(len(rows)==1,'Expected one face in the official static fixture')
            engine.register('isolated-smoke','test',frame)
            score,name=engine.match(engine.feature(frame,rows[0]))
            require(name=='isolated-smoke' and score>.95,'Self-match failed')
            return 'YuNet detection, encrypted enrolment and SFace self-match succeeded in a temporary gallery. Self-match is not accuracy evaluation.'
        check('M1','Real face-model smoke test',face_models)
        def vehicle_storage():
            from module2_vehicle.database.vehicle_db import VehicleDatabase
            db=VehicleDatabase(str(Path(temp)/'vehicles.h5'),str(Path(temp)/'unknown.h5'))
            db.register('TEST123','isolated-owner')
            require(db.lookup_by_plate('TEST123')['method']=='exact_plate','Exact plate lookup failed')
            require(db.lookup_by_plate('ZZZ999') is None,'Unregistered plate falsely matched')
            return 'Temporary vehicle registration and exact/unregistered plate lookup passed; OCR accuracy was not tested.'
        check('M2','Vehicle registry and exact plate lookup',vehicle_storage)
        def low_light():
            image=np.full((64,64,3),12,dtype=np.uint8)
            output=enhance(image)
            require(output.shape==image.shape and np.isfinite(output).all(),'Invalid enhancement output')
            return f'CLAHE processed a dark RGB fixture; mean intensity {image.mean():.1f} -> {output.mean():.1f}. No detection improvement claim.'
        check('M3','Low-light processing',low_light)
        def visits():
            require(runtime.visit('test-stranger','test-source','visit-one',clock[0])==1,'First visit missing')
            require(runtime.visit('test-stranger','test-source','visit-one',clock[0])==1,'Duplicate frame counted as a visit')
            runtime.visit('test-stranger','test-source','visit-two',clock[0]+10)
            require(runtime.visit('test-stranger','test-source','visit-three',clock[0]+20)==3,'Third visit missing')
            require(store.count_open_alerts()==1,'Repeat-visit escalation missing')
            return 'One appearance is deduplicated; three distinct visits produce one repeat-visitor alert.'
        check('M4','Visit deduplication and repeat alert',visits)
        def tracking():
            tracker=Tracker()
            first=tracker.update([((0,0,20,20),'person',.9),((50,0,70,20),'person',.9)],10)
            second=tracker.update([((1,0,21,20),'person',.9),((51,0,71,20),'person',.9)],10.2)
            require(len(second)==2 and [r['track_id'] for r in first]==[r['track_id'] for r in second],'Track count/association failed')
            return 'Two synthetic tracks remained distinct over adjacent observations; real-video counting accuracy still requires labelled input.'
        check('M5','Basic tracking/counting',tracking)
        def interlocks():
            runtime.command('OPEN','isolated-check')
            runtime.obstruction(True,'test')
            require(runtime.gate()['state']=='STOPPED','Obstruction did not stop motion')
            try:
                runtime.command('OPEN','isolated-check')
            except ValueError:
                pass
            else:
                raise AssertionError('Motion was admitted through an obstruction')
            runtime.obstruction(False,'test')
            runtime.command('RESET','isolated-check')
            return 'Obstruction stopped and blocked virtual gate motion. This does not test a physical actuator.'
        check('M6','Virtual obstruction safety interlock',interlocks)
        def reporting():
            report=runtime.report()
            require(report['total']>0 and report['evidence'],'No evidence in report')
            require('not a VLM' in report['method'],'Template output was not labelled')
            return f"Template report contains {report['total']} stored test events and linked evidence IDs."
        check('M7','Evidence report generation',reporting)
        def frontend():
            require((Path(root)/'module8_dashboard/frontend/dist/index.html').exists(),'Run npm run build')
            return 'Production dashboard bundle exists. This check does not exercise browser interactions.'
        check('M8','Frontend bundle availability',frontend)
        def gate_cycle():
            runtime.command('OPEN','isolated-check','idempotent')
            clock[0]+=1
            require(runtime.tick()['state']=='OPEN','Virtual gate failed to finish opening')
            runtime.command('STOP','isolated-check')
            clock[0]+=10
            require(runtime.tick()['state']=='STOPPED','STOP was not preserved')
            return 'Virtual opening and STOP pre-emption passed.'
        check('M9','Virtual gate transitions',gate_cycle)
        def auth_storage():
            auth=Auth(store)
            auth.create('isolated-admin','isolated-check-password','admin',bootstrap=True)
            token=auth.login('isolated-admin','isolated-check-password','test')
            require(auth.user(token)['role']=='admin','Session/role lookup failed')
            auth.logout(token)
            require(auth.user(token) is None,'Logout did not revoke the session')
            reopened=EventStore(str(store.db_path))
            try:
                require(reopened.count_events(since_hours=24)>0,'Events did not persist')
            finally:
                reopened.close()
                store.close()
            return 'Temporary account, password verification, session/role lookup, logout and event persistence passed. HTTP RBAC is covered separately by pytest.'
        check('M10','Authentication and persistence',auth_storage)
    for module,name,detail in [
        ('M1','Validated anti-spoofing','No validated anti-spoof model is connected to the CPU face path; access requires review.'),
        ('M2','Trained vehicle Re-ID','Appearance uses colour histograms; trained Re-ID is not connected.'),
        ('M4','SSM behaviour prediction','Current visit/dwell analytics use rules, not SSM inference.'),
        ('M7','VLM incident captions','Reports use event templates, not VLM image understanding.')]:
        checks.append({'module':module,'name':name,'status':'NOT_IMPLEMENTED','detail':detail})
    for module,name,detail in [
        ('M1','Held-out face recognition accuracy','Provide a separate image/video and an expected identity; enrolment self-match does not establish accuracy.'),
        ('M2','Plate OCR and vehicle-video accuracy','Provide readable vehicle footage and expected plate labels.'),
        ('M3','Thermal detection accuracy','Provide genuine infrared recordings with annotations; an unannotated illustration is not an accuracy benchmark.'),
        ('M5','Queue count accuracy','Provide clips with reference counts and ROI definitions.')]:
        checks.append({'module':module,'name':name,'status':'NEEDS_DATA','detail':detail})
    return checks


class TestWorkbench:
    def __init__(self,root,production_db):
        self.root,self.production_db=Path(root),Path(production_db)
        self.directory=self.production_db.parent/'testing'
        self.directory.mkdir(parents=True,exist_ok=True)
        self.lock=threading.Lock()
        self.active_id=None
        for report in self.directory.glob('*/report.json'):
            try:
                data=json.loads(report.read_text(encoding='utf-8'))
                if data.get('status') in {'queued','running'}:
                    data.update(status='error',error='Backend restarted before this run completed; run the test again')
                    self.write(report.parent.name,data)
            except (OSError,ValueError):
                continue

    def folder(self,run_id):
        # Only server-generated UUID directories may be read.
        if str(uuid.UUID(run_id))!=run_id:
            raise ValueError('Invalid run ID')
        return self.directory/run_id

    def write(self,run_id,report):
        folder=self.folder(run_id)
        folder.mkdir(exist_ok=True)
        pending=folder/'report.pending'
        pending.write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8')
        for attempt in range(4):
            try:
                pending.replace(folder/'report.json')
                break
            except PermissionError:
                if attempt==3:
                    raise
                time.sleep(.02)

    def get(self,run_id):
        path=self.folder(run_id)/'report.json'
        if not path.exists():
            raise FileNotFoundError('Test run not found')
        return json.loads(path.read_text(encoding='utf-8'))

    def reserve(self,kind):
        with self.lock:
            if self.active_id:
                raise ValueError('A test is already running; wait for it to finish')
            run_id=str(uuid.uuid4())
            self.active_id=run_id
        self.write(run_id,{'id':run_id,'kind':kind,'status':'queued','created_at':datetime.now(timezone.utc).isoformat()})
        return run_id

    def launch(self,run_id,action,input_path=None):
        def execute():
            report=self.get(run_id)
            report['status']='running';self.write(run_id,report)
            try:
                report.update(action())
                report['status']='completed'
            except Exception as exc:
                report.update(status='error',error=f'{type(exc).__name__}: {exc}')
            finally:
                report['finished_at']=datetime.now(timezone.utc).isoformat()
                try:
                    self.write(run_id,report)
                finally:
                    try:
                        if input_path:
                            Path(input_path).unlink(missing_ok=True)
                    finally:
                        with self.lock:
                            self.active_id=None
        threading.Thread(target=execute,daemon=True,name=f'test-{run_id[:8]}').start()

    def checks(self):
        run_id=self.reserve('component_checks')
        self.launch(run_id,lambda:{'checks':component_checks(self.root),'scope':'Narrow functional checks; not full SRS compliance or a real-data accuracy evaluation.'})
        return run_id

    def media(self,run_id,path,mode,max_frames,expected_identity=None,expected_count=None):
        def action():
            return self.evaluate(run_id,path,mode,max_frames,expected_identity,expected_count)
        self.launch(run_id,action,path)

    def evaluate(self,run_id,path,mode,max_frames,expected_identity,expected_count):
        timeline=[0.0]
        records=[]
        face_frames=[]
        latencies=[]
        last_preview=None
        with tempfile.TemporaryDirectory(prefix='intelligate-media-') as directory:
            sandbox=Path(directory)
            store=EventStore(str(sandbox/'test.db'))
            runtime=SoftwareRuntime(store,lambda:time.time()+timeline[0])
            face_engine,vehicle_db=None,None
            if mode=='face':
                with INFERENCE_LOCK:
                    # Preserve current gallery/key; test strangers and events go only to temp DB.
                    key=self.production_db.parent/'biometric.key'
                    if key.exists():
                        shutil.copyfile(key,sandbox/'biometric.key')
                    face_engine=FaceEngine(store)
                    with sqlite3.connect(self.production_db) as source:
                        exists=source.execute("SELECT 1 FROM sqlite_master WHERE name='cpu_faces'").fetchone()
                        rows=source.execute('SELECT id,name,role,embedding,created FROM cpu_faces').fetchall() if exists else []
                    with store._conn() as dest:
                        dest.executemany('INSERT INTO cpu_faces VALUES (?,?,?,?,?)',rows)
            if mode=='vehicle':
                from module2_vehicle.database.vehicle_db import VehicleDatabase
                with INFERENCE_LOCK:
                    current=self.production_db.parent/'cpu_vehicles.h5'
                    if current.exists():
                        shutil.copyfile(current,sandbox/'vehicles.h5')
                    vehicle_db=VehicleDatabase(str(sandbox/'vehicles.h5'),str(sandbox/'unknown.h5'))
            def callback(event):
                event_meta = event.setdefault('meta', {})
                event_meta.setdefault('provenance', 'perception_internal')
                event_meta.setdefault('trusted_perception', True)
                decision=runtime.observe(event, trusted=True) if event['module'] in {'face','vehicle'} else {'event_id':runtime.emit(event['module'],event['event_type'],event.get('identity'),**event.get('meta',{}))}
                records.append({**event,**decision})
            pipeline=ReplayPipeline(mode,runtime,callback,face_engine,vehicle_db,clock=lambda:timeline[0])
            capture=None
            is_image=Path(path).suffix.lower() in {'.jpg','.jpeg','.png','.webp','.bmp'}
            if is_image:
                from PIL import Image
                with Image.open(path) as image:
                    if image.width*image.height>20_000_000:
                        raise ValueError('Image exceeds 20 megapixels')
                image=cv2.imread(str(path))
                if image is None:
                    raise ValueError('Cannot decode the uploaded image')
                iterator=[(0,image)]
            else:
                capture=cv2.VideoCapture(str(path))
                if not capture.isOpened():
                    capture.release()
                    raise ValueError('Cannot decode the uploaded video')
                fps=capture.get(cv2.CAP_PROP_FPS)
                fps=fps if np.isfinite(fps) and 1<=fps<=120 else 25
                def sample_frames():
                    stride=max(1,round(fps/5))
                    for index in range(max_frames):
                        position=index*stride
                        capture.set(cv2.CAP_PROP_POS_FRAMES,position)
                        ok,frame=capture.read()
                        if not ok:
                            break
                        yield position/fps,frame
                iterator=sample_frames()
            try:
                for source_seconds,frame in iterator:
                    timeline[0]=source_seconds
                    scale=min(1,960/frame.shape[1])
                    if scale<1:
                        frame=cv2.resize(frame,None,fx=scale,fy=scale)
                    started=time.perf_counter()
                    result=pipeline.process_frame(frame)
                    latencies.append((time.perf_counter()-started)*1000)
                    last_preview=result['annotated_frame']
                    face_frames.append({'source_seconds':round(source_seconds,2),
                                        'faces_detected':int(result.get('faces_detected',0)),
                                        'count':int(result.get('count',0)),
                                        'obstruction':bool(result.get('obstruction',False)),
                                        'faces':list(pipeline.metrics.get('face_results',[]))})
            finally:
                if capture:
                    capture.release()
            if not latencies:
                raise ValueError('The file contained no decodable frames')
            cv2.imwrite(str(self.folder(run_id)/'preview.jpg'),last_preview)
            assertions=[]
            if expected_identity is not None:
                if mode!='face':
                    raise ValueError('Expected identity is supported for face tests only')
                expected=expected_identity.strip().casefold()
                detected=[row for item in face_frames for row in item['faces']]
                match=[row for row in detected if row['label'].casefold()==expected]
                assertions.append({'name':'Expected identity appears','status':'PASS' if match else 'FAIL',
                                   'expected':expected_identity,'observed':sorted({row['label'] for row in detected})})
            if expected_count is not None:
                observed=[item['faces_detected'] if mode=='face' else item['count'] for item in face_frames]
                assertions.append({'name':'Expected count on every sampled frame','status':'PASS' if observed and all(value==expected_count for value in observed) else 'FAIL',
                                   'expected':expected_count,'observed':observed})
            gallery_cnt = len(face_engine.list()) if face_engine else None
            store.close()
            return {'mode':mode,'frames_processed':len(latencies),'sampled_frames':face_frames,
                    'events':records,'event_count':len(records),'event_types':dict(Counter(row['event_type'] for row in records)),
                    'metrics':dict(pipeline.metrics),'assertions':assertions,
                    'verdict':('FAIL' if any(row['status']=='FAIL' for row in assertions) else 'PASS' if assertions else 'OBSERVED_NO_EXPECTATIONS'),
                    'timing':{'median_ms':round(float(np.median(latencies)),2),'p95_ms':round(float(np.percentile(latencies,95)),2)},
                    'gallery_count':gallery_cnt,
                    'preview_url':f'/api/testing/runs/{run_id}/preview',
                    'scope':'Uses a copied gallery and temporary events/device state; never opens the actual virtual gate or writes test visits to the live timeline. Video observations use source timestamps.',
                    'limitations':['No validated anti-spoofing or VLM/SSM inference.','An unlabeled successful execution is not an accuracy pass.','A single expected identity/count assertion is not a dataset accuracy benchmark.']}
