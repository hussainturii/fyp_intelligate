from types import SimpleNamespace
import time

import numpy as np
import pytest

from module8_dashboard.backend.event_store import EventStore
from module8_dashboard.backend.software_runtime import SoftwareRuntime
from module8_dashboard.backend.vision import ReplayPipeline, PresenceLedger, Tracker
from module8_dashboard.backend.camera_manager import CameraStream


class Faces:
    gallery_revision = 0
    def __init__(self):
        self.rows = [np.array([x,10,50,50]+[0]*10+[.99],dtype=np.float32) for x in [10,90,170,250]]
        self.feature_calls=0
    def detect(self, frame):
        return self.rows
    def feature(self, frame, row):
        self.feature_calls+=1
        return np.array([int(row[0])])
    def match(self, embedding):
        name={10:'Ali',90:'Sara',170:None,250:'Hussain'}[int(embedding[0])]
        return (.99,name) if name else (.1,None)
    def stranger(self, embedding):
        return f'unknown-{embedding[0]}'


def pipeline(tmp_path, faces, callback):
    store=EventStore(str(tmp_path/'events.db'))
    return ReplayPipeline('face',SoftwareRuntime(store),callback,faces)


def test_four_faces_once_per_continuous_appearance_and_red_unknown(tmp_path, monkeypatch):
    clock=[10.0]
    monkeypatch.setattr('module8_dashboard.backend.vision.time.monotonic',lambda:clock[0])
    faces,events=Faces(),[]
    system=pipeline(tmp_path,faces,events.append)
    frame=np.zeros((200,400,3),dtype=np.uint8)
    for _ in range(30):
        result=system.process_frame(frame)
        clock[0]+=.05
    assert faces.feature_calls==4
    assert len(events)==4
    assert len(result['tracks'])==4
    assert result['tracks'][0]['text'].startswith('Ali')
    assert result['tracks'][2]['text'].startswith('Unknown')
    assert result['tracks'][2]['color']==(0,0,230)
    assert tuple(result['annotated_frame'][55,170])==(0,0,230)
    assert system.metrics['recognition_cache_hits']==116
    assert len({event['meta']['session_id'] for event in events})==4


def test_short_miss_keeps_identity_long_exit_allows_new_event(tmp_path, monkeypatch):
    clock=[10.0]
    monkeypatch.setattr('module8_dashboard.backend.vision.time.monotonic',lambda:clock[0])
    faces,events=Faces(),[]
    system=pipeline(tmp_path,faces,events.append)
    frame=np.zeros((200,400,3),dtype=np.uint8)
    rows=faces.rows
    system.process_frame(frame)
    faces.rows=[];clock[0]+=1;system.process_frame(frame)
    faces.rows=rows;clock[0]+=.2;system.process_frame(frame)
    assert faces.feature_calls==4 and len(events)==4
    faces.rows=[];clock[0]+=2.1;system.process_frame(frame)
    faces.rows=rows;clock[0]+=.1;system.process_frame(frame)
    assert faces.feature_calls==8 and len(events)==8


def test_registry_changes_invalidate_cached_labels(tmp_path):
    faces=Faces()
    system=pipeline(tmp_path,faces,lambda event:None)
    frame=np.zeros((200,400,3),dtype=np.uint8)
    system.process_frame(frame)
    faces.gallery_revision+=1
    system.process_frame(frame)
    assert faces.feature_calls==8


def test_abrupt_face_change_rechecks_identity_in_same_position(tmp_path,monkeypatch):
    faces=Faces();faces.rows=faces.rows[:1]
    events=[]
    system=pipeline(tmp_path,faces,events.append)
    signature=[np.zeros(63,dtype=bool)]
    monkeypatch.setattr('module8_dashboard.backend.vision.crop_fingerprint',lambda crop:signature[0])
    frame=np.zeros((100,100,3),dtype=np.uint8)
    assert system.process_frame(frame)['tracks'][0]['text'].startswith('Ali')
    signature[0]=np.ones(63,dtype=bool)
    faces.match=lambda embedding:(.99,'Sara')
    assert system.process_frame(frame)['tracks'][0]['text'].startswith('Sara')
    assert faces.feature_calls==2
    assert len(events)==2


def test_presence_ledger_bounded_and_distinct_returns():
    ledger=PresenceLedger()
    first,visit=ledger.touch('person',10)
    assert first
    assert ledger.touch('person',11)==(False,visit)
    assert ledger.touch('person',13.1)[0]
    ledger.prune(20)
    assert ledger.present=={}


def test_ambiguous_overlap_does_not_reuse_old_identity_track():
    tracker=Tracker()
    original=tracker.update([((0,0,100,100),'face',.99),((30,0,130,100),'face',.99)],10)
    overlap=tracker.update([((15,0,115,100),'face',.99)],10.2)
    assert overlap[0]['ambiguous']
    assert overlap[0]['track_id'] not in {track['track_id'] for track in original}


def test_vehicle_ocr_cached_and_events_not_repeated(tmp_path):
    store=EventStore(str(tmp_path/'events.db'))
    events=[]
    system=ReplayPipeline('raw',SoftwareRuntime(store),events.append)
    system.mode='vehicle'
    box=SimpleNamespace(xyxy=[SimpleNamespace(cpu=lambda:SimpleNamespace(tolist=lambda:[20,20,100,100]))],cls=[2],conf=[.99])
    system.model=SimpleNamespace(names={2:'car'},predict=lambda *args,**kwargs:[SimpleNamespace(boxes=[box])])
    calls=[]
    def recognize(crop,track_id):
        calls.append(track_id)
        return {'kind':'access_granted','identity':'ABC123','confidence':.95,'method':'exact_plate','text':'Owner / ABC123','color':(0,170,0),'meta':{'plate':'ABC123','ocr_confidence':.95}}
    system._vehicle=recognize
    for _ in range(20):
        system.process_frame(np.zeros((150,150,3),dtype=np.uint8))
    assert len(calls)==1
    assert len(events)==1
    assert events[0]['meta']['plate']=='ABC123'


def test_preview_capture_continues_during_slow_inference(monkeypatch):
    class Capture:
        reads=0
        released=False
        def isOpened(self):return True
        def get(self,prop):return 30
        def set(self,*args):return True
        def read(self):
            time.sleep(1/30)
            self.reads+=1
            return True,np.full((48,64,3),self.reads%255,dtype=np.uint8)
        def release(self):self.released=True
    capture=Capture()
    monkeypatch.setattr('module8_dashboard.backend.camera_manager.cv2.VideoCapture',lambda *args:capture)
    calls=[]
    def slow(frame):
        calls.append(1);time.sleep(.3);return {'annotated_frame':frame}
    camera=CameraStream(0,pipeline_func=slow)
    assert camera.start()
    time.sleep(.65)
    snapshot=camera.get_snapshot()
    camera.stop()
    assert snapshot is not None
    assert capture.reads>=10
    assert len(calls)<capture.reads/3
    assert capture.released
