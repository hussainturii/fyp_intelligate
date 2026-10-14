import importlib

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv('IG_EVENT_DB', str(tmp_path / 'api.db'))
    import module8_dashboard.backend.main as main
    importlib.reload(main)
    with TestClient(main.app) as client:
        yield client


def admin(client):
    credentials = {'username': 'admin', 'password': 'test-password-123'}
    assert client.post('/api/auth/setup', json=credentials).status_code == 200
    assert client.post('/api/auth/login', json=credentials).status_code == 200


def test_setup_authentication_and_origin(client):
    assert client.get('/api/gate').status_code == 401
    admin(client)
    assert client.get('/api/gate').json()['simulated']
    assert client.post('/api/auth/setup', json={'username':'other','password':'another-password'}).status_code == 409
    assert client.post('/api/gate/command', json={'command':'OPEN'}, headers={'Origin':'https://untrusted.example'}).status_code == 403
    assert client.post('/api/auth/logout').status_code == 200
    assert client.get('/api/gate').status_code == 401


def test_viewer_read_only_and_staff_cannot_enrol(client):
    admin(client)
    assert client.post('/api/auth/users', json={'username':'viewer','password':'viewer-password','role':'viewer'}).status_code == 200
    assert client.post('/api/auth/users', json={'username':'staff','password':'staff-password','role':'security_staff'}).status_code == 200
    client.post('/api/auth/login',json={'username':'viewer','password':'viewer-password'})
    assert client.get('/api/modules').status_code == 200
    assert client.post('/api/gate/command', json={'command':'STOP'}).status_code == 403
    client.post('/api/auth/login',json={'username':'staff','password':'staff-password'})
    assert client.post('/api/gate/command', json={'command':'STOP'}).status_code == 200
    assert client.post('/api/vehicles/register',data={'plate':'ABC123','owner':'Person'}).status_code == 403


def test_api_spoofed_liveness_cannot_grant_and_review_interlock(client):
    admin(client)
    result = client.post('/api/events',json={'module':'face','event_type':'access_granted','identity':'known',
        'meta':{'validated_liveness':True,'liveness_score':1}}).json()
    assert result['decision'] == 'REVIEW_REQUIRED'
    pending = client.get('/api/reviews').json()[0]
    client.post('/api/gate/obstruction',json={'present':True})
    assert client.post('/api/reviews/'+pending['id'],json={'approve':True}).status_code == 409
    assert client.get('/api/gate').json()['state'] == 'STOPPED'


def test_api_validation_and_report(client):
    admin(client)
    assert client.post('/api/gate/command',json={'command':'INVALID'}).status_code == 422
    assert client.post('/api/scene/config',json={'gate_roi':[.8,.1,.2,.5]}).status_code == 422
    assert client.get('/api/search',params={'q':'unsupported question'}).status_code == 422
    assert client.get('/api/reports').json()['method'] == 'event-template (not a VLM)'
    assert client.post('/api/cameras/-1/start').status_code == 422
    assert client.get('/api/stream/0').status_code == 404
    assert client.put('/api/alerts/999/resolve').status_code == 404


def test_bad_password_rate_limit(client):
    admin(client)
    for _ in range(5):
        assert client.post('/api/auth/login',json={'username':'admin','password':'incorrect'}).status_code == 401
    assert client.post('/api/auth/login',json={'username':'admin','password':'incorrect'}).status_code == 429


def test_websocket_requires_session_and_receives_init(client):
    admin(client)
    with client.websocket_connect('/ws') as socket:
        assert socket.receive_json()['type'] == 'init'
        socket.send_text('ping')
        assert socket.receive_json()['type'] == 'pong'


def test_video_upload_and_persistence(client, tmp_path):
    import cv2
    import numpy as np
    admin(client)
    video = tmp_path/'clip.avi'
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*'MJPG'), 10, (64,48))
    for _ in range(5): writer.write(np.zeros((48,64,3),dtype=np.uint8))
    writer.release()
    response = client.post('/api/cameras/0/upload-video', data={'mode':'raw'}, files={'video':('clip.avi',video.read_bytes(),'video/avi')})
    assert response.status_code == 200, response.text
    assert client.get('/api/cameras').json()['0']['replayed']
    assert client.post('/api/cameras/0/stop').status_code == 200
    assert any(event['event_type']=='replay_started' for event in client.get('/api/events').json())


def test_vehicle_registration_without_optional_image(client):
    admin(client)
    response = client.post('/api/vehicles/register', data={'plate':'ABC-123','owner':'Test owner'}, files={'image':('', b'', 'application/octet-stream')})
    assert response.status_code == 200, response.text
    assert 'ABC123' in client.get('/api/vehicles').json()['vehicles']
    assert client.delete('/api/vehicles/ABC123').status_code == 200
    assert client.delete('/api/vehicles/ABC123').status_code == 404


def test_curated_llvip_cannot_be_used_for_identity(client):
    admin(client)
    assert client.post('/api/demo/replay', json={'source':'llvip_infrared','mode':'face'}).status_code == 422


def test_live_camera_endpoint_with_synthetic_capture(client, monkeypatch):
    import cv2
    import numpy as np
    import time
    admin(client)
    class Capture:
        released=False
        def isOpened(self):return True
        def set(self,*args):return True
        def get(self,*args):return 30
        def read(self):
            time.sleep(.01)
            return True,np.zeros((48,64,3),dtype=np.uint8)
        def release(self):self.released=True
    capture=Capture()
    monkeypatch.setattr('module8_dashboard.backend.camera_manager.cv2.VideoCapture',lambda *args:capture)
    response=client.post('/api/cameras/0/start?mode=raw')
    assert response.status_code==200,response.text
    assert response.json()['replayed'] is False
    assert client.get('/api/cameras').json()['0']['replayed'] is False
    assert client.post('/api/cameras/0/stop').status_code==200
    assert capture.released
    assert any(event['event_type']=='camera_started' for event in client.get('/api/events').json())


def test_camera_enrolment_validates_missing_source_and_staff_role(client):
    admin(client)
    assert client.post('/api/faces/register-camera',json={'name':'Member','camera_id':0}).status_code==422
    client.post('/api/auth/users',json={'username':'staff','password':'staff-password','role':'security_staff'})
    client.post('/api/auth/login',json={'username':'staff','password':'staff-password'})
    assert client.post('/api/faces/register-camera',json={'name':'Member','camera_id':0}).status_code==403


def test_test_center_enforces_roles_and_rejects_bad_inputs(client):
    admin(client)
    assert client.post('/api/testing/runs',data={'mode':'face'},files={'file':('input.txt',b'test','text/plain')}).status_code==422
    assert client.post('/api/testing/camera',json={'camera_id':0}).status_code==422
    assert client.get('/api/testing/runs/not-a-run').status_code==404
    assert client.get('/api/testing/runs/not-a-run/preview').status_code==404
    client.post('/api/auth/users',json={'username':'read-only','password':'viewer-password','role':'viewer'})
    client.post('/api/auth/login',json={'username':'read-only','password':'viewer-password'})
    assert client.post('/api/testing/checks').status_code==403
    assert client.post('/api/testing/camera',json={'camera_id':0}).status_code==403


def test_test_center_image_run_returns_evidence(client):
    import cv2
    import numpy as np
    import time
    admin(client)
    ok,content=cv2.imencode('.png',np.full((48,64,3),10,dtype=np.uint8))
    assert ok
    response=client.post('/api/testing/runs',data={'mode':'low_light'},files={'file':('dark.png',content.tobytes(),'image/png')})
    assert response.status_code==202,response.text
    run_id=response.json()['id']
    deadline=time.monotonic()+10
    report={}
    while time.monotonic()<deadline:
        report=client.get('/api/testing/runs/'+run_id).json()
        if report['status'] in {'completed','error'}:break
        time.sleep(.05)
    assert report['status']=='completed',report
    assert report['verdict']=='OBSERVED_NO_EXPECTATIONS'
    assert client.get(report['preview_url']).status_code==200
    assert client.get('/api/events').json()==[]
    assert client.get('/api/testing/runs').json()[0]['id']==run_id
