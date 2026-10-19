from pathlib import Path

import cv2
import numpy as np
import pytest
from cryptography.exceptions import InvalidTag

from module8_dashboard.backend.event_store import EventStore
from module8_dashboard.backend.software_runtime import SoftwareRuntime
from module8_dashboard.backend.vision import FaceEngine, ReplayPipeline, ROOT


@pytest.fixture
def engine(tmp_path):
    if not (ROOT/'models/face_recognition_sface_2021dec.onnx').exists():
        pytest.skip('Download optional real model assets with tools/download_face_models.py')
    return FaceEngine(EventStore(str(tmp_path/'face.db')))


def test_embedding_encryption_binds_record_id(engine):
    vector = np.ones(128, dtype=np.float32)
    blob = engine.seal(vector, 'one')
    assert vector.tobytes() not in blob
    assert np.array_equal(engine.open(blob, 'one'), vector)
    with pytest.raises(InvalidTag):
        engine.open(blob, 'other')
    tampered = bytearray(blob); tampered[-1] ^= 1
    with pytest.raises(InvalidTag):
        engine.open(bytes(tampered), 'one')


def test_real_face_enrolment_requires_review_before_virtual_open(engine):
    sample = ROOT/'logs/demo_data/opencv_lena.jpg'
    if not sample.exists():
        pytest.skip('Run tools/prepare_demo_data.py for the static smoke fixture')
    frame = cv2.imread(str(sample))
    identity = engine.register('OpenCV static sample', 'test', frame)
    runtime = SoftwareRuntime(engine.store)
    results = []
    pipeline = ReplayPipeline('face', runtime, lambda event:results.append(runtime.observe(event, trusted=True)), engine)
    assert pipeline.process_frame(frame)['faces_detected'] == 1
    assert results[0]['decision'] == 'REVIEW_REQUIRED'
    assert runtime.gate()['state'] == 'CLOSED'
    request = runtime.reviews()[0]
    runtime.review(request['id'], True, 'test-operator')
    assert runtime.gate()['state'] == 'OPENING'
    assert identity


def test_registry_reload_and_delete(engine):
    with engine.store._conn() as db:
        db.execute('INSERT INTO cpu_faces VALUES (?,?,?,?,?)', ('sample', 'name', 'test', engine.seal(np.ones(128,dtype=np.float32),'sample'), 1))
    reloaded = FaceEngine(engine.store)
    assert 'name' in reloaded.list()
    assert reloaded.delete('name') is True
    assert reloaded.delete('name') is False
