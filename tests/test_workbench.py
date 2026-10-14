import json
import shutil
import time
from pathlib import Path

import pytest

from module8_dashboard.backend.event_store import EventStore
from module8_dashboard.backend.test_workbench import TestWorkbench as Workbench, component_checks
from module8_dashboard.backend.vision import FaceEngine, ROOT


def wait_for(manager,run_id):
    deadline=time.monotonic()+30
    while time.monotonic()<deadline:
        report=manager.get(run_id)
        if report['status'] in {'completed','error'}:
            return report
        time.sleep(.05)
    raise AssertionError('Test workbench did not finish')


def test_component_results_separate_functionality_and_missing_research():
    checks=component_checks(ROOT)
    assert {f'M{index}' for index in range(1,11)} <= {row['module'] for row in checks}
    assert any(row['module']=='M1' and row['status']=='NOT_IMPLEMENTED' and 'spoof' in row['name'] for row in checks)
    assert any(row['module']=='M7' and row['status']=='NOT_IMPLEMENTED' for row in checks)
    assert any(row['status']=='NEEDS_DATA' for row in checks)
    assert all(row['status']=='PASS' for row in checks if row['module'] in {'M4','M6','M9','M10'} and row['status']!='NOT_IMPLEMENTED')


@pytest.fixture
def media_manager(tmp_path):
    sample=ROOT/'logs/demo_data/opencv_lena.jpg'
    if not sample.exists() or not (ROOT/'models/face_recognition_sface_2021dec.onnx').exists():
        pytest.skip('Optional real model/demo assets are unavailable')
    import cv2
    store=EventStore(str(tmp_path/'production.db'))
    engine=FaceEngine(store)
    engine.register('known-test-person','resident',cv2.imread(str(sample)))
    return Workbench(ROOT,store.db_path),store,sample


def submit(manager,sample,expected_name=None,expected_count=None):
    run_id=manager.reserve('media_test')
    path=manager.folder(run_id)/'input.jpg'
    shutil.copyfile(sample,path)
    manager.media(run_id,path,'face',1,expected_name,expected_count)
    return run_id,wait_for(manager,run_id),path


def test_supplied_image_uses_copied_gallery_without_polluting_production(media_manager):
    manager,store,sample=media_manager
    run_id,report,input_path=submit(manager,sample,'known-test-person',1)
    assert report['status']=='completed',report
    assert report['verdict']=='PASS'
    assert report['gallery_count']==1
    assert report['sampled_frames'][0]['faces'][0]['label']=='known-test-person'
    assert report['events'][0]['decision']=='REVIEW_REQUIRED'
    assert (manager.folder(run_id)/'preview.jpg').exists()
    assert store.get_events()==[]
    with store._conn() as db:
        assert db.execute('SELECT COUNT(*) FROM cpu_faces').fetchone()[0]==1
        assert not db.execute("SELECT 1 FROM sqlite_master WHERE name='review_requests'").fetchone()
    # Uploaded input is removed; only the result/annotated evidence remains.
    deadline=time.monotonic()+1
    while input_path.exists() and time.monotonic()<deadline:time.sleep(.01)
    assert not input_path.exists()


def test_wrong_expectation_is_failure_not_successful_execution(media_manager):
    manager,_,sample=media_manager
    _,report,_=submit(manager,sample,'different-person',2)
    assert report['status']=='completed'
    assert report['verdict']=='FAIL'
    assert all(row['status']=='FAIL' for row in report['assertions'])


def test_unlabelled_execution_is_not_accuracy_pass(media_manager):
    manager,_,sample=media_manager
    _,report,_=submit(manager,sample)
    assert report['verdict']=='OBSERVED_NO_EXPECTATIONS'
    assert not report['assertions']


def test_invalid_media_reports_error_and_releases_busy_slot(tmp_path):
    store=EventStore(str(tmp_path/'production.db'))
    manager=Workbench(ROOT,store.db_path)
    run_id=manager.reserve('media_test')
    path=manager.folder(run_id)/'input.jpg'
    path.write_bytes(b'not an image')
    manager.media(run_id,path,'low_light',1)
    report=wait_for(manager,run_id)
    assert report['status']=='error'
    deadline=time.monotonic()+1
    while manager.active_id and time.monotonic()<deadline:time.sleep(.01)
    assert manager.active_id is None
    with pytest.raises(ValueError):manager.folder('../events.db')
