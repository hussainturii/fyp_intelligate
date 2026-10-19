import json
import time

import pytest

from module8_dashboard.backend.event_store import EventStore
from module8_dashboard.backend.software_runtime import SoftwareRuntime
from module8_dashboard.backend.vision import Tracker, enhance, appearance


@pytest.fixture
def system(tmp_path):
    clock = [1_000_000.0]
    store = EventStore(str(tmp_path / 'test.db'))
    return SoftwareRuntime(store, lambda: clock[0]), store, clock


def test_stop_preempts_motion_and_persists(system):
    runtime, store, clock = system
    runtime.command('OPEN', 'admin')
    assert runtime.gate()['state'] == 'OPENING'
    runtime.command('STOP', 'admin', expires=clock[0]-1)
    clock[0] += 10
    assert runtime.tick()['state'] == 'STOPPED'
    assert SoftwareRuntime(store).gate()['state'] == 'STOPPED'


def test_obstruction_blocks_both_motion_commands(system):
    runtime, _, _ = system
    runtime.obstruction(True)
    for command in ['OPEN', 'CLOSE', 'RESET']:
        with pytest.raises(ValueError):
            runtime.command(command, 'admin')
    runtime.obstruction(False)
    assert runtime.command('RESET', 'admin')['state'] == 'CLOSED'


def test_manual_obstruction_not_cleared_by_video(system):
    runtime, _, _ = system
    runtime.obstruction(True, 'operator')
    assert runtime.obstruction(False, 'replay-0')['obstruction'] is True


def test_command_idempotence_and_conflict(system):
    runtime, store, clock = system
    first = runtime.command('OPEN', 'admin', 'same')
    clock[0] += 1
    runtime.tick()
    assert runtime.command('OPEN', 'admin', 'same') == first
    assert store.count_events('gate', 'open', 876000) == 1
    with pytest.raises(ValueError):
        runtime.command('CLOSE', 'admin', 'same')


def test_lockdown_and_restart_never_resume_motion(system):
    runtime, store, _ = system
    runtime.command('OPEN', 'admin')
    assert SoftwareRuntime(store).gate()['state'] == 'STOPPED'
    runtime.command('LOCKDOWN', 'admin')
    with pytest.raises(ValueError):
        runtime.command('OPEN', 'admin')


def test_expired_command_rejected(system):
    runtime, _, clock = system
    with pytest.raises(ValueError):
        runtime.command('OPEN', 'admin', expires=clock[0]-1)


def test_review_policy_never_grants_appearance_match(system):
    runtime, store, _ = system
    event = {'module': 'vehicle', 'event_type': 'access_granted', 'identity': 'owner', 'method': 'reid', 'confidence': 0.99}
    result = runtime.observe(event)
    assert result['decision'] == 'REVIEW_REQUIRED'
    assert runtime.gate()['state'] == 'CLOSED'
    assert store.get_events()[0]['event_type'] == 'recognition_candidate'
    review = runtime.reviews()[0]
    assert runtime.review(review['id'], True, 'operator')['gate']['state'] == 'OPENING'
    with pytest.raises(ValueError):
        runtime.review(review['id'], True, 'operator')


def test_face_requires_validated_liveness_and_pending_dedup(system):
    runtime, _, _ = system
    event = {'module': 'face', 'event_type': 'access_granted', 'identity': 'known', 'meta': {'validated_liveness': False}}
    assert runtime.observe(event)['decision'] == 'REVIEW_REQUIRED'
    runtime.observe(event)
    assert len(runtime.reviews()) == 1


def test_spoof_denied_and_alerted(system):
    runtime, store, _ = system
    assert runtime.observe({'module': 'face', 'event_type': 'spoof_detected'})['decision'] == 'DENY'
    assert store.get_alerts()[0]['category'] == 'spoof_attempt'
    assert runtime.reviews() == []


def test_high_confidence_exact_plate_allows_unless_interlocked(system):
    runtime, _, _ = system
    trusted_event = {
        'module': 'vehicle',
        'event_type': 'access_granted',
        'method': 'exact_plate',
        'meta': {'ocr_confidence': 0.91, 'provenance': 'perception_internal', 'trusted_perception': True}
    }
    assert runtime.observe(trusted_event, trusted=True)['decision'] == 'ALLOW'
    runtime.obstruction(True)
    assert runtime.observe(trusted_event, trusted=True)['decision'] == 'DENY'
    runtime.obstruction(False)

    # Missing provenance must fail closed and require review
    unverified_event = {
        'module': 'vehicle',
        'event_type': 'access_granted',
        'method': 'exact_plate',
        'meta': {'ocr_confidence': 0.91}
    }
    assert runtime.observe(unverified_event)['decision'] == 'REVIEW_REQUIRED'


def test_expired_review_cannot_open(system):
    runtime, _, clock = system
    runtime.observe({'module': 'face', 'event_type': 'stranger_alert'})
    request_id = runtime.reviews()[0]['id']
    clock[0] += 121
    with pytest.raises(ValueError):
        runtime.review(request_id, True, 'admin')


def test_distinct_visits_only_and_window(system):
    runtime, store, clock = system
    for _ in range(10):
        assert runtime.visit('stranger', 'camera', 'one', clock[0]) == 1
    assert runtime.visit('stranger', 'camera', 'two', clock[0]+10) == 2
    assert runtime.visit('stranger', 'camera', 'three', clock[0]+20) == 3
    assert store.count_open_alerts() == 1
    runtime.visit('stranger', 'camera', 'three', clock[0]+30)
    assert store.count_open_alerts() == 1
    assert runtime.visit('stranger', 'camera', 'four', clock[0]+86401) == 3
    assert runtime.visit('other', 'camera', 'one', clock[0]+86401) == 1


def test_report_counts_beyond_preview_and_search_validation(system):
    runtime, _, _ = system
    for i in range(110):
        runtime.emit('queue', 'count_changed', count=i)
    report = runtime.report()
    assert report['total'] == 110 and len(report['evidence']) == 100
    assert 'not a VLM' in report['method']
    assert len(runtime.search('queue today')['events']) == 110
    with pytest.raises(ValueError):
        runtime.search('tell me who looks suspicious')


def test_tracker_counts_unique_and_expires():
    tracker = Tracker()
    first = tracker.update([((0,0,10,10),'person',.9)], 0)
    second = tracker.update([((1,0,11,10),'person',.9)], 1)
    assert first[0]['track_id'] == second[0]['track_id']
    crowded = tracker.update([((1,0,11,10),'person',.9),((2,0,12,10),'person',.9)], 1.5)
    assert len({t['track_id'] for t in crowded}) == 2
    assert tracker.update([], 4) == []
    assert not tracker.tracks


def test_low_light_and_histogram_are_finite():
    import numpy as np
    frame = np.full((32,32,3), 15, np.uint8)
    assert enhance(frame).shape == frame.shape
    vector = appearance(frame)
    assert vector.shape == (768,)
    assert np.isfinite(vector).all()
    assert np.linalg.norm(vector) == pytest.approx(1)
