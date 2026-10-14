"""Regression: cold webcams may take longer than the condition timeout."""
import time

import numpy as np

from module8_dashboard.backend.camera_manager import CameraStream


def test_slow_first_camera_frame_does_not_kill_recognition_worker(monkeypatch):
    class ColdCamera:
        calls=0
        def isOpened(self): return True
        def set(self,*args): return True
        def get(self,*args): return 30
        def read(self):
            self.calls+=1
            time.sleep(.75 if self.calls==1 else .03)
            return True,np.zeros((48,64,3),dtype=np.uint8)
        def release(self): pass
    device=ColdCamera()
    monkeypatch.setattr('module8_dashboard.backend.camera_manager.cv2.VideoCapture',lambda *args:device)
    processed=[]
    stream=CameraStream(0,pipeline_func=lambda frame:processed.append(frame.shape))
    assert stream.start()
    try:
        deadline=time.monotonic()+2
        while not processed and time.monotonic()<deadline:
            time.sleep(.02)
        assert processed, 'Preview started but recognition died before the first camera frame arrived'
        assert stream._inference_thread.is_alive()
    finally:
        stream.stop()
