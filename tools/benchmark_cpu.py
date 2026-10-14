"""Measure real CPU inference on recorded frames. This is not an accuracy study."""
import argparse
import json
import platform
import statistics
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import cv2
import numpy as np

from module8_dashboard.backend.event_store import EventStore
from module8_dashboard.backend.vision import FaceEngine, ReplayPipeline, enhance
from module8_dashboard.backend.software_runtime import SoftwareRuntime


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--video', required=True)
    parser.add_argument('--frames', type=int, default=8)
    args = parser.parse_args()
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise SystemExit('Cannot open input video')
    frames = []
    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    for index in np.linspace(0, max(0,count-1), args.frames, dtype=int):
        cap.set(cv2.CAP_PROP_POS_FRAMES,int(index))
        ok,frame = cap.read()
        if ok:
            frames.append(cv2.resize(frame,(640,360)))
    cap.release()
    if not frames:
        raise SystemExit('No decodable frames')
    report = {'video':Path(args.video).name,'sampled_frames':len(frames),'input_size':[640,360],
              'device':'cpu','platform':platform.platform(),'opencv':cv2.__version__,
              'scope':'Inference timings only; no labelled accuracy, liveness or thermal evaluation', 'results':{}}
    with tempfile.TemporaryDirectory() as directory:
        store = EventStore(str(Path(directory)/'benchmark.db'))
        runtime = SoftwareRuntime(store)
        face = FaceEngine(store)
        scene = ReplayPipeline('scene',runtime,lambda event:None)
        tasks = {'face_detection':lambda f:len(face.detect(f)), 'scene_detection_tracking':lambda f:scene.process_frame(f)['count'], 'low_light_enhancement':lambda f:enhance(f).shape[0]}
        for name, task in tasks.items():
            task(frames[0])
            times, outputs = [], []
            for frame in frames:
                started=time.perf_counter(); outputs.append(task(frame)); times.append((time.perf_counter()-started)*1000)
            report['results'][name] = {'median_ms':round(statistics.median(times),2), 'p95_ms':round(float(np.percentile(times,95)),2),'mean_inference_fps':round(1000/statistics.mean(times),2),'outputs':outputs}
    output = ROOT/'logs'/'benchmarks'
    output.mkdir(parents=True,exist_ok=True)
    target=output/'cpu_baseline.json'
    target.write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))
    print('Saved:',target)


if __name__=='__main__':
    main()
