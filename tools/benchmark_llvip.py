"""Explore RGB/infrared domain shift on official unannotated LLVIP illustration."""
import json
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import cv2
import numpy as np
from ultralytics import YOLO
from module8_dashboard.backend.vision import enhance

if __name__ == '__main__':
    model = YOLO(str(ROOT / 'yolov8n.pt'))
    results = {}
    for mode in ['visible', 'enhanced_visible', 'infrared']:
        file = 'llvip_infrared.mp4' if mode == 'infrared' else 'llvip_visible.mp4'
        capture = cv2.VideoCapture(str(ROOT / 'logs/demo_data' / file))
        timings, counts, indices = [], [], []
        for index in [0, 30, 60, 90, 120, 150, 180, 210]:
            capture.set(cv2.CAP_PROP_POS_FRAMES, index)
            ok, frame = capture.read()
            if not ok:
                raise RuntimeError('Run tools/prepare_demo_data.py first')
            if mode == 'enhanced_visible':
                frame = enhance(frame)
            if not timings:
                model.predict(frame, imgsz=416, classes=[0], conf=.4, device='cpu', verbose=False)
            started = time.perf_counter()
            prediction = model.predict(frame, imgsz=416, classes=[0], conf=.4, device='cpu', verbose=False)[0]
            timings.append((time.perf_counter()-started)*1000)
            counts.append(len(prediction.boxes)); indices.append(index)
        capture.release()
        results[mode] = {'sample_indices': indices, 'person_detection_counts': counts,
                         'median_inference_ms': round(statistics.median(timings),2), 'p95_inference_ms': round(float(np.percentile(timings,95)),2)}
    report = {'source': 'Official LLVIP illustration GIF split into paired replays',
              'source_url': 'https://github.com/bupt-ai-cz/LLVIP', 'model':'COCO-pretrained YOLOv8n CPU',
              'input': '256x256 source, imgsz=416, confidence=0.4',
              'scope': 'Unannotated demonstration subset: no accuracy, recall, superiority or thermal temperature claims',
              'results': results}
    directory = ROOT / 'logs/benchmarks'; directory.mkdir(parents=True,exist_ok=True)
    (directory/'llvip_exploration.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))
