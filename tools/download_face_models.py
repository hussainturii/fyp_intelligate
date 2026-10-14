"""Download official OpenCV Zoo CPU face models; keep assets out of Git."""
import hashlib
import json
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODELS = {
    'face_detection_yunet_2023mar.onnx': 'face_detection_yunet',
    'face_recognition_sface_2021dec.onnx': 'face_recognition_sface',
}

if __name__ == '__main__':
    directory = ROOT / 'models'
    directory.mkdir(exist_ok=True)
    records = []
    for name, project in MODELS.items():
        url = f'https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models/{project}/{name}'
        target = directory / name
        if not target.exists():
            print('Downloading', name, flush=True)
            temporary = target.with_suffix('.partial')
            with urllib.request.urlopen(url, timeout=60) as source, temporary.open('wb') as dest:
                while chunk := source.read(1024 * 1024):
                    dest.write(chunk)
            temporary.replace(target)
        records.append({'file': name, 'source': url, 'sha256': hashlib.sha256(target.read_bytes()).hexdigest(),
                        'terms': f'https://github.com/opencv/opencv_zoo/tree/main/models/{project}'})
        print(name, target.stat().st_size, 'bytes', flush=True)
    (directory / 'download_manifest.json').write_text(json.dumps(records, indent=2), encoding='utf-8')
