"""Prepare official LLVIP illustration replays; these are unannotated demo data."""
import hashlib
import json
import urllib.request
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
DIRECTORY = ROOT / 'logs/demo_data'


def main():
    DIRECTORY.mkdir(parents=True, exist_ok=True)
    sources = {
        'llvip_illustration.gif': 'https://raw.githubusercontent.com/bupt-ai-cz/LLVIP/main/imgs/LLVIP.gif',
        'opencv_lena.jpg': 'https://raw.githubusercontent.com/opencv/opencv/master/samples/data/lena.jpg',
    }
    manifest = []
    for name, url in sources.items():
        path = DIRECTORY / name
        if not path.exists():
            with urllib.request.urlopen(url, timeout=60) as response, path.open('wb') as dest:
                while chunk := response.read(1024*1024):
                    dest.write(chunk)
        manifest.append({'file': name, 'source': url, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
    image = Image.open(DIRECTORY / 'llvip_illustration.gif')
    writers = [cv2.VideoWriter(str(DIRECTORY / name), cv2.VideoWriter_fourcc(*'mp4v'), 10, (256,256))
               for name in ['llvip_visible.mp4', 'llvip_infrared.mp4']]
    for index in range(image.n_frames):
        image.seek(index)
        frame = cv2.cvtColor(np.array(image.convert('RGB')), cv2.COLOR_RGB2BGR)
        writers[0].write(frame[:, :256])
        writers[1].write(frame[:, 256:])
    for writer in writers:
        writer.release()
    sample = cv2.imread(str(DIRECTORY / 'opencv_lena.jpg'))
    writer = cv2.VideoWriter(str(DIRECTORY / 'face_smoke.mp4'), cv2.VideoWriter_fourcc(*'mp4v'), 10, (512,512))
    for _ in range(100):
        writer.write(sample)
    writer.release()
    manifest.append({'derived': ['llvip_visible.mp4','llvip_infrared.mp4'], 'parent': 'llvip_illustration.gif',
                     'method': 'Split official paired illustration into left visible/right infrared halves; fixed 10 FPS playback',
                     'terms': 'https://github.com/bupt-ai-cz/LLVIP/blob/main/Term%20of%20Use%20and%20License.md',
                     'scope': 'Unannotated illustration; throughput/domain-shift exploration only. Not a test split or accuracy benchmark. No identity recognition on LLVIP.',
                     'citation': 'Jia et al., LLVIP: A Visible-infrared Paired Dataset for Low-light Vision, ICCV Workshops 2021'})
    manifest.append({'derived': 'face_smoke.mp4', 'parent': 'opencv_lena.jpg',
                     'scope': 'Repeated static image for recognition/enrolment smoke tests only; cannot validate liveness'})
    (DIRECTORY / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print('Prepared visible, genuine infrared and static-face smoke replays at', DIRECTORY)


if __name__ == '__main__':
    main()
