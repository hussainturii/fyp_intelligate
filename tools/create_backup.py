"""Create a timestamped offline backup outside Git repository with SHA-256 manifest."""
import datetime
import hashlib
import os
from pathlib import Path
import subprocess
import zipfile

repo = Path('d:/MZHR_FYP/INTELLIGATE')
backup_root = Path('d:/MZHR_FYP/backups')
backup_root.mkdir(parents=True, exist_ok=True)
timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
backup_dir = backup_root / f'intelligate_baseline_{timestamp}'
backup_dir.mkdir(parents=True, exist_ok=True)

# 1. Record Git metadata
git_log = subprocess.check_output(['git', 'log', '-1', '--stat'], cwd=repo, text=True)
git_status = subprocess.check_output(['git', 'status'], cwd=repo, text=True)
(backup_dir / 'GIT_BASELINE.txt').write_text(
    f'=== GIT LOG -1 ===\n{git_log}\n\n=== GIT STATUS ===\n{git_status}\n',
    encoding='utf-8'
)

# 2. Archive essential files (exclude .venv and .git from zip to keep it clean and fast)
zip_path = backup_root / f'intelligate_baseline_{timestamp}.zip'
manifest = []

skip_dirs = {'.venv', '.git', '.pytest_cache', 'node_modules', '__pycache__'}

with zipfile.ZipFile(zip_path, 'w', compression=zipfile.ZIP_DEFLATED) as zf:
    for root, dirs, files in os.walk(repo):
        rel_root = os.path.relpath(root, repo)
        if any(part in skip_dirs for part in Path(rel_root).parts):
            continue
        for file in sorted(files):
            fp = Path(root) / file
            arcname = os.path.relpath(fp, repo).replace('\\', '/')
            zf.write(fp, arcname)
            # Hash for manifest
            h = hashlib.sha256(fp.read_bytes()).hexdigest()
            manifest.append(f'{h}  {arcname}')

(backup_dir / 'SHA256SUMS.txt').write_text('\n'.join(manifest) + '\n', encoding='utf-8')
zip_hash = hashlib.sha256(zip_path.read_bytes()).hexdigest()
(backup_root / f'intelligate_baseline_{timestamp}.zip.sha256').write_text(
    f'{zip_hash}  {zip_path.name}\n',
    encoding='utf-8'
)

# Verify archive readability
with zipfile.ZipFile(zip_path, 'r') as zf:
    test_result = zf.testzip()
    assert test_result is None, f'Zip corruption in: {test_result}'
    namelist = zf.namelist()

has_h5 = 'module1_face_recognition/database/face_embeddings.h5' in namelist
has_yolo = 'yolov8n.pt' in namelist
has_main = 'module8_dashboard/backend/main.py' in namelist

print(f'BACKUP_SUCCESS: Created {zip_path}')
print(f'ZIP_SHA256: {zip_hash}')
print(f'TOTAL_FILES_ARCHIVED: {len(namelist)}')
print(f'CONTAINS_FACE_EMBEDDINGS: {has_h5}')
print(f'CONTAINS_YOLOV8: {has_yolo}')
print(f'CONTAINS_BACKEND_MAIN: {has_main}')
print(f'BACKUP_DIR: {backup_dir}')
