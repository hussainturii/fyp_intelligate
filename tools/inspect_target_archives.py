import os
import zipfile
import tarfile
from pathlib import Path

downloads = Path("C:/Users/user/Downloads")

targets = {
    "lfw_tgz": downloads / "lfw.tgz",
    "lfw_dir": downloads / "lfw",
    "dataset_a_zip": downloads / "Pakistani-Number-plates.v6i.yolov8.zip",
    "dataset_b_zip": downloads / "PAKISTANI_LICENSE_NUMBER_PLATES.v1i.yolov8.zip",
    "pairs_txt": downloads / "pairs.txt",
}

print("=== DEEP INSPECTION OF DOWNLOADED DATASETS ===")
for key, p in targets.items():
    print(f"\n==========================================")
    print(f"Target: {key} -> {p}")
    if not p.exists():
        print("DOES NOT EXIST!")
        continue
    
    if p.is_file():
        sz_mb = p.stat().st_size / (1024 * 1024)
        print(f"File Size: {p.stat().st_size:,} bytes ({sz_mb:.2f} MB)")
        
        if zipfile.is_zipfile(p):
            with zipfile.ZipFile(p, "r") as z:
                names = z.namelist()
                print(f"Archive Format: ZIP | Total Members: {len(names)}")
                # Check for directories, images, labels, yaml
                yamls = [n for n in names if n.endswith(('.yaml', '.yml'))]
                train_imgs = [n for n in names if 'train' in n and n.endswith(('.jpg', '.png', '.jpeg'))]
                val_imgs = [n for n in names if ('valid' in n or 'val' in n) and n.endswith(('.jpg', '.png', '.jpeg'))]
                test_imgs = [n for n in names if 'test' in n and n.endswith(('.jpg', '.png', '.jpeg'))]
                labels = [n for n in names if n.endswith('.txt')]
                print(f"YAML configs: {yamls}")
                print(f"Train images: {len(train_imgs)} | Val images: {len(val_imgs)} | Test images: {len(test_imgs)}")
                print(f"Total Label files: {len(labels)}")
                
                # Print data.yaml content if present
                for y in yamls:
                    print(f"\n--- Content of {y} ---")
                    try:
                        content = z.read(y).decode('utf-8', errors='ignore')
                        print(content)
                    except Exception as e:
                        print(f"Error reading {y}: {e}")
                        
                # Inspect first 3 label files to see if text exists or only class id + bbox
                txt_labels = [l for l in labels if not l.endswith('README.dataset.txt') and not l.endswith('README.roboflow.txt')]
                print(f"\nInspecting sample label files ({len(txt_labels)} found):")
                for lbl in txt_labels[:5]:
                    raw = z.read(lbl).decode('utf-8', errors='ignore').strip()
                    print(f"  {lbl} -> {raw}")
                    
        elif p.name.endswith(('.tgz', '.tar.gz', '.tar')):
            try:
                with tarfile.open(p, "r:*") as t:
                    names = t.getnames()
                    print(f"Archive Format: TAR.GZ | Total Members: {len(names)}")
                    subdirs = set(n.split('/')[1] for n in names if '/' in n)
                    print(f"Total subdirectories/identities: {len(subdirs)}")
                    print(f"Sample members: {names[:10]}")
            except Exception as e:
                print(f"Error reading tar: {e}")
                
        elif p.name == "pairs.txt":
            lines = p.read_text(encoding='utf-8', errors='ignore').splitlines()
            print(f"Format: Text file | Total lines: {len(lines)}")
            print(f"Header: {lines[0] if lines else 'EMPTY'}")
            print("First 5 lines:")
            for l in lines[:5]:
                print(f"  {l}")

    elif p.is_dir():
        subdirs = [d for d in p.iterdir() if d.is_dir()]
        images = list(p.glob("*/*.jpg")) + list(p.glob("*/*.png"))
        print(f"Directory: Found {len(subdirs)} subdirectories and {len(images)} images.")
        if subdirs:
            print(f"Sample subdirectories: {[d.name for d in subdirs[:10]]}")
