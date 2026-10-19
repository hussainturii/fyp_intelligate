import os
import zipfile
import tarfile
from pathlib import Path

downloads = Path("C:/Users/user/Downloads")
print(f"=== INSPECTING DOWNLOADS FOLDER: {downloads} ===")

if not downloads.exists():
    print("Downloads folder does not exist!")
    exit(1)

candidates = []
for p in sorted(downloads.iterdir(), key=lambda x: x.stat().st_mtime, reverse=True):
    if p.is_file():
        sz_mb = p.stat().st_size / (1024 * 1024)
        name_lower = p.name.lower()
        if any(term in name_lower for term in ["lfw", "plate", "pakistan", "vehicle", "car", "face", "pair", "yolo", "roboflow"]) or p.suffix in [".zip", ".tar", ".gz", ".tgz", ".txt"]:
            candidates.append(p)
            print(f"- {p.name} | {sz_mb:.2f} MB | Modified: {p.stat().st_mtime}")

print(f"\nDiscovered {len(candidates)} candidate files. Inspecting archive contents...")

for p in candidates:
    print(f"\n--- Archive Content Inspection: {p.name} ---")
    if zipfile.is_zipfile(p):
        try:
            with zipfile.ZipFile(p, "r") as z:
                names = z.namelist()
                print(f"Format: ZIP | Total entries: {len(names)}")
                print("First 15 entries:")
                for n in names[:15]:
                    print(f"  {n}")
        except Exception as e:
            print(f"Error reading zip: {e}")
    elif p.name.endswith((".tar.gz", ".tgz", ".tar")):
        try:
            with tarfile.open(p, "r:*") as t:
                names = t.getnames()
                print(f"Format: TAR/TGZ | Total entries: {len(names)}")
                print("First 15 entries:")
                for n in names[:15]:
                    print(f"  {n}")
        except Exception as e:
            print(f"Error reading tar: {e}")
    elif p.suffix == ".txt":
        print(f"Format: TEXT | First 3 lines:")
        try:
            lines = p.read_text(encoding="utf-8", errors="ignore").splitlines()[:3]
            for l in lines:
                print(f"  {l}")
        except Exception as e:
            print(f"Error reading text: {e}")
