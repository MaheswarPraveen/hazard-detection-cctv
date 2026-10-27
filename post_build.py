"""Post-build assembly and packaging script for Safety CCTV.
Copies models, configs, assets, tools, and docs into dist/SafetyCCTV,
and creates the final distributable ZIP archive.
"""
import os
import shutil
import subprocess
import zipfile
from pathlib import Path

BASE = Path(__file__).resolve().parent
DIST = BASE / "dist" / "SafetyCCTV"

if not DIST.exists():
    print(f"[ERROR] Build directory not found: {DIST}")
    raise SystemExit(1)

print("[INFO] Assembling SafetyCCTV distribution layout...")

# 1. Writable data folders
for d in ("logs", "reports", "evidence", "dataset"):
    (DIST / d).mkdir(parents=True, exist_ok=True)

# 2. Models
(DIST / "models").mkdir(parents=True, exist_ok=True)
models = ["ppe_v8n.pt", "ppe_ft.pt", "yolo26n.onnx", "hand_landmarker.task", "tracker_ppe.yaml"]
for m in models:
    src = BASE / "models" / m
    if not src.exists():
        src = BASE / m
    if src.exists():
        shutil.copy2(src, DIST / "models" / m)
        print(f"  + Model: {m}")

# 3. Config
(DIST / "config").mkdir(parents=True, exist_ok=True)
configs = ["settings.json", "zones.json", "glove_hsv.json", "alerts.example.json"]
for c in configs:
    src = BASE / "config" / c
    if not src.exists():
        src = BASE / c
    if src.exists():
        shutil.copy2(src, DIST / "config" / c)
        print(f"  + Config: {c}")

# Ensure settings.json has default if missing
settings_path = DIST / "config" / "settings.json"
if not settings_path.exists():
    import json
    settings_path.write_text(json.dumps({"camera": "1", "port": 8000}, indent=2), encoding="utf-8")
    print("  + Created default config/settings.json")

# 4. Assets
(DIST / "assets").mkdir(parents=True, exist_ok=True)
assets = ["console.html", "alarm.wav"]
for a in assets:
    src = BASE / "assets" / a
    if not src.exists():
        src = BASE / a
    if src.exists():
        shutil.copy2(src, DIST / "assets" / a)
        print(f"  + Asset: {a}")

# 5. Tools
(DIST / "tools").mkdir(parents=True, exist_ok=True)
tools = [
    "DefineZones.exe", "CalibrateGlove.exe", "CollectData.exe",
    "DefineZones.bat", "CalibrateGlove.bat", "CollectData.bat"
]
for t in tools:
    src = BASE / "tools" / t
    if src.exists():
        shutil.copy2(src, DIST / "tools" / t)
        print(f"  + Tool: {t}")

# Try to create a junction in tools/_internal to point to ../_internal
junction_target = DIST / "tools" / "_internal"
if not junction_target.exists():
    try:
        subprocess.run(
            ["cmd", "/c", "mklink", "/J", str(junction_target), str(DIST / "_internal")],
            capture_output=True, timeout=5
        )
        print("  + Created tools/_internal junction")
    except Exception:
        pass

# 6. Documentation
docs = ["README.txt", "SafetyCCTV_Client_Guide.pdf"]
for doc in docs:
    src = BASE / doc
    if src.exists():
        shutil.copy2(src, DIST / doc)
        print(f"  + Document: {doc}")

# 7. Create ZIP archive
zip_path = BASE / "dist" / "SafetyCCTV_win64.zip"
print(f"[INFO] Compressing release package into {zip_path.name}...")
if zip_path.exists():
    try:
        zip_path.unlink()
    except OSError:
        pass

with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
    for root, dirs, files in os.walk(DIST):
        # Do not recurse into junction tools/_internal when zipping to avoid duplicating files
        if Path(root) == junction_target or junction_target in Path(root).parents:
            continue
        for file in files:
            full_path = Path(root) / file
            rel_path = full_path.relative_to(BASE / "dist")
            zf.write(full_path, str(rel_path))

zip_size_mb = zip_path.stat().st_size / (1024 * 1024)
print(f"[OK] Release package created: {zip_path} ({zip_size_mb:.1f} MB)")
