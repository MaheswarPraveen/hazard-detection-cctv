# Prompt for Antigravity: package Safety CCTV as a runnable Windows app

Paste everything below the line into Antigravity, opened on this folder.

---

## Goal
Package this project (`hazard-detection-cctv`) as ONE double-click Windows app,
`SafetyCCTV.exe`. It launches the web dashboard and opens the browser at
http://localhost:8000. All three stations must work from the dashboard:
- **Zone Alert**: restricted-area intrusion (`mode_zone.py` + `zones.json`)
- **PPE Mask + Helmet** (`mode_ppe.py --checks helmet,mask`)
- **PPE Helmet + Vest + Gloves** (`mode_ppe.py --checks helmet,vest,gloves --gloves-model ppe_ft.pt`)

Target: Windows 10/11, CPU only, no Python installed on the target PC.

## Rules
- Do NOT change detection logic, thresholds or model files. Packaging only.
- Plan first, show me the plan, wait for my OK.
- Keep `python dashboard.py` working from source as before.

## Output folder layout (PyInstaller one-folder build, not one-file)
```
SafetyCCTV/
  SafetyCCTV.exe            <- launcher (dashboard + opens browser)
  _internal/                <- PyInstaller runtime (python, cv2, torch, ultralytics, mediapipe)
  models/
    ppe_v8n.pt  ppe_ft.pt  yolo26n.onnx  hand_landmarker.task  tracker_ppe.yaml
  config/
    zones.json  glove_hsv.json  alerts.example.json   (alerts.json created by user)
  assets/
    console.html  alarm.wav
  tools/
    DefineZones.exe  CalibrateGlove.exe  CollectData.exe
  logs/  reports/  evidence/  dataset/      <- created at first run, writable
  README.txt
```

## Steps
1. **One path helper**: add `paths.py` with `APP_DIR` = folder of the exe when frozen
   (`sys.frozen` → `Path(sys.executable).parent`), else the script folder. Add
   `models/`, `config/`, `assets/`, and writable `logs/ reports/ evidence/ dataset/`.
   Replace every `BASE / "<file>"` in dashboard.py, mode_ppe.py, mode_zone.py, alerts.py,
   report.py, define_zones.py, calibrate_glove.py and collect_data.py with the right subfolder.
   Model lookups fall back to `models/`.
2. **Stations inside one exe**: the dashboard currently runs
   `[sys.executable, "mode_ppe.py", ...]`. When frozen, `sys.executable` is the exe itself, so
   add a dispatcher in the launcher: `SafetyCCTV.exe --run ppe <args>` calls `mode_ppe.main()`,
   `--run zone` calls `mode_zone.main()`, `--run report` calls `report.main()`. The dashboard
   spawns `[sys.executable, "--run", "ppe", ...]` when frozen and keeps the current command
   from source.
3. **Launcher** `launcher.py`: run the dispatcher if `--run` is given; otherwise start the
   dashboard, then `webbrowser.open("http://localhost:8000")` after the port answers.
   Single instance: if port 8000 is already serving, just open the browser and exit.
4. **Camera setting**: move `CAM_SOURCE` out of dashboard.py into `config/settings.json`
   (`{"camera": "1", "port": 8000}`), created on first run if missing.
5. **PyInstaller spec** `SafetyCCTV.spec` (one-folder, `console=False` for the launcher):
   - hiddenimports / `collect_all` for `ultralytics`, `mediapipe`, `cv2`, `supervision`, `lap`
   - datas: mediapipe's model files, ultralytics' `cfg/` (needed for bytetrack.yaml)
   - extra EXEs for the tools (DefineZones, CalibrateGlove, CollectData), sharing `_internal`
   - copy `models/ config/ assets/` next to the exe after build
   - use CPU-only torch to keep the size down:
     `pip install torch --index-url https://download.pytorch.org/whl/cpu`
6. **build.bat**: makes a venv, installs `requirements.txt` + `pyinstaller`, runs the spec,
   then zips `dist/SafetyCCTV` → `SafetyCCTV_win64.zip`.
7. **README.txt** for the end user: unzip → run `tools/DefineZones.exe` once to draw the
   restricted area → double-click `SafetyCCTV.exe` → Start a station on the dashboard. For
   WhatsApp/email alerts, copy `config/alerts.example.json` to `alerts.json` and fill it in.

## Must test before calling it done (on the built exe, not from source)
- [ ] `SafetyCCTV.exe` opens the dashboard in the browser
- [ ] Zone Alert starts, draws zones, saves evidence to `evidence/<date>/`, siren plays
- [ ] Both PPE stations start; the gloves model loads (`gloves backup live` in the log)
- [ ] Starting one station stops the other (shared camera)
- [ ] Excel download works for zone and PPE
- [ ] `/files/` serves only logs/reports/evidence (403 for `config/alerts.json`)
- [ ] Runs from a folder path with spaces, on a PC without Python
- [ ] `python dashboard.py` from source still works

## Don't
- Don't use `--onefile`: torch + mediapipe unpack slowly at every start, and the models need a writable folder.
- Don't bundle `alerts.json`, `dataset/`, `logs/` or the unused spare models (`ppe_v8m.pt`, `yolo26s.pt`, `yolo11n.pt`, `yolov8n.pt`, `sv_v8n.pt`).
