# Industrial Hazard Detection & CCTV Safety AI

AI-powered real-time workplace safety computer vision pipeline designed for edge monitoring on CCTV and industrial camera feeds. Detects worker positioning relative to designated danger zones, verifies hardhat and safety vest PPE compliance, tracks visitor tallies, and records timestamped incident evidence.

---

## Overview

Industrial environments require strict boundary enforcement around heavy machinery, robot workcells, and high-voltage zones, as well as strict PPE adherence. This system processes live camera streams (USB webcams or RTSP network cameras), provides two distinct operational modes, and includes a built-in web monitoring console.

---

## Core Operational Modes

### Mode 1: Zone Alert (`mode_zone.py`)
- **Polygon Geofencing**: Configurable danger zone boundaries with multi-point polygon containment geometry.
- **Entry & Exit Grace**: Multi-frame debounce smoothing to eliminate false positives from fleeting motion.
- **Intrusion Evidence Capture**: Automated snapshot saving to `evidence/` with configurable cooldown intervals per tracked person.
- **Audio Alarm**: Local siren playback via Windows waveform audio (`alarm.wav`) on breach events.
- **Incident Audit Logging**: CSV event logging to `logs/incidents_YYYY-MM-DD.csv` recording camera ID, timestamp, violation type, and photo path.

### Mode 2: PPE Check (`mode_ppe.py`)
- **Compliance Tracking**: Real-time bounding box detection for hardhats, missing hardhats, safety vests, and missing safety vests.
- **Visitor Counting**: Distinguishes between compliant and non-compliant personnel entering the facility.
- **Visual HUD**: Color-coded on-screen status bounding boxes (green for compliant, red for violation).

---

## System Architecture

```
[ CCTV / RTSP / Webcam Feed ]
              │
              ▼
    [ Frame Acquisition ]
              │
      ┌───────┴─────────────────────────┐
      ▼                                 ▼
[ Mode 1: Zone Alert ]        [ Mode 2: PPE Check ]
  - YOLO26n ONNX Inference      - YOLOv8 PPE Inference
  - Multi-point Polygon Test    - Hardhat / Vest Classifiers
  - Debounce Grace Windows      - Bounding Box Intersection
  - Audio Siren Trigger         - Compliance HUD Overlay
  - Evidence Snapshot Save      - Visitor Tally Metrics
      │                                 │
      └───────┬─────────────────────────┘
              │
              ▼
[ Runtime Outputs (Auto-Generated on Demand) ]
  - evidence/ (Timestamped snapshots)
  - logs/ (Incident CSVs)
  - reports/ (Generated Excel sheets)
              │
              ▼
[ Management & Reporting ]
  - Web Console Dashboard (dashboard.py / console.html -> localhost:8000)
  - Daily Excel Incident Reporter (report.py -> reports/report_YYYY-MM-DD.xlsx)
```

---

## Key Modules & Tools

| File | Purpose |
|---|---|
| `dashboard.py` | Embedded, zero-dependency local HTTP server and web dashboard interface |
| `console.html` | Front-end console for starting/stopping modes, monitoring live metrics, and downloading logs |
| `mode_zone.py` | Restricted area intrusion detection engine with polygon geofencing |
| `mode_ppe.py` | Helmet and safety vest compliance detection engine |
| `define_zones.py` | Interactive polygon drawing utility for defining danger zones on a live video frame |
| `check_zone.py` | Zone boundary geometry verification tool |
| `discover.py` | Network camera discovery probe using SSDP (UPnP) and ONVIF WS-Discovery |
| `report.py` | Automated daily incident report generator outputting formatted Excel spreadsheets |
| `make_alarm.py` | Standalone audio synthesizer generating alert waveforms (`alarm.wav`) |
| `bench.py` | YOLO PyTorch inference performance benchmarking utility |
| `bench_onnx.py` | ONNX Runtime CPU inference benchmarking utility |

---

## Installation & Setup

### Prerequisites

- Python 3.10+
- USB Webcam or RTSP Network Camera feed

### Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Usage Guide

### 1. One-Click Launchers (Windows)

- `run_dashboard.bat`: Starts the web console on `http://localhost:8000`
- `run_zone.bat`: Launches the Zone Intrusion Alert engine
- `run_ppe.bat`: Launches the PPE Compliance engine

### 2. Defining Custom Danger Zones

To define customized danger zones on your specific camera layout:

```bash
python define_zones.py --source 0
```
- Click with the left mouse button to place polygon boundary points.
- Press `S` to save the current zone.
- Press `R` to reset points.
- Press `Q` to finalize and save boundaries to `zones.json`.

### 3. Running Zone Intrusion Alert

```bash
# Using standard webcam:
python mode_zone.py --source 0

# Using an RTSP IP camera feed:
python mode_zone.py --source "rtsp://username:password@192.168.1.100:554/stream" --camera CAM01
```

### 4. Running PPE Compliance Verification

```bash
python mode_ppe.py --source 0 --conf 0.25 --imgsz 640
# single-photo check without a camera (saves *_annotated.jpg):
python mode_ppe.py --source live_ready.jpg --imgsz 640
# gloves at a 2-3 m gate via backup weights (ppe_v8n has no glove classes):
python mode_ppe.py --source 0 --gloves-model ppe_v8m.pt
```
Detection notes: inference runs at the lowest per-class threshold (mask/glove
~0.20) with larger 640px frames, person boxes are expanded for helmet-above-head
association, warnings need 3 consecutive frames (no flicker), and mask/gloves are
only judged close-up (small/far faces read FAR, not NO MASK). Yellow cloth masks
and blue gloves outside the training colors may still need fine-tuning (see the
FINE-TUNE NOTE at the bottom of `mode_ppe.py`).

### 5. Running the Web Console Dashboard

```bash
python dashboard.py
```
Open `http://localhost:8000` in any browser to view live status, start/stop processes, view captured violation snapshots, and download audit CSVs.

### 6. Generating Daily Audit Reports

```bash
python report.py --date 2026-09-12
```
Outputs an Excel spreadsheet (`reports/report_YYYY-MM-DD.xlsx`) with aggregate summaries, hourly incident distributions, and itemized logs.

---

## Network Camera Auto-Discovery

To detect connected ONVIF and SSDP network cameras on your local network:

```bash
python discover.py
```

---

## Edge & Hardware Compatibility

- **CPU Optimization**: Pre-configured with YOLO26n ONNX weights for low-latency real-time inference on standard desktop and industrial edge CPUs without dedicated GPUs.
- **GPU Acceleration**: Fully compatible with NVIDIA CUDA via standard PyTorch and ONNX Runtime GPU providers.

---

## License

MIT License. Authored by Maheswar N Praveen.
