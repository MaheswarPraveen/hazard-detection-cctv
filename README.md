# Industrial Hazard Identification & Safety AI

Real-time workplace safety computer vision pipeline designed for edge monitoring on CCTV and industrial camera feeds. Detects worker positioning relative to designated danger zones, tracks compliance, and records timestamped incident evidence.

---

## Overview

Industrial environments require strict boundary enforcement around heavy machinery, robot workcells, and high-voltage zones. This system processes live camera streams, identifies personnel using YOLOv8, computes polygon containment within geofenced danger zones, and generates automated alerts with visual evidence snapshots.

---

## Key Features

- **Personnel Detection**: Real-time worker tracking using YOLOv8 running on edge hardware or CPU.
- **Dynamic Geofencing**: Configurable danger zone boundaries with polygon containment geometry.
- **Safety HUD**: On-screen status display indicating normal operation, warnings, and active breach events.
- **Incident Snapshot Logging**: Automatic frame capture with cooldown timers to prevent disk saturation.
- **Flexible Video Sources**: Compatible with local USB webcams, RTSP IP camera streams, and pre-recorded video feeds.

---

## Architecture & Data Flow

```
[ CCTV / RTSP Stream ]
         │
         ▼
[ OpenCV Frame Capture ]
         │
         ▼
[ YOLOv8 Inference ] ───> [ Bounding Box & Centroid Calculation ]
                                      │
                                      ▼
                      [ Geofence Polygon Collision Check ]
                                      │
                         ┌────────────┴────────────┐
                         ▼                         ▼
                  [ Safe Zone ]             [ Breach Event ]
                         │                         │
                         ▼                         ▼
                  [ Normal HUD ]           [ Trigger Alarm HUD ]
                                           [ Save Evidence Snapshot ]
```

---

## Installation & Setup

### Prerequisites

- Python 3.10+
- Webcam or RTSP camera feed

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Running the Application

```bash
python app.py
```

### Controls

- `Q`: Exit application.
- `Z`: Toggle danger zone layout between machinery zone and loading perimeter.

---

## Configuration

Edit the top of `app.py` to customize parameters:

```python
WEBCAM_INDEX = 0             # 0 for webcam, or "rtsp://user:pass@ip:port/stream"
CONF_THRESHOLD = 0.5         # Minimum detection confidence
SNAPSHOT_COOLDOWN = 3        # Cooldown in seconds between snapshot writes
OUTPUT_DIR = "alerts"        # Destination directory for incident images
```

---

## License

MIT License. Authored by Maheswar N Praveen.
