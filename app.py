"
Industrial Hazard Identification & Safety AI Demo
-------------------------------------------------
Designed for quick client demonstrations using any standard Webcam or CCTV feed.

Features:
1. Real-time Worker Detection (YOLOv8)
2. Virtual Restricted Danger Zone (Geofencing Alert)
3. Live On-Screen Safety HUD (Compliance Status & Alerts)
4. Auto-Snapshot Logger (Saves proof of hazard into 'alerts/' folder)
"

import cv2
import numpy as np
import time
import os
from datetime import datetime
from ultralytics import YOLO

# ---------------- CONFIGURATION ----------------
WEBCAM_INDEX = 0             # 0 for default webcam, or replace with 'rtsp://admin:pass@ip:port/stream'
CONF_THRESHOLD = 0.5         # Detection confidence
SNAPSHOT_COOLDOWN = 3        # Minimum seconds between saving snapshot images
OUTPUT_DIR = alerts
# -----------------------------------------------

# Create alerts folder if not exists
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# Load lightweight YOLOv8 model (auto-downloads on first run)
print([INFO] Loading YOLO model...)
model = YOLO(yolov8n.pt)  # Fast & accurate for real-time edge/webcam demos

# Initialize Webcam
cap = cv2.VideoCapture(WEBCAM_INDEX)
if not cap.isOpened():
    print(f[ERROR] Could not open video source: {WEBCAM_INDEX})
    exit()

# Set video resolution
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

last_alert_time = 0
total_incidents = 0

print(\n + =*50)
print( INDUSTRIAL SAFETY AI - LIVE DEMO STARTED)
print( Press 'Q' to exit.)
print( Press 'Z' to toggle Restricted Zone position.)
print(=*50 + \n)

zone_mode = 0

def get_danger_zone(w, h, mode):
    if mode == 0:
        # Right half dangerous machinery zone
        return np.array([
            [int(w * 0.55), int(h * 0.20)],
            [int(w * 0.95), int(h * 0.20)],
            [int(w * 0.95), int(h * 0.85)],
            [int(w * 0.55), int(h * 0.85)]
        ], np.int32)
    else:
        # Center hazardous zone
        return np.array([
            [int(w * 0.25), int(h * 0.30)],
            [int(w * 0.75), int(h * 0.30)],
            [int(w * 0.75), int(h * 0.80)],
            [int(w * 0.25), int(h * 0.80)]
        ], np.int32)

while True:
    ret, frame = cap.read()
    if not ret:
        print([WARN] Failed to grab frame.)
        break

    frame = cv2.flip(frame, 1)  # Mirror effect for natural webcam presentation
    h, w, _ = frame.shape
    danger_zone = get_danger_zone(w, h, zone_mode)

    # 1. Run YOLO Object Detection (Class 0 = person)
    results = model(frame, stream=True, verbose=False)

    person_detected = 0
    hazard_detected = False
    hazard_reason = "

 # Semi-transparent overlay for Restricted Danger Zone
 overlay = frame.copy()
 cv2.fillPoly(overlay, [danger_zone], (0, 0, 180))
 cv2.polylines(frame, [danger_zone], isClosed=True, color=(0, 0, 255), thickness=3)

 # Blend overlay
 cv2.addWeighted(overlay, 0.25, frame, 0.75, 0, frame)

 # Label the Restricted Zone
 zone_center_x = int((danger_zone[0][0] + danger_zone[1][0]) / 2) - 110
 zone_center_y = int(danger_zone[0][1]) + 30
 cv2.putText(frame, RESTRICTED DANGER ZONE, (zone_center_x, zone_center_y),
 cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

 # 2. Process Detected Workers
 for r in results:
 boxes = r.boxes
 for box in boxes:
 cls = int(box.cls[0])
 conf = float(box.conf[0])

 # Person detection (YOLO class 0)
 if cls == 0 and conf >= CONF_THRESHOLD:
 person_detected += 1
 x1, y1, x2, y2 = map(int, box.xyxy[0])

 # Calculate feet coordinate (base of bounding box)
 feet_x = int((x1 + x2) / 2)
 feet_y = int(y2)

 # Check if worker stepped into Restricted Zone
 is_in_zone = cv2.pointPolygonTest(danger_zone, (feet_x, feet_y), False) >= 0

 if is_in_zone:
 hazard_detected = True
 hazard_reason = UNAUTHORIZED ACCESS (RESTRICTED ZONE)
 box_color = (0, 0, 255) # Red alert
 label = fHAZARD: Intrusion! ({conf*100:.0f}%)
 else:
 box_color = (0, 255, 0) # Green safe
 label = fWorker Verified ({conf*100:.0f}%)

 # Draw Person Bounding Box & Label
 cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
 cv2.rectangle(frame, (x1, y1 - 25), (x1 + len(label)*11, y1), box_color, -1)
 cv2.putText(frame, label, (x1 + 5, y1 - 7),
 cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
 cv2.circle(frame, (feet_x, feet_y), 6, box_color, -1)

 # 3. Handle Active Hazards & Auto-Snapshot Logging
 current_time = time.time()
 if hazard_detected:
 # Flashing Top Warning Header
 cv2.rectangle(frame, (0, 0), (w, 60), (0, 0, 220), -1)
 cv2.putText(frame, fCRITICAL ALERT: {hazard_reason}, (30, 40),
 cv2.FONT_HERSHEY_DUPLEX, 0.85, (255, 255, 255), 2)

 # Save Snapshot Evidence automatically with cooldown
 if current_time - last_alert_time > SNAPSHOT_COOLDOWN:
 last_alert_time = current_time
 total_incidents += 1
 timestamp_str = datetime.now().strftime(%Y%m%d_%H%M%S)
 snapshot_filename = os.path.join(OUTPUT_DIR, fincident_{timestamp_str}.jpg)
 cv2.imwrite(snapshot_filename, frame)
 print(f[ALERT LOGGED] Saved violation photo: {snapshot_filename})
 else:
 # Safe Banner
 cv2.rectangle(frame, (0, 0), (w, 40), (40, 40, 40), -1)
 cv2.putText(frame, STATUS: WORKPLACE MONITORED - ALL CLEAR, (30, 27),
 cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 120), 2)

 # 4. Client Dashboard HUD (Bottom Info Bar)
 cv2.rectangle(frame, (0, h - 45), (w, h), (20, 20, 20), -1)
 hud_text = fWorkers Detected: {person_detected}  |  Logged Incidents: {total_incidents}  |  Zone: #{zone_mode + 1}  |  [Z: Toggle Zone | Q: Exit]
 cv2.putText(frame, hud_text, (20, h - 15),
 cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)

 cv2.imshow(AI CCTV Safety Monitor - Live Client Demo, frame)

 key = cv2.waitKey(1) & 0xFF
 if key == ord('q') or key == 27:
 break
 elif key == ord('z'):
 zone_mode = 1 - zone_mode

cap.release()
cv2.destroyAllWindows()
print(f\n[DONE] Demo finished. Total incidents logged in '{OUTPUT_DIR}/': {total_incidents})
