"""Headless pre-flight: webcam grab + YOLO26 track works? No windows opened."""
import cv2
from ultralytics import YOLO
from pathlib import Path

BASE = Path(__file__).parent
cap = cv2.VideoCapture(0)
ok, frame = cap.read()
print("webcam open:", ok, "frame:", None if not ok else frame.shape)
cap.release()
if not ok:
    raise SystemExit("WEBCAM NOT ACCESSIBLE - check privacy settings / cable")

model = YOLO(str(BASE / "yolo26n.pt"))
res = model.track(frame, persist=True, classes=[0], verbose=False, tracker="bytetrack.yaml")
n = 0 if res[0].boxes is None else len(res[0].boxes)
print(f"track OK, persons in test frame: {n}")
print("PRE-FLIGHT PASS - run define_zones.py next")
