"""Benchmark nano models on this CPU + ensure weights downloaded."""
import time
import numpy as np
from ultralytics import YOLO

img = np.zeros((640, 640, 3), dtype=np.uint8)

for w in ["yolov8n.pt", "yolo11n.pt", "yolo26n.pt"]:
    try:
        m = YOLO(w)
        m(img, verbose=False)  # warmup
        t = time.time()
        for _ in range(5):
            m(img, verbose=False)
        print(f"{w} OK avg_ms={round((time.time() - t) / 5 * 1000, 1)}", flush=True)
    except Exception as e:
        print(f"{w} FAIL {str(e)[:300]}", flush=True)
print("BENCH DONE", flush=True)
