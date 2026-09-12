"""Benchmark yolo26n.pt vs yolo26n.onnx at 480px on this CPU."""
import time
import numpy as np
from ultralytics import YOLO

img = np.zeros((480, 480, 3), dtype=np.uint8)

for w in ["yolo26n.pt", "yolo26n.onnx"]:
    m = YOLO(w)
    m(img, verbose=False)  # warmup
    t = time.time()
    for _ in range(10):
        m(img, verbose=False)
    print(f"{w} avg_ms={round((time.time() - t) / 10 * 1000, 1)}", flush=True)
print("BENCH DONE", flush=True)
