"""Calibrate YOUR glove color in 10 seconds (no training needed).
Run:  python calibrate_glove.py --source 1
  1. hold the BLUE glove so it fills the center box (2m, normal gate light)
  2. press S -> median HSV inside the box is saved to glove_hsv.json
  3. masked preview pops up: white = what counts as glove. Re-pose + S again
     until the glove is solid white and the background mostly black.
  4. Q quits. The Vest+Gloves station picks up glove_hsv.json automatically
     on next start and blue becomes real YES evidence (hand zones only).
TIP: calibrate in the SAME light as the gate. Recalibrate if light changes.
"""
import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np

BASE = Path(__file__).parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="1")
    args = ap.parse_args()
    src = int(args.source) if str(args.source).isdigit() else str(args.source)
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        print(f"[X] Cannot open source {args.source}")
        return
    print("[OK] Fill the center box with the glove, press S to sample, Q to quit.")

    while True:
        ok, frame = cap.read()
        if not ok:
            time.sleep(0.05)
            continue
        h, w = frame.shape[:2]
        bx1, by1, bx2, by2 = w // 2 - 110, h // 2 - 110, w // 2 + 110, h // 2 + 110
        cv2.rectangle(frame, (bx1, by1), (bx2, by2), (0, 255, 0), 2)
        cv2.putText(frame, "glove in box, press S", (bx1, by1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.imshow("Calibrate glove color - S sample, Q quit", frame)
        key = cv2.waitKey(30) & 0xFF
        if key in (ord("q"), ord("Q")):
            break
        if key in (ord("s"), ord("S")):
            crop = frame[by1:by2, bx1:bx2]
            hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
            flat = hsv.reshape(-1, 3)
            vivid = flat[flat[:, 1] > 60]  # ignore dull background pixels
            if len(vivid) < 500:
                print("[!] too dull - get closer / add light, try again")
                continue
            med = np.median(vivid, axis=0)
            spec = {"h": float(med[0]), "s": float(med[1]), "v": float(med[2])}
            (BASE / "glove_hsv.json").write_text(json.dumps(spec))
            lo = np.array([max(0, med[0] - 10), max(0, med[1] - 60), max(0, med[2] - 60)],
                          dtype=np.uint8)
            hi = np.array([min(179, med[0] + 10), 255, 255], dtype=np.uint8)
            prev = cv2.inRange(cv2.cvtColor(frame, cv2.COLOR_BGR2HSV), lo, hi)
            cv2.imshow("Preview: white = counts as glove (any key continues)", prev)
            cv2.waitKey(0)
            print(f"[OK] saved glove_hsv.json h={med[0]:.0f} s={med[1]:.0f} v={med[2]:.0f} "
                  f"(restart the Vest+Gloves station to use it)")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
