"""Draw danger polygons once, save to zones.json.
Run: python define_zones.py --source 0
Click points with mouse. S = save zone, R = reset current, Q = quit.
"""
import argparse
import json
from pathlib import Path
import cv2

BASE = Path(__file__).parent
ZONES_FILE = BASE / "zones.json"

points = []
zones = []
frame_copy = None


def mouse_cb(event, x, y, flags, param):
    global points, frame_copy
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append([x, y])


def main():
    global points, frame_copy
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="0")
    args = ap.parse_args()
    src = int(args.source) if str(args.source).isdigit() else str(args.source)

    if ZONES_FILE.exists():
        try:
            zones.extend(json.loads(ZONES_FILE.read_text()))
            print(f"Loaded {len(zones)} existing zones.")
        except Exception:
            pass

    cap = cv2.VideoCapture(src)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        print(f"[X] Cannot open {args.source}")
        return
    frame_copy = frame.copy()

    cv2.namedWindow("Draw zones")
    cv2.setMouseCallback("Draw zones", mouse_cb)
    print("Click polygon points. S=save zone, R=reset, Q=quit+save.")
    while True:
        vis = frame_copy.copy()
        for z in zones:
            import numpy as np
            cv2.polylines(vis, [np.array(z["points"], "int32")], True, (255, 165, 0), 2)
        for i, p in enumerate(points):
            cv2.circle(vis, tuple(p), 5, (0, 0, 255), -1)
            if i > 0:
                cv2.line(vis, tuple(points[i - 1]), tuple(p), (0, 0, 255), 2)
        cv2.imshow("Draw zones", vis)
        k = cv2.waitKey(30) & 0xFF
        if k in (ord("s"), ord("S")) and len(points) >= 3:
            zones.append({"name": f"DANGER-{len(zones)+1}", "points": points})
            print(f"Saved zone {len(zones)} ({len(points)} pts)")
            points = []
        elif k in (ord("r"), ord("R")):
            points = []
        elif k in (ord("q"), ord("Q")):
            break
    ZONES_FILE.write_text(json.dumps(zones, indent=2))
    print(f"[OK] {len(zones)} zones -> {ZONES_FILE}")
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
