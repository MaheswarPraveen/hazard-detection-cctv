"""Dataset collector: photograph YOUR gear for fine-tuning.
Plug in the webcam and run:  python collect_data.py
Then per item:
  1. press its key (1-3 below), pose 2-3 m from the camera
  2. press SPACE to take a photo (beep = saved)
  3. between shots, shift a little: turn, step closer/farther,
     change light, different background corner
  4. get --goal shots per item, then press the next key
Keys: 1 Vest | 2 Helmet | 3 Gloves | SPACE capture | C switch camera | S save & quit
Output: dataset/raw/<label>_####.jpg  (flat files, ready for Roboflow/labelImg)
Next: label boxes in Roboflow (add Person boxes too), export YOLO, train on Kaggle.
"""
import argparse
import json
from datetime import datetime
from pathlib import Path

import cv2

try:
    import winsound
except ImportError:
    winsound = None

from paths import APP_DIR, DATASET_DIR, get_settings

BASE = APP_DIR
LABELS = ["Safety Vest", "Hardhat", "Gloves"]
KEYS = {ord(str(i + 1)): lab for i, lab in enumerate(LABELS)}  # 1..3


def next_seq(outdir, label):
    seq = 0
    for f in outdir.glob(f"{label}_*.jpg"):  # resume numbering
        try:
            seq = max(seq, int(f.stem.split("_")[1]))
        except (ValueError, IndexError):
            pass
    return seq


def main():
    default_src = str(get_settings().get("camera", "1"))
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default=default_src, help="1 = USB cam, 0 = built-in")
    ap.add_argument("--goal", type=int, default=12, help="target shots per item")
    args = ap.parse_args()

    outdir = DATASET_DIR / "raw"
    outdir.mkdir(parents=True, exist_ok=True)
    counts_path = DATASET_DIR / "counts.json"
    counts = {lab: len(list(outdir.glob(f"{lab}_*.jpg"))) for lab in LABELS}

    src = int(args.source) if str(args.source).isdigit() else str(args.source)
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        print(f"[X] Cannot open camera {args.source} (stop the dashboard station first)")
        return
    print(f"[OK] Collector running: 1-3 pick item, SPACE capture, S save & quit. "
          f"Goal {args.goal}/item.", flush=True)

    label = LABELS[0]
    seq = next_seq(outdir, label)
    flash = 0  # frames of white border after a shot

    while True:
        ok, frame = cap.read()
        if not ok:
            cv2.waitKey(50)
            continue
        clean = frame.copy()  # saved photo has no overlay

        # overlay: current item, progress, key help
        h, w = frame.shape[:2]
        done = counts.get(label, 0)
        cv2.rectangle(frame, (0, 0), (w, 70), (20, 20, 20), -1)
        cv2.putText(frame, f"{label}  {done}/{args.goal}", (12, 32),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0) if done >= args.goal else (255, 255, 255), 2)
        summary = "  ".join(f"{i + 1}:{lab.split()[-1]} {counts.get(lab, 0)}" for i, lab in enumerate(LABELS))
        cv2.putText(frame, f"{summary}  |  SPACE shot  C cam {src}  S save+quit", (12, 58),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (170, 170, 170), 1)
        if flash:
            cv2.rectangle(frame, (0, 0), (w - 1, h - 1), (255, 255, 255), 8)
            flash -= 1
        cv2.imshow("Collect PPE dataset - SPACE capture, S save+quit", frame)

        key = cv2.waitKey(30) & 0xFF
        if key in (ord("s"), ord("S")):
            break
        if key == ord(" "):
            seq += 1
            name = f"{label}_{seq:04d}_{datetime.now():%H%M%S}.jpg"
            cv2.imwrite(str(outdir / name), clean)
            counts[label] = counts.get(label, 0) + 1
            print(f"[SHOT] {name} ({counts[label]}/{args.goal} {label})", flush=True)
            if winsound is not None:
                try:
                    winsound.Beep(880, 120)
                except Exception:
                    pass
            flash = 4
        if key in (ord("c"), ord("C")):  # flip USB cam <-> laptop cam
            cams = [src, 1 - src] if isinstance(src, int) and src in (0, 1) else [src]
            cap.release()
            cap = cv2.VideoCapture(cams[-1])
            if cap.isOpened():
                src = cams[-1]
            else:  # other cam missing: fall back
                cap = cv2.VideoCapture(src)
            print(f"[CAM] now camera {src} ({'USB' if src == 1 else 'laptop'})", flush=True)
        if key in KEYS and KEYS[key] != label:
            label = KEYS[key]
            seq = next_seq(outdir, label)
            print(f"[ITEM] now capturing: {label} ({counts.get(label, 0)}/{args.goal} so far)", flush=True)

    cap.release()
    cv2.destroyAllWindows()
    counts_path.write_text(json.dumps(counts, indent=1))
    print(f"[OK] Saved: {sum(counts.values())} shots -> {outdir}")
    for lab in LABELS:
        print(f"    {lab}: {counts.get(lab, 0)}/{args.goal}")


if __name__ == "__main__":
    main()
