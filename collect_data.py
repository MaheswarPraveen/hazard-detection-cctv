"""Dataset collector: photograph YOUR gear for fine-tuning.
Plug in the webcam and run:  python collect_data.py
Then per equipment state:
  1. press its key (1-8 below), hold the pose 2-3 m from the camera
  2. shots save AUTOMATICALLY every --every seconds (default 5)
  3. between shots, shift a little: turn head, step closer/farther,
     tilt, change light, different background corner
  4. get 20-30 shots per label, then press the next key
Keys: 1 Hardhat | 2 NO-Hardhat | 3 Mask (YOUR yellow!) | 4 NO-Mask |
      5 Safety Vest | 6 NO-Safety Vest | 7 Gloves (YOUR blue!) | 8 NO-Gloves
      Q quit (counts saved to dataset/counts.json)
Output: dataset/raw/<label>_####.jpg  (flat files, ready for Roboflow/labelImg)
Next: label 9 classes (add Person boxes too), export YOLO, train on Colab -
see FINE-TUNE NOTE at the bottom of mode_ppe.py. 15-20 min of posing total.
"""
import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import cv2

try:
    import winsound
except ImportError:
    winsound = None

BASE = Path(__file__).parent
LABELS = ["Hardhat", "NO-Hardhat", "Mask", "NO-Mask",
          "Safety Vest", "NO-Safety Vest", "Gloves", "NO-Gloves"]
KEYS = {ord(str(i + 1)): lab for i, lab in enumerate(LABELS)}  # 1..8


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="0")
    ap.add_argument("--every", type=float, default=5.0, help="seconds between auto-captures")
    ap.add_argument("--goal", type=int, default=25, help="target shots per label")
    args = ap.parse_args()

    outdir = BASE / "dataset" / "raw"
    outdir.mkdir(parents=True, exist_ok=True)
    counts_path = BASE / "dataset" / "counts.json"
    counts = {lab: len(list(outdir.glob(f"{lab}_*.jpg"))) for lab in LABELS}
    if counts_path.exists():
        try:
            counts.update(json.loads(counts_path.read_text()))
        except Exception:
            pass

    src = int(args.source) if str(args.source).isdigit() else str(args.source)
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        print(f"[X] Cannot open source {args.source}")
        return
    print("[OK] Collector running: keys 1-8 select gear, auto-capture "
          f"every {args.every:.0f}s, goal {args.goal}/label. Q quits.")

    label, seq = LABELS[0], 0
    for f in sorted(outdir.glob(f"{label}_*.jpg")):  # resume numbering
        try:
            seq = max(seq, int(f.stem.rsplit("_", 1)[1]))
        except ValueError:
            pass
    last_shot = time.time() + 2.0  # 2 s grace to pose before first capture

    while True:
        ok, frame = cap.read()
        if not ok:
            time.sleep(0.05)
            continue
        now = time.time()
        due_in = max(0.0, args.every - (now - last_shot))

        if due_in <= 0:  # capture!
            seq += 1
            ts = datetime.now().strftime("%H%M%S")
            name = f"{label}_{seq:04d}_{ts}.jpg"
            cv2.imwrite(str(outdir / name), frame)
            counts[label] = counts.get(label, 0) + 1
            counts_path.write_text(json.dumps(counts, indent=1))
            print(f"[SHOT] {name} ({counts[label]}/{args.goal} {label})", flush=True)
            if winsound is not None:
                try:
                    winsound.Beep(880, 150)
                except Exception:
                    pass
            last_shot = now
            due_in = args.every

        # overlay: current label, progress, countdown bar
        h, w = frame.shape[:2]
        done = counts.get(label, 0)
        cv2.rectangle(frame, (0, 0), (w, 86), (20, 20, 20), -1)
        cv2.putText(frame, f"{label}  {done}/{args.goal}", (12, 32),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0) if done >= args.goal else (255, 255, 255), 2)
        bar_w = int((w - 24) * (1 - due_in / args.every))
        cv2.rectangle(frame, (12, 48), (w - 12, 64), (60, 60, 60), -1)
        cv2.rectangle(frame, (12, 48), (12 + bar_w, 64), (0, 200, 0), -1)
        cv2.putText(frame, "1-8 gear | Q quit", (12, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (170, 170, 170), 1)
        cv2.imshow("Collect PPE dataset - pose, wait for beep", frame)
        key = cv2.waitKey(30) & 0xFF
        if key in (ord("q"), ord("Q")):
            break
        if key in KEYS and KEYS[key] != label:
            label = KEYS[key]
            seq = 0
            for f in sorted(outdir.glob(f"{label}_*.jpg")):
                try:
                    seq = max(seq, int(f.stem.rsplit("_", 1)[1]))
                except ValueError:
                    pass
            last_shot = time.time() + 2.0  # grace to pose after switching
            print(f"[LABEL] now capturing: {label} "
                  f"({counts.get(label, 0)}/{args.goal} so far)", flush=True)

    cap.release()
    cv2.destroyAllWindows()
    total = sum(counts.values())
    print(f"[OK] Done: {total} shots -> {outdir}")
    for lab in LABELS:
        print(f"    {lab}: {counts.get(lab, 0)}/{args.goal}")


if __name__ == "__main__":
    main()
