"""MODE 1: PPE_Check - helmet + vest live warnings + visitor count.
Run: python mode_ppe.py --source 0
Live screen only: green OK / red WARNING per person. No evidence files.
Needs ppe_v8m.pt next to this script (included).
"""
import argparse
import math
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

try:
    import winsound  # Windows-only alarm playback
except ImportError:
    winsound = None

BASE = Path(__file__).parent
ALARM_WAV = BASE / "alarm.wav"

# PPE class ids are resolved dynamically after model load (supports ppe_v8n and ppe_v8m with gloves)


def center_in(box, person, margin=0.05):
    """Is the center of `box` inside `person` box (with small margin)?"""
    px1, py1, px2, py2 = person
    pw, ph = px2 - px1, py2 - py1
    cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    return (px1 - pw * margin <= cx <= px2 + pw * margin
            and py1 - ph * margin <= cy <= py2 + ph * margin)


def save_tally(path, visitors, ok_visitors, tally):
    """Persist counts-only tally (no photos). Called on every new violation event."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        f.write("metric,count\n")
        f.write(f"visitors,{len(visitors)}\n")
        f.write(f"ppe_ok,{len(ok_visitors)}\n")
        for k, v in tally.items():
            f.write(f"{k},{v}\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="0", help="0 for webcam or RTSP URL")
    ap.add_argument("--model", default="ppe_v8n.pt", help="PPE weights (nano, CPU-friendly)")
    ap.add_argument("--camera", default="CAM01")
    ap.add_argument("--imgsz", type=int, default=480)
    ap.add_argument("--conf", type=float, default=0.4, help="confidence threshold")
    args = ap.parse_args()

    src = int(args.source) if str(args.source).isdigit() else str(args.source)
    wpath = BASE / args.model if not Path(args.model).exists() else Path(args.model)
    model = YOLO(str(wpath))
    # resolve class ids case-insensitively from model.names
    lower_names = {i: str(n).lower() for i, n in model.names.items()}
    def _find(cands):
        for cand in cands:
            cand_l = cand.lower()
            for idx, nm in lower_names.items():
                if nm == cand_l or nm.replace("-", " ").replace("_", " ") == cand_l:
                    return idx
        return None
    HARDHAT = _find(["hardhat", "hard hat"])
    NO_HARDHAT = _find(["no-hardhat", "no hardhat", "no-hard hat", "no- hardhat"])
    VEST = _find(["safety vest", "safety-vest", "vest"])
    NO_VEST = _find(["no-safety vest", "no-safety-vest", "no vest", "no-vest"])
    PERSON = _find(["person"])
    GLOVES = _find(["gloves", "glove", "safety gloves"])
    NO_GLOVES = _find(["no-gloves", "no gloves", "no glove", "no-glove"])
    HAS_GLOVES = GLOVES is not None or NO_GLOVES is not None
    print(f"[INFO] Model {wpath.name} classes: {model.names}")
    print(f"[INFO] Mapped ids — HARDHAT={HARDHAT} NO_HARDHAT={NO_HARDHAT} VEST={VEST} NO_VEST={NO_VEST} PERSON={PERSON} GLOVES={GLOVES} NO_GLOVES={NO_GLOVES} gloves_enabled={HAS_GLOVES}")

    print("[...] Warming up model (3 dummy frames)...")
    for _ in range(3):
        model(np.zeros((480, 480, 3), dtype=np.uint8), verbose=False)
    print("[OK] Model hot.")

    cap = None
    for attempt in range(1, 4):
        cap = cv2.VideoCapture(src)
        if cap.isOpened():
            break
        print(f"[!] Camera open failed (attempt {attempt}/3), retrying in 3s...")
        time.sleep(3)
    if not cap.isOpened():
        print(f"[X] Cannot open source {args.source} - check camera cable / RTSP URL")
        return
    print(f"[OK] Camera opened: {args.source}")

    today = datetime.now().strftime("%Y-%m-%d")
    visitors = set()  # unique track ids seen (people walked through)
    ok_visitors = set()  # ids seen fully compliant at least once
    warn_state = {}  # track_id -> last warned violation (rate-limit console spam)
    tally = {"no_helmet": 0, "no_vest": 0, "no_gloves": 0}  # violation events today
    if not HAS_GLOVES:
        tally.pop("no_gloves", None)
    tally_path = BASE / "logs" / f"ppe_stats_{today}.csv"
    if tally_path.exists():  # resume today's counts across restarts
        for line in tally_path.read_text().splitlines()[1:]:
            k, v = line.split(",")
            if k in tally:
                tally[k] = int(v)
    alarm_on = False
    last_siren = 0.0
    fps_t0 = time.time()
    fps_n = 0
    fps = 0.0

    print(f"[OK] PPE_Check running on {args.source}. Press Q to quit.")
    for r in model.track(source=src, stream=True, persist=True, imgsz=args.imgsz,
                         conf=args.conf, verbose=False, vid_stride=2, tracker="bytetrack.yaml"):
        frame = r.orig_img
        h, w = frame.shape[:2]

        persons, hats, nohats, vests, novests, gloves, nogloves = [], [], [], [], [], [], []
        if r.boxes is not None and r.boxes.id is not None:
            ids = r.boxes.id.cpu().numpy().astype(int)
            xyxy = r.boxes.xyxy.cpu().numpy()
            cls = r.boxes.cls.cpu().numpy().astype(int)
            for box, c, tid in zip(xyxy, cls, ids):
                if PERSON is not None and c == PERSON:
                    persons.append((box, tid))
                elif HARDHAT is not None and c == HARDHAT:
                    hats.append(box)
                elif NO_HARDHAT is not None and c == NO_HARDHAT:
                    nohats.append(box)
                elif VEST is not None and c == VEST:
                    vests.append(box)
                elif NO_VEST is not None and c == NO_VEST:
                    novests.append(box)
                elif GLOVES is not None and c == GLOVES:
                    gloves.append(box)
                elif NO_GLOVES is not None and c == NO_GLOVES:
                    nogloves.append(box)

        violations = 0
        for pbox, tid in persons:
            x1, y1, x2, y2 = map(int, pbox)
            has_hat = any(center_in(hb, pbox) for hb in hats)
            flag_nohat = any(center_in(hb, pbox) for hb in nohats)
            has_vest = any(center_in(vb, pbox) for vb in vests)
            flag_novest = any(center_in(vb, pbox) for vb in novests)
            has_gloves = any(center_in(gb, pbox) for gb in gloves) if HAS_GLOVES else True
            flag_nogloves = any(center_in(gb, pbox) for gb in nogloves) if HAS_GLOVES else False
            bad_helmet = (not has_hat) or flag_nohat
            bad_vest = (not has_vest) or flag_novest
            bad_gloves = ((not has_gloves) or flag_nogloves) if HAS_GLOVES else False
            tags = []
            if bad_helmet:
                tags.append("NO HELMET")
            if bad_vest:
                tags.append("NO VEST")
            if bad_gloves:
                tags.append("NO GLOVES")
            ok = not tags
            color = (0, 255, 0) if ok else (0, 0, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, "OK" if ok else "WARNING " + "+".join(tags), (x1, y1 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            visitors.add(int(tid))
            if not ok:
                violations += 1
                vtype = "+".join(tags)
                if warn_state.get(int(tid)) != vtype:  # count + warn once per change
                    warn_state[int(tid)] = vtype
                    if bad_helmet:
                        tally["no_helmet"] += 1
                    if bad_vest:
                        tally["no_vest"] += 1
                    if HAS_GLOVES and bad_gloves:
                        tally["no_gloves"] += 1
                    save_tally(tally_path, visitors, ok_visitors, tally)
                    print(f"[WARNING] ID:{tid} {vtype}")
            else:
                ok_visitors.add(int(tid))
                warn_state.pop(int(tid), None)

        now = time.time()
        fps_n += 1
        if now - fps_t0 >= 1.0:
            fps = fps_n / (now - fps_t0)
            fps_n = 0
            fps_t0 = now

        pill_color = (0, 0, 210) if violations > 0 else (0, 170, 0)
        cv2.rectangle(frame, (8, 8), (248, 38), pill_color, -1)
        cv2.circle(frame, (26, 23), 6, (255, 255, 255), -1)
        cv2.putText(frame, "WARNING" if violations > 0 else "ALL COMPLIANT", (40, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, f"{len(persons)} in view | {len(visitors)} visitors | {fps:.0f} FPS",
                    (10, 54), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (230, 230, 230), 1)
        gloves_txt = f" | No gloves: {tally.get('no_gloves', 0)}" if HAS_GLOVES else ""
        cv2.putText(frame, f"No helmet: {tally['no_helmet']} | No vest: {tally['no_vest']}{gloves_txt} | OK: {len(ok_visitors)}",
                    (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        if violations > 0:
            if winsound is not None and ALARM_WAV.exists():
                if (not alarm_on) or (now - last_siren > 4.0):
                    winsound.PlaySound(str(ALARM_WAV),
                                       winsound.SND_FILENAME | winsound.SND_ASYNC | winsound.SND_LOOP)
                    if not alarm_on:
                        print("[ALARM] siren ON")
                    last_siren = now
                    alarm_on = True
        else:
            if alarm_on and winsound is not None:
                winsound.PlaySound(None, winsound.SND_PURGE)
                print("[ALARM] siren OFF")
                alarm_on = False

        cv2.imshow("PPE_Check - Q to quit", frame)
        if cv2.waitKey(1) & 0xFF in (ord("q"), ord("Q")):
            break

    cap.release()
    cv2.destroyAllWindows()
    save_tally(tally_path, visitors, ok_visitors, tally)
    gloves_s = f" no_gloves={tally.get('no_gloves', 0)}" if HAS_GLOVES else ""
    print(f"[OK] PPE session ended | visitors={len(visitors)} ok={len(ok_visitors)} "
          f"no_helmet={tally['no_helmet']} no_vest={tally['no_vest']}{gloves_s}")


if __name__ == "__main__":
    main()
