"""MODE 2: Zone_Alert - restricted area intrusion detection.
Run: python mode_zone.py --source 0
     python mode_zone.py --source "rtsp://user:pass@ip:554/stream"
"""
import argparse
import csv
import math
import os
import subprocess
import time
from datetime import datetime
from pathlib import Path
import json

import cv2
import numpy as np
from ultralytics import YOLO

try:
    import winsound  # Windows-only alarm playback
except ImportError:
    winsound = None

BASE = Path(__file__).parent
LOG_DIR = BASE / "logs"
EVIDENCE_DIR = BASE / "evidence"
ZONES_FILE = BASE / "zones.json"
ALARM_WAV = BASE / "alarm.wav"
COOLDOWN_SEC = 5  # one snapshot per track_id per 5s


def load_zones():
    if not ZONES_FILE.exists():
        return []
    data = json.loads(ZONES_FILE.read_text())
    # normalize: [{"name":..., "points":[[x,y],...]}]
    return data


def log_incident(csv_path, row):
    is_new = not csv_path.exists()
    with open(csv_path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["time", "camera", "violation_type", "track_id", "in_frame_count", "photo_path"])
        if is_new:
            w.writeheader()
        w.writerow(row)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="0", help="0 for webcam or RTSP URL")
    ap.add_argument("--model", default="yolo26n.onnx", help="person model (YOLO26n ONNX: fastest CPU)")
    ap.add_argument("--camera", default="CAM01")
    ap.add_argument("--imgsz", type=int, default=416)
    ap.add_argument("--conf", type=float, default=0.3, help="confidence threshold")
    ap.add_argument("--entry-grace", type=int, default=3, help="inside-frames before alarm fires")
    ap.add_argument("--exit-grace", type=int, default=8, help="clear frames before alarm stops")
    ap.add_argument("--photo-every", type=int, default=30, help="seconds between evidence re-captures")
    args = ap.parse_args()

    src = int(args.source) if str(args.source).isdigit() else str(args.source)
    zones = load_zones()
    if not zones:
        print("[!] No zones.json found. Run: python define_zones.py --source 0 first")
        print("    (click polygon points, press S to save, Q to quit)")

    wpath = Path(args.model)
    if not wpath.exists():
        wpath = BASE / args.model  # weights live next to the script
    model = YOLO(str(wpath))  # person class = 0
    print("[...] Warming up model (3 dummy frames)...")
    _warm = np.zeros((480, 480, 3), dtype=np.uint8)
    for _ in range(3):
        model(_warm, verbose=False)
    print("[OK] Model hot.")

    # single-instance guard: one camera, one holder.
    _zlock = BASE / "zone.lock"
    if _zlock.exists():
        try:
            _zold = int(_zlock.read_text().strip())
        except ValueError:
            _zold = None
        _zalive = False
        if _zold and _zold != os.getpid():
            try:
                _zout = subprocess.run(["tasklist", "/FI", f"PID eq {_zold}", "/FO", "CSV", "/NH"],
                                       capture_output=True, text=True, timeout=15).stdout
                _zalive = f'"{_zold}"' in _zout
            except Exception:
                _zalive = False
        if _zalive:
            print(f"[!] Another Zone_Alert (pid {_zold}) already holds the camera - exiting.")
            return
    _zlock.write_text(str(os.getpid()))
    cap = None
    for attempt in range(1, 4):
        try:
            cap = cv2.VideoCapture(src)
        except Exception as e:  # flaky USB drivers can throw on open
            print(f"[!] Camera open raised {e} (attempt {attempt}/3), retrying in 3s...")
            cap = None
            time.sleep(3)
            continue
        if cap is not None and cap.isOpened():
            break
        print(f"[!] Camera open failed (attempt {attempt}/3), retrying in 3s...")
        time.sleep(3)
    if not cap.isOpened():
        print(f"[X] Cannot open source {args.source} - check camera cable / RTSP URL")
        return
    print(f"[OK] Camera opened: {args.source}")

    today = datetime.now().strftime("%Y-%m-%d")
    (EVIDENCE_DIR / today).mkdir(parents=True, exist_ok=True)
    (LOG_DIR).mkdir(parents=True, exist_ok=True)
    csv_path = LOG_DIR / f"incidents_{today}.csv"
    entries_today = 0
    if csv_path.exists():  # resume today's count across restarts
        with open(csv_path) as _f:
            entries_today = max(0, sum(1 for _ in _f) - 1)
    entries_run = 0
    last_alert = {}  # zone_name -> timestamp for occupied re-capture
    alarm_on = False  # looping siren state: on while anyone is inside
    last_siren = 0.0  # last heartbeat replay (self-heals a died loop)
    alarm_active = False  # debounced zone state (entry/exit logic)
    inside_streak = 0
    outside_streak = 0
    ENTRY_GRACE = args.entry_grace
    EXIT_GRACE = args.exit_grace
    PHOTO_EVERY = args.photo_every
    fps_t0 = time.time()
    fps_n = 0
    fps = 0.0

    print(f"[OK] Zone_Alert running on {args.source}. Press Q to quit.")
    for r in model.track(source=src, stream=True, persist=True, imgsz=args.imgsz,
                         classes=[0], conf=args.conf, verbose=False,
                         vid_stride=2, tracker="bytetrack.yaml"):
        frame = r.orig_img
        h, w = frame.shape[:2]

        # draw zones
        for z in zones:
            pts = np.array(z["points"], np.int32)
            cv2.polylines(frame, [pts], True, (255, 165, 0), 2)
            cv2.putText(frame, z.get("name", "ZONE"), tuple(pts[0]),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 165, 0), 2)

        # SAFE (left) | DANGER (right) divider, minimal chips at bottom corners
        cv2.line(frame, (w // 2, 0), (w // 2, h), (200, 200, 200), 1)
        cv2.putText(frame, "SAFE", (10, h - 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.putText(frame, "DANGER", (w - 80, h - 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        in_frame = 0
        intrusions = 0
        first_tid = None
        first_zone = ""
        if r.boxes is not None and r.boxes.id is not None:
            ids = r.boxes.id.cpu().numpy().astype(int)
            xyxy = r.boxes.xyxy.cpu().numpy()
            in_frame = len(xyxy)
            for box, tid in zip(xyxy, ids):
                x1, y1, x2, y2 = map(int, box)
                foot = ((x1 + x2) // 2, y2)  # feet dot (best for overhead CCTV)
                center = ((x1 + x2) // 2, (y1 + y2) // 2)  # body center (best for webcam demo)
                inside = False
                hit_zone = ""
                for z in zones:
                    poly = np.array(z["points"], np.float32)
                    if (cv2.pointPolygonTest(poly, foot, False) >= 0
                            or cv2.pointPolygonTest(poly, center, False) >= 0):
                        inside = True
                        hit_zone = z.get("name", "ZONE")
                        break
                color = (0, 0, 255) if inside else (0, 255, 0)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1)
                cv2.circle(frame, foot, 3, color, -1)
                cv2.putText(frame, f"ID {tid}", (x1, y1 - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                if inside:
                    intrusions += 1
                    if first_tid is None:
                        first_tid, first_zone = int(tid), hit_zone

        # debounced entry/exit state machine
        now = time.time()
        raw_inside = intrusions > 0
        if raw_inside:
            inside_streak += 1
            outside_streak = 0
        else:
            outside_streak += 1
            inside_streak = 0
        if not alarm_active and inside_streak >= ENTRY_GRACE:
            alarm_active = True
            last_alert[first_zone] = 0  # force immediate entry capture below
        if alarm_active and outside_streak >= EXIT_GRACE:
            alarm_active = False
        if alarm_active and raw_inside and now - last_alert.get(first_zone, 0) > PHOTO_EVERY:
            last_alert[first_zone] = now
            ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            photo = EVIDENCE_DIR / today / f"{args.camera}_{ts}_ZONE_INTRUSION_ID{first_tid}.jpg"
            snap = frame  # default: full frame
            zb = next((z for z in zones if z.get("name") == first_zone), zones[0] if zones else None)
            if zb is not None:  # crop to the danger box only
                pts = np.array(zb["points"])
                zx1, zy1 = max(0, int(pts[:, 0].min())), max(0, int(pts[:, 1].min()))
                zx2, zy2 = min(w, int(pts[:, 0].max())), min(h, int(pts[:, 1].max()))
                if zx2 > zx1 and zy2 > zy1:
                    snap = frame[zy1:zy2, zx1:zx2]
            cv2.imwrite(str(photo), snap)
            log_incident(csv_path, {
                "time": datetime.now().isoformat(timespec="seconds"),
                "camera": args.camera,
                "violation_type": "ZONE_INTRUSION",
                "track_id": first_tid,
                "in_frame_count": in_frame,
                "photo_path": str(photo),
            })
            print(f"[!] Intrusion ID:{first_tid} saved {photo.name}")
            entries_run += 1
            entries_today += 1

        # live FPS meter
        fps_n += 1
        if now - fps_t0 >= 1.0:
            fps = fps_n / (now - fps_t0)
            fps_n = 0
            fps_t0 = now

        # status pill + counter
        pill_color = (0, 0, 210) if alarm_active else (0, 170, 0)
        cv2.rectangle(frame, (8, 8), (178, 38), pill_color, -1)
        cv2.circle(frame, (26, 23), 6, (255, 255, 255), -1)
        cv2.putText(frame, "INTRUSION" if alarm_active else "SECURE", (40, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, f"{in_frame} in view | {fps:.0f} FPS", (10, 54),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (230, 230, 230), 1)
        cv2.putText(frame, f"Entries today: {entries_today} | Run: {entries_run}", (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        if alarm_active:
            # no screen halo - alarm + REC pill carry the alert
            if int(time.time() * 2) % 2 == 0:  # blinking REC evidence indicator
                cv2.circle(frame, (w - 232, 23), 7, (0, 0, 255), -1)
                cv2.putText(frame, "REC  EVIDENCE CAPTURING", (w - 218, 28),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
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
        cv2.imshow("Zone_Alert - Q to quit", frame)
        if cv2.waitKey(1) & 0xFF in (ord("q"), ord("Q")):
            break

    cap.release()
    cv2.destroyAllWindows()
    try:
        _zlock.unlink()
    except OSError:
        pass
    print(f"[OK] Log: {csv_path} | entries run={entries_run} today={entries_today}")


if __name__ == "__main__":
    main()
