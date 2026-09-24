"""MODE 1: PPE_Check - helmet + vest (+ mask / gloves where model supports it).
Run live:   python mode_ppe.py --source 0
Test photo: python mode_ppe.py --source live_ready.jpg   (saves live_ready_annotated.jpg)

Why this version detects small items (mask/gloves) better than the old one:
  1. Single conf=0.4 killed small boxes. Now inference runs at the LOWEST
     threshold (mask/glove ~0.20) and each class is filtered separately.
  2. Old center_in() dropped helmets sitting above the person box. Now the
     person box is expanded (top +25%) and overlap (IoA) also counts.
  3. Old code warned on ANY single frame without a hat box (flicker = siren
     spam). Now a violation needs --smooth consecutive frames (default 3).
  4. Mask/gloves are only judged when the person is CLOSE (tall box). Far
     faces are too few pixels for any nano model -> marked FAR, not NO MASK.
  5. ppe_v8n.pt has NO glove classes at all, so gloves can never come from it.
     Use --gloves-model ppe_v8m.pt at a 2-3 m gate for gloves-only backup.
     (ppe_v8m can't see Person on webcam close-ups - that is a broken weight,
     not a settings bug - so it is never used for person/helmet/vest.)

Speed (i5 CPU, no GPU): threaded design - the camera display runs at the full
camera rate (~30 FPS smooth video) while AI inference runs behind at ~10 Hz
and its boxes are redrawn live on every video frame. Main pass at 480px (big
items: person/helmet/vest); face-zoom crop re-scans tiny masks; gloves backup
at 320px, only when a person is present. Measured on i5-1334U:
v8n 65ms@480, v8m 167ms@320, face crop ~35ms.

Yellow V-44 mask note: ppe_v8n was trained on white/blue surgical masks.
A yellow cloth mask usually scores as NO-Mask even at conf 0.15. The lower
mask threshold (0.20) + face-zoom crop gives it the best chance, but if it
still reads NO-Mask, the honest fix is fine-tuning nano on 50-100 yellow-mask
photos (see bottom of this file). Same story for blue gloves (Phase 2).
"""
import argparse
import json
import os
import subprocess
import threading
import time
import traceback
from collections import defaultdict
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
TRACKER_YAML = BASE / "tracker_ppe.yaml"  # long-memory ByteTrack (falls back to stock if missing)
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
T0 = time.time()  # process start (the imports above already cost ~10s on CPU-only machines)


def pid_alive(pid):
    """tasklist-based PID check (os.kill probing lies on Win10 Home)."""
    try:
        out = subprocess.run(["tasklist", "/FI", f"PID eq {pid}", "/FO", "CSV", "/NH"],
                             capture_output=True, text=True, timeout=15).stdout
        return f'"{pid}"' in out
    except Exception:
        return False


def expand_person(pbox, top=0.25, side=0.10, bottom=0.05):
    """Expand a person box so helmets above the head and hands at the edge count."""
    x1, y1, x2, y2 = (float(v) for v in pbox)
    pw, ph = max(1.0, x2 - x1), max(1.0, y2 - y1)
    return (x1 - pw * side, y1 - ph * top, x2 + pw * side, y2 + ph * bottom)


def _iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    aa = max(1.0, (a[2] - a[0]) * (a[3] - a[1]))
    bb = max(1.0, (b[2] - b[0]) * (b[3] - b[1]))
    return inter / (aa + bb - inter)


def dedup_scored(items, thr=0.6):
    """Drop near-duplicate (box, conf) detections of the same object, keep the
    highest-confidence one."""
    kept = []
    for b, cf in sorted(items, key=lambda t: t[1], reverse=True):
        if all(_iou(b, k) < thr for k, _ in kept):
            kept.append((b, cf))
    return kept


def dedup_persons(persons, thr=0.65):
    """Same-object person tracks detected twice -> keep highest conf triple."""
    kept = []
    for (b, tid, cf) in sorted(persons, key=lambda t: t[2], reverse=True):
        if all(_iou(b, k) < thr for k, _, _ in kept):
            kept.append((b, tid, cf))
    return kept


CONF_MARGIN = 0.10  # explicit NO-* only counts if it beats the positive by this


def _side(has_list, flag_list, margin=CONF_MARGIN):
    """Arbitrate conflicting detections. Returns (has_item, flag_effective):
    a NO-* box convicts only when nothing positive is worn, or when it beats
    the best positive box by `margin`. Stops a stray low-conf NO-Hardhat from
    overruling a solid Hardhat (or vice versa)."""
    best_has = max((cf for _, cf in has_list), default=0.0)
    best_flag = max((cf for _, cf in flag_list), default=0.0)
    has = best_has > 0.0
    flag = (best_flag > best_has + margin) if has else (best_flag > 0.0)
    return has, flag


def associated(ppe_box, pbox):
    """Is a PPE box worn by this person? Center-in-expanded OR big overlap."""
    ex1, ey1, ex2, ey2 = expand_person(pbox)
    cx, cy = (ppe_box[0] + ppe_box[2]) / 2, (ppe_box[1] + ppe_box[3]) / 2
    if ex1 <= cx <= ex2 and ey1 <= cy <= ey2:
        return True
    # IoA: intersection over PPE-box area (small box mostly inside person)
    ix1, iy1 = max(ex1, ppe_box[0]), max(ey1, ppe_box[1])
    ix2, iy2 = min(ex2, ppe_box[2]), min(ey2, ppe_box[3])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    pa = max(1.0, (ppe_box[2] - ppe_box[0]) * (ppe_box[3] - ppe_box[1]))
    return (iw * ih) / pa > 0.30


def save_tally(path, visitors, ok_visitors, tally):
    """Persist counts-only tally (no photos). Called on every new violation event."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        f.write("metric,count\n")
        f.write(f"visitors,{len(visitors)}\n")
        f.write(f"ppe_ok,{len(ok_visitors)}\n")
        for k, v in tally.items():
            f.write(f"{k},{v}\n")


def resolve_ids(model):
    lower = {i: str(n).lower() for i, n in model.names.items()}

    def _find(cands):
        for cand in cands:
            cl = cand.lower()
            for idx, nm in lower.items():
                if nm == cl or nm.replace("-", " ").replace("_", " ") == cl:
                    return idx
        return None

    return {
        "HARDHAT": _find(["hardhat", "hard hat"]),
        "NO_HARDHAT": _find(["no-hardhat", "no hardhat", "no-hard hat", "no- hardhat"]),
        "VEST": _find(["safety vest", "safety-vest", "vest"]),
        "NO_VEST": _find(["no-safety vest", "no-safety-vest", "no vest", "no-vest"]),
        "PERSON": _find(["person"]),
        "GLOVES": _find(["gloves", "glove", "safety gloves"]),
        "NO_GLOVES": _find(["no-gloves", "no gloves", "no glove", "no-glove"]),
        "MASK": _find(["mask", "face mask"]),
        "NO_MASK": _find(["no-mask", "no mask"]),
    }


def min_thresh(args, ids):
    """Inference threshold = lowest enabled per-class threshold."""
    cands = [args.conf]
    if ids["PERSON"] is not None:
        cands.append(args.person_conf)
    if ids["HARDHAT"] is not None or ids["VEST"] is not None:
        cands.append(args.ppe_conf)
    if (ids["MASK"] is not None or ids["NO_MASK"] is not None) and not args.ignore_mask:
        cands.append(args.mask_conf)
    if (ids["GLOVES"] is not None or ids["NO_GLOVES"] is not None) and not args.ignore_gloves:
        cands.append(args.glove_conf)
    return min(cands)


def box_passes(cls_idx, conf_val, args, ids):
    """Per-class confidence gate (lets tiny mask/glove boxes survive)."""
    if ids["PERSON"] is not None and cls_idx == ids["PERSON"]:
        return conf_val >= args.person_conf
    if cls_idx in (ids["MASK"], ids["NO_MASK"]):
        return conf_val >= args.mask_conf
    if cls_idx in (ids["GLOVES"], ids["NO_GLOVES"]):
        return conf_val >= args.glove_conf
    return conf_val >= args.ppe_conf


BOX_COLORS = {}  # filled per run from resolved ids


def draw_ppe_box(frame, box, cls_idx, conf_val, names):
    color = BOX_COLORS.get(cls_idx, (255, 255, 0))
    x1, y1, x2, y2 = map(int, box)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1)
    cv2.putText(frame, f"{names[cls_idx]} {conf_val:.2f}", (x1, max(0, y1 - 5)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)


def face_zoom_boxes(frame, persons, model, ids, args):
    """Second close-up pass: crop each head region, re-run nano on the crop.
    A mask that is 40px in full frame becomes ~150px in the crop - this is what
    makes 2m+ mask detection possible. Returns ([(box, conf)], [(box, conf)])
    in frame coords (boxes come back in crop-pixel coords, so just add offset)
    and also draws them on the given frame (used by the photo-test path)."""
    out_masks, out_nomasks = [], []
    if ids["MASK"] is None and ids["NO_MASK"] is None:
        return out_masks, out_nomasks
    fh, fw = frame.shape[:2]
    for pbox, _tid, _pc in persons:
        x1, y1, x2, y2 = (float(v) for v in pbox)
        ph, pw = y2 - y1, x2 - x1
        if ph < args.min_face_h:
            continue
        hx1, hy1 = max(0, int(x1 - 0.15 * pw)), max(0, int(y1 - 0.15 * ph))
        hx2, hy2 = min(fw, int(x2 + 0.15 * pw)), min(fh, int(y1 + 0.42 * ph))
        if hx2 - hx1 < 40 or hy2 - hy1 < 40:
            continue
        crop = frame[hy1:hy2, hx1:hx2]
        try:
            rz = model(crop, conf=args.mask_conf, imgsz=args.face_imgsz, verbose=False)[0]
        except Exception:
            continue
        if rz.boxes is None or len(rz.boxes) == 0:
            continue
        for box, c, cf in zip(rz.boxes.xyxy.cpu().numpy(), rz.boxes.cls.cpu().numpy().astype(int),
                              rz.boxes.conf.cpu().numpy()):
            if c not in (ids["MASK"], ids["NO_MASK"]) or float(cf) < args.mask_conf:
                continue
            b = [float(box[0]) + hx1, float(box[1]) + hy1, float(box[2]) + hx1, float(box[3]) + hy1]
            (out_masks if c == ids["MASK"] else out_nomasks).append((b, float(cf)))
            x1b, y1b, x2b, y2b = map(int, b)
            cv2.rectangle(frame, (x1b, y1b), (x2b, y2b), (0, 255, 255), 1)
            cv2.putText(frame, f"{model.names[c]} {float(cf):.2f} (zoom)", (x1b, max(0, y1b - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
    return out_masks, out_nomasks


def glove_plausible(gbox, pbox):
    """Hands are small and below the head. A glove box only counts as worn if:
    area <= 12% of the person box (rejects blue shirts/tarps/walls) and its
    center sits below the top 20% of the person (rejects caps/hair-nets).
    This keeps color-blob false hits from convicting; it cannot make a blind
    model SEE unseen glove colors - that needs fine-tuning."""
    px1, py1, px2, py2 = (float(v) for v in pbox)
    pw, ph = max(1.0, px2 - px1), max(1.0, py2 - py1)
    ga = max(0.0, (gbox[2] - gbox[0])) * max(0.0, (gbox[3] - gbox[1]))
    if ga > 0.12 * pw * ph:
        return False
    cx, cy = (gbox[0] + gbox[2]) / 2, (gbox[1] + gbox[3]) / 2
    if cy < py1 + 0.20 * ph:
        return False
    return True


def _is_textured(gray, bx, by, bw2, bh2, thresh=0.08):
    """True if the blob interior is full of edges (plaid/checks). A smooth glove
    (or hand) has a quiet interior; a shirt check is edgy even inside. Looks at
    the inner 50% so the blob's own outline never counts against it."""
    ix1, iy1 = bx + bw2 // 4, by + bh2 // 4
    ix2, iy2 = bx + 3 * bw2 // 4, by + 3 * bh2 // 4
    if ix2 - ix1 < 10 or iy2 - iy1 < 10:
        return False
    edges = cv2.Canny(gray[iy1:iy2, ix1:ix2], 50, 150)
    return (float(edges.mean()) / 255.0) > thresh


def bare_hand_boxes(frame, pbox, face_boxes):
    """Classical-vision fallback for bare hands (no training needed): skin blobs
    shaped like a hand, below the head, face excluded. Returns [(box, 0.90)] as
    NO-Gloves pseudo-detections flowing through the normal association /
    arbitration / lock pipeline. Fires red on bare skin; gloved or hidden hands
    stay idle. Long sleeves read cleanest (bare forearms can mimic hands)."""
    x1, y1, x2, y2 = (int(v) for v in pbox)
    fh, fw = frame.shape[:2]
    x1, y1, x2, y2 = max(0, x1), max(0, y1), min(fw, x2), min(fh, y2)
    if x2 - x1 < 30 or y2 - y1 < 60:
        return []
    crop = frame[y1:y2, x1:x2]
    ycc = cv2.cvtColor(crop, cv2.COLOR_BGR2YCrCb)
    skin = cv2.inRange(ycc, np.array([0, 133, 77], dtype=np.uint8),
                       np.array([255, 173, 127], dtype=np.uint8))
    h, w = skin.shape[:2]
    skin[:int(h * 0.30), :] = 0  # head zone: never hands
    for (fx1, fy1, fx2, fy2) in face_boxes:  # face + full neck column excluded
        ex1, ex2 = max(0, int(fx1 - x1 - 10)), min(w, int(fx2 - x1 + 10))
        ey1, ey2 = max(0, int(fy1 - y1 - 10)), min(h, int(fy2 - y1 + int(max(0.0, fy2 - fy1) * 1.0)))
        if ex2 > ex1 and ey2 > ey1:
            skin[ey1:ey2, ex1:ex2] = 0
    skin = cv2.morphologyEx(skin, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    cnts, _ = cv2.findContours(skin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out, pa = [], max(1.0, float(w * h))
    for c in cnts:
        area = cv2.contourArea(c)
        if not (0.005 * pa <= area <= 0.08 * pa):
            continue
        bx, by, bw2, bh2 = cv2.boundingRect(c)
        ar = bw2 / max(1, bh2)
        if not (0.4 <= ar <= 2.5):  # hands are compact; forearms are elongated
            continue
        if _is_textured(gray, bx, by, bw2, bh2):
            continue  # plaid/checks, not skin-on-hand
        out.append(([float(x1 + bx), float(y1 + by),
                     float(x1 + bx + bw2), float(y1 + by + bh2)], 0.90))
    return out


def blue_glove_boxes(frame, pbox, face_boxes, spec):
    """Calibrated-color glove detection (see calibrate_glove.py): YOUR sampled
    blue, in hand zones only, becomes (box, 0.85) positive evidence through the
    normal association / arbitration / lock pipeline - real green YES without
    any training. Same zone/exclusion/shape gates as the skin fallback, so
    blue walls and shirts outside hand zones never count."""
    x1, y1, x2, y2 = (int(v) for v in pbox)
    fh, fw = frame.shape[:2]
    x1, y1, x2, y2 = max(0, x1), max(0, y1), min(fw, x2), min(fh, y2)
    if x2 - x1 < 30 or y2 - y1 < 60:
        return []
    crop = frame[y1:y2, x1:x2]
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    lo = np.array([max(0.0, spec["h"] - 10), max(0.0, spec["s"] - 60), max(0.0, spec["v"] - 60)],
                  dtype=np.uint8)
    hi = np.array([min(179.0, spec["h"] + 10), 255, 255], dtype=np.uint8)
    blue = cv2.inRange(hsv, lo, hi)
    h, w = blue.shape[:2]
    blue[:int(h * 0.30), :] = 0  # head zone: never hands
    for (fx1, fy1, fx2, fy2) in face_boxes:  # face + full neck column excluded
        ex1, ex2 = max(0, int(fx1 - x1 - 10)), min(w, int(fx2 - x1 + 10))
        ey1, ey2 = max(0, int(fy1 - y1 - 10)), min(h, int(fy2 - y1 + int(max(0.0, fy2 - fy1) * 1.0)))
        if ex2 > ex1 and ey2 > ey1:
            blue[ey1:ey2, ex1:ex2] = 0
    blue = cv2.morphologyEx(blue, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    cnts, _ = cv2.findContours(blue, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out, pa = [], max(1.0, float(w * h))
    for c in cnts:
        area = cv2.contourArea(c)
        if not (0.005 * pa <= area <= 0.08 * pa):
            continue
        bx, by, bw2, bh2 = cv2.boundingRect(c)
        ar = bw2 / max(1, bh2)
        if not (0.4 <= ar <= 2.5):  # compact blobs only, not sleeves/walls
            continue
        if _is_textured(gray, bx, by, bw2, bh2):
            continue  # check pattern, not a smooth glove
        out.append(([float(x1 + bx), float(y1 + by),
                     float(x1 + bx + bw2), float(y1 + by + bh2)], 0.85))
    return out


def judge_person(pbox, hats, nohats, vests, novests, gloves, nogloves,
                 masks, nomasks, judge_mask, judge_gloves, glove_absence_counts=True):
    """Return (tags, detail, confs). Lists hold (box, conf); absence alone only
    counts when close + persistent (persistence is handled by the caller locks);
    explicit NO-* convicts via confidence arbitration (_side).
    confs = {item: (best associated positive conf, best associated NO-* conf)}.
    glove_absence_counts=False when gloves come only from the backup model
    (its recall is poor, so absence means 'can't tell', not 'no gloves')."""
    _assoc = lambda items: [(b, cf) for b, cf in items if associated(b, pbox)]
    _gassoc = lambda items: [(b, cf) for b, cf in items
                             if associated(b, pbox) and glove_plausible(b, pbox)]
    a_hats, a_nohats = _assoc(hats), _assoc(nohats)
    a_vests, a_novests = _assoc(vests), _assoc(novests)
    a_masks, a_nomasks = _assoc(masks), _assoc(nomasks)
    a_gloves, a_nogloves = _gassoc(gloves), _gassoc(nogloves)
    has_hat, flag_nohat = _side(a_hats, a_nohats)
    has_vest, flag_novest = _side(a_vests, a_novests)
    has_mask, flag_nomask = _side(a_masks, a_nomasks)
    has_gloves, flag_nogloves = _side(a_gloves, a_nogloves)
    _best = lambda pairs: max((cf for _, cf in pairs), default=0.0)
    confs = {"helmet": (_best(a_hats), _best(a_nohats)),
             "vest": (_best(a_vests), _best(a_novests)),
             "mask": (_best(a_masks), _best(a_nomasks)),
             "gloves": (_best(a_gloves), _best(a_nogloves))}
    tags = []
    if flag_nohat or not has_hat:
        tags.append("NO HELMET")
    if flag_novest or not has_vest:
        tags.append("NO VEST")
    if judge_gloves and (flag_nogloves or ((not has_gloves) and glove_absence_counts)):
        tags.append("NO GLOVES")
    if judge_mask and (flag_nomask or not has_mask):
        tags.append("NO MASK")
    return tags, (has_hat, has_vest, has_mask, has_gloves), confs


# ---------------------------------------------------------------- model loading
def load_models(args, checks):
    """Load primary (+ optional gloves backup) weights. Nano takes ~10s,
    ppe_v8m adds ~30s on CPU. The live window is already up meanwhile, so the
    user sees video + a LOADING banner instead of a dead click."""
    wpath = BASE / args.model if not Path(args.model).exists() else Path(args.model)
    model = YOLO(str(wpath))
    ids = resolve_ids(model)
    has_mask = (ids["MASK"] is not None or ids["NO_MASK"] is not None) and not args.ignore_mask \
        and "mask" in checks
    has_gloves_main = (ids["GLOVES"] is not None or ids["NO_GLOVES"] is not None) \
        and not args.ignore_gloves

    gmodel, gids = None, {}
    if args.gloves_model and not args.ignore_gloves and "gloves" in checks:
        gpath = BASE / args.gloves_model if not Path(args.gloves_model).exists() else Path(args.gloves_model)
        if gpath.exists():
            gmodel = YOLO(str(gpath))
            gids = resolve_ids(gmodel)
            print(f"[INFO] gloves backup {gpath.name}: "
                  f"GLOVES={gids['GLOVES']} NO_GLOVES={gids['NO_GLOVES']}", flush=True)
            if gids["GLOVES"] is None and gids["NO_GLOVES"] is None:
                print("[!] gloves-model has no glove classes - ignoring it", flush=True)
                gmodel = None
        else:
            print(f"[!] --gloves-model {args.gloves_model} not found - gloves from primary model only",
                  flush=True)

    global BOX_COLORS
    BOX_COLORS = {v: c for v, c in [
        (ids["HARDHAT"], (0, 255, 0)), (ids["NO_HARDHAT"], (0, 0, 255)),
        (ids["VEST"], (0, 255, 0)), (ids["NO_VEST"], (0, 0, 255)),
        (ids["MASK"], (0, 255, 255)), (ids["NO_MASK"], (0, 165, 255)),
        (ids["GLOVES"], (255, 0, 0)), (ids["NO_GLOVES"], (255, 0, 255)),
    ] if v is not None}

    print(f"[INFO] Model {wpath.name} classes: {model.names}", flush=True)
    print(f"[INFO] ids HARDHAT={ids['HARDHAT']} NO_HARDHAT={ids['NO_HARDHAT']} VEST={ids['VEST']} "
          f"NO_VEST={ids['NO_VEST']} PERSON={ids['PERSON']} GLOVES={ids['GLOVES']} "
          f"NO_GLOVES={ids['NO_GLOVES']} MASK={ids['MASK']} NO_MASK={ids['NO_MASK']}", flush=True)
    base = min_thresh(args, ids)
    print(f"[INFO] thresholds base={base:.2f} person={args.person_conf:.2f} ppe={args.ppe_conf:.2f} "
          f"mask={args.mask_conf:.2f} glove={args.glove_conf:.2f} smooth={args.smooth} imgsz={args.imgsz}",
          flush=True)
    if ids["GLOVES"] is None and gmodel is None and not args.ignore_gloves:
        _skin_will = ("gloves" in checks) and args.skin
        if _skin_will:
            print("[INFO] no glove classes in weights -> skin-based bare-hand fallback "
                  "active (red on bare skin, idle otherwise)", flush=True)
        else:
            print("[INFO] no glove classes in weights -> gloves idle until fine-tuned "
                  "weights arrive (--gloves-model best.pt) or --skin is passed", flush=True)
    return model, ids, has_mask, has_gloves_main, gmodel, gids, base


# ---------------------------------------------------------------- image test
def run_image_test(args, model, ids, gmodel, gids, checks):
    img_path = Path(args.source)
    frame = cv2.imread(str(img_path))
    if frame is None:
        print(f"[X] Cannot read image {img_path}")
        return
    h, w = frame.shape[:2]
    base = min_thresh(args, ids)
    r = model(frame, conf=base, imgsz=args.imgsz, verbose=False)[0]
    persons, hats, nohats, vests, novests, gloves, nogloves, masks, nomasks = [], [], [], [], [], [], [], [], []
    if r.boxes is not None and len(r.boxes) > 0:
        for box, c, cf in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.cls.cpu().numpy().astype(int),
                              r.boxes.conf.cpu().numpy()):
            if not box_passes(c, float(cf), args, ids):
                continue
            if c in (ids["HARDHAT"], ids["NO_HARDHAT"]) and "helmet" not in checks:
                continue
            if c in (ids["VEST"], ids["NO_VEST"]) and "vest" not in checks:
                continue
            if c in (ids["MASK"], ids["NO_MASK"]) and "mask" not in checks:
                continue
            if c in (ids["GLOVES"], ids["NO_GLOVES"]) and "gloves" not in checks:
                continue
            b = [float(v) for v in box]
            draw_ppe_box(frame, b, c, float(cf), model.names)
            if ids["PERSON"] is not None and c == ids["PERSON"]:
                persons.append((b, float(cf)))
            elif ids["HARDHAT"] is not None and c == ids["HARDHAT"]:
                hats.append((b, float(cf)))
            elif ids["NO_HARDHAT"] is not None and c == ids["NO_HARDHAT"]:
                nohats.append((b, float(cf)))
            elif ids["VEST"] is not None and c == ids["VEST"]:
                vests.append((b, float(cf)))
            elif ids["NO_VEST"] is not None and c == ids["NO_VEST"]:
                novests.append((b, float(cf)))
            elif ids["GLOVES"] is not None and c == ids["GLOVES"]:
                gloves.append((b, float(cf)))
            elif ids["NO_GLOVES"] is not None and c == ids["NO_GLOVES"]:
                nogloves.append((b, float(cf)))
            elif ids["MASK"] is not None and c == ids["MASK"]:
                masks.append((b, float(cf)))
            elif ids["NO_MASK"] is not None and c == ids["NO_MASK"]:
                nomasks.append((b, float(cf)))
    # optional gloves-only backup model
    if gmodel is not None and not args.ignore_gloves:
        rg = gmodel(frame, conf=args.glove_conf, imgsz=args.glove_imgsz, verbose=False)[0]
        if rg.boxes is not None and len(rg.boxes) > 0:
            for box, c, cf in zip(rg.boxes.xyxy.cpu().numpy(), rg.boxes.cls.cpu().numpy().astype(int),
                                  rg.boxes.conf.cpu().numpy()):
                if c not in (gids["GLOVES"], gids["NO_GLOVES"]) or float(cf) < args.glove_conf:
                    continue
                b = [float(v) for v in box]
                (gloves if c == gids["GLOVES"] else nogloves).append((b, float(cf)))
                x1, y1, x2, y2 = map(int, b)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 1)
                cv2.putText(frame, f"{gmodel.names[c]} {float(cf):.2f} (glove-m)",
                            (x1, max(0, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 0, 0), 1)
    print(f"[TEST] {img_path.name} {w}x{h} base_conf={base:.2f} imgsz={args.imgsz}")
    print(f"       raw kept: person={len(persons)} hat={len(hats)} nohat={len(nohats)} "
          f"vest={len(vests)} novest={len(novests)} mask={len(masks)} nomask={len(nomasks)} "
          f"gloves={len(gloves)} nogloves={len(nogloves)}")
    if args.face_zoom and persons and not args.ignore_mask:
        zm, zn = face_zoom_boxes(frame, [(b, -1, cf) for b, cf in persons], model, ids, args)
        masks += zm
        nomasks += zn
        print(f"       face-zoom added: mask={len(zm)} nomask={len(zn)}")
    if not persons:
        print("[TEST] NO Person box kept -> try --person-conf 0.25 --imgsz 640, move to 2m, center chest-up.")
    for i, (pbox, pconf) in enumerate(persons):
        ph = pbox[3] - pbox[1]
        judge_mask = not args.ignore_mask and "mask" in checks \
            and (ids["MASK"] is not None or ids["NO_MASK"] is not None) \
            and ph >= args.min_face_h
        judge_gloves = not args.ignore_gloves and "gloves" in checks \
            and (gmodel is not None or ids["GLOVES"] is not None) \
            and ph >= args.min_glove_h
        tags, _, _ = judge_person(pbox, hats, nohats, vests, novests, gloves, nogloves,
                               masks, nomasks, judge_mask, judge_gloves,
                               glove_absence_counts=(ids["GLOVES"] is not None))
        if "helmet" not in checks:
            tags = [t for t in tags if t != "NO HELMET"]
        if "vest" not in checks:
            tags = [t for t in tags if t != "NO VEST"]
        x1, y1, x2, y2 = map(int, pbox)
        color = (0, 255, 0) if not tags else (0, 0, 255)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, "OK" if not tags else "WARNING " + "+".join(tags), (x1, y1 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        extra = []
        if "mask" not in checks:
            extra.append("mask not in profile")
        elif not judge_mask:
            extra.append(f"mask skipped (h={ph:.0f}<{args.min_face_h})")
        if (ids["GLOVES"] is None and gmodel is None) or args.ignore_gloves:
            extra.append("gloves n/a")
        elif "gloves" not in checks:
            extra.append("gloves not in profile")
        elif not judge_gloves:
            extra.append(f"gloves skipped (h={ph:.0f}<{args.min_glove_h})")
        print(f"[TEST] person{i} conf={pconf:.2f} h={ph:.0f}px -> "
              f"{'OK' if not tags else 'WARNING ' + '+'.join(tags)}"
              + (f" [{'; '.join(extra)}]" if extra else ""))
    out = img_path.with_name(img_path.stem + "_annotated.jpg")
    cv2.imwrite(str(out), frame)
    print(f"[TEST] annotated -> {out.name}")
    print("[TIP] white helmet: seat it (hand off), 2 m, centered, face light on. "
          "yellow mask: pull over nose+mouth; if still NO-Mask, it needs fine-tuning (see file docstring).")


# ---------------------------------------------------------------- live loop
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="0", help="0 webcam / RTSP URL / image.jpg for single-photo test")
    ap.add_argument("--model", default="ppe_v8n.pt", help="primary PPE weights (nano, CPU-friendly)")
    ap.add_argument("--checks", default="all",
                    help="comma list from helmet,vest,mask,gloves (e.g. 'helmet,mask' or 'vest,gloves'); 'all' checks everything")
    ap.add_argument("--gloves-model", default="", help="optional 2nd weights for gloves only, e.g. ppe_v8m.pt")
    ap.add_argument("--camera", default="CAM01")
    ap.add_argument("--imgsz", type=int, default=480, help="main pass size (480 = sweet spot: big items stay accurate, ~35%% faster than 640)")
    ap.add_argument("--conf", type=float, default=0.25, help="legacy global floor (inference uses the min of all thresholds)")
    ap.add_argument("--person-conf", type=float, default=0.35)
    ap.add_argument("--ppe-conf", type=float, default=0.30, help="helmet/vest + explicit NO-* boxes")
    ap.add_argument("--mask-conf", type=float, default=0.20, help="mask boxes are tiny -> keep low")
    ap.add_argument("--glove-conf", type=float, default=0.20, help="glove boxes are tiny -> keep low")
    ap.add_argument("--smooth", type=int, default=3, help="frames to raise an alarm (fast to warn)")
    ap.add_argument("--clear-after", type=int, default=12,
                    help="contrary frames before a LOCKED verdict flips (slow to un-warn: kills YES/NO flicker)")
    ap.add_argument("--min-face-h", type=int, default=130, help="px person height below which mask is NOT judged (FAR)")
    ap.add_argument("--min-glove-h", type=int, default=180, help="px person height below which gloves are NOT judged (FAR)")
    ap.add_argument("--ignore-mask", action="store_true", help="turn off mask checking (far-field cams)")
    ap.add_argument("--ignore-gloves", action="store_true", help="turn off glove checking")
    ap.add_argument("--skin", action="store_true",
                    help="opt-in skin-based bare-hand fallback (off by default: skin-tone "
                         "plaid shirts false-fire it; solid dark sleeves read cleanest)")
    ap.add_argument("--skin-every", type=int, default=3, help="run bare-hand check every N inferences")
    ap.add_argument("--silent", action="store_true", help="no siren sound - display only")
    ap.add_argument("--glove-every", type=int, default=8, help="run 2nd gloves model every N inferences (CPU saver)")
    ap.add_argument("--glove-imgsz", type=int, default=320, help="inference size for gloves backup (320 is 3x faster than 640, fine at gate range)")
    ap.add_argument("--face-zoom", action=argparse.BooleanOptionalAction, default=True,
                    help="second close-up pass on face crop for tiny masks at 2m+ (use --no-face-zoom to disable)")
    ap.add_argument("--face-every", type=int, default=3, help="run face-zoom every N inferences (CPU saver)")
    ap.add_argument("--face-imgsz", type=int, default=256, help="inference size for the face crop")
    args = ap.parse_args()

    # ---- check profile: split stations (helmet+vest) vs (mask+gloves) ----
    CHECKS = set(x.strip().lower() for x in str(args.checks).split(",") if x.strip())
    if "all" in CHECKS:
        CHECKS = {"helmet", "vest", "mask", "gloves"}
    unknown = CHECKS - {"helmet", "vest", "mask", "gloves"}
    if unknown or not CHECKS:
        print(f"[X] --checks must be 'all' or a comma list from helmet,vest,mask,gloves (got: {args.checks})")
        return
    CHECK_MH = CHECKS == {"helmet", "mask"}
    CHECK_VG = CHECKS == {"vest", "gloves"}
    TAG = "mh" if CHECK_MH else "vg" if CHECK_VG else "all" if len(CHECKS) == 4 else "custom"

    # single-photo test path (loads models itself, no camera needed)
    if Path(str(args.source)).suffix.lower() in IMG_EXTS and Path(str(args.source)).exists():
        _m, _ids, _hm, _hg, _gm, _gg, _base = load_models(args, CHECKS)
        run_image_test(args, _m, _ids, _gm, _gg, CHECKS)
        return

    # camera FIRST: the window appears in ~2s; heavy AI models (~45s on CPU)
    # load inside the worker meanwhile, behind a LOADING banner.
    # single-instance guard: one camera, one holder per profile.
    lock_path = BASE / f"ppe_{TAG}.lock"
    if lock_path.exists():
        try:
            _old = int(lock_path.read_text().strip())
        except ValueError:
            _old = None
        if _old and _old != os.getpid() and pid_alive(_old):
            print(f"[!] Another PPE_{TAG} (pid {_old}) already holds the camera - exiting.", flush=True)
            return
    lock_path.write_text(str(os.getpid()))
    src = int(args.source) if str(args.source).isdigit() else str(args.source)
    cap = None
    for attempt in range(1, 4):
        try:
            cap = cv2.VideoCapture(src)
        except Exception as e:  # flaky USB drivers can throw on open
            print(f"[!] Camera open raised {e} (attempt {attempt}/3), retrying in 3s...",
                  flush=True)
            cap = None
            time.sleep(3)
            continue
        if cap is not None and cap.isOpened():
            break
        print(f"[!] Camera open failed (attempt {attempt}/3), retrying in 3s...", flush=True)
        time.sleep(3)
    if cap is None or not cap.isOpened():
        print(f"[X] Cannot open source {args.source} - check camera cable / RTSP URL", flush=True)
        return
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # never display stale buffered frames
    # negotiate resolution top-down: some USB cams die when asked for 1080p,
    # so verify with a real grab and step down until frames flow.
    _negotiated = False
    for _rw, _rh in ((1920, 1080), (1280, 720), (640, 480)):
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, _rw)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, _rh)
        time.sleep(0.3)
        _ok, _ = cap.read()
        if _ok:
            _negotiated = True
            break
    if not _negotiated:
        print(f"[X] Camera {args.source} opens but delivers no frames - "
              f"close other camera apps / replug USB, then Start again.", flush=True)
        cap.release()
        try:
            lock_path.unlink()
        except OSError:
            pass
        return
    _cw, _ch = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[OK] Camera opened: {args.source} ({_cw}x{_ch}) (+{time.time() - T0:.0f}s after click)",
          flush=True)

    # Shared display state: main thread captures + draws at camera rate,
    # worker thread loads AI models, then runs inference behind at ~5 Hz.
    # The live screen shows NO boxes - just a side status board (OK / NO xxx).
    shared = {
        "raw": None,
        "panel": [],  # [(label, state)] state in ok/bad/idle, profile order
        "models_ready": False,
        "hud": {"violations": 0, "persons": 0, "visitors": 0, "ok": 0,
                "tally": {}, "infer_fps": 0.0, "cam_fps": 0.0},
        "stop": False,
        "lock": threading.Lock(),
    }

    def infer_worker():
        # STAGED LOADING: primary first so masks go live fast (~20s); the slow
        # 49MB gloves backup loads in background and joins when ready (~70s).
        _gm_arg = args.gloves_model
        args.gloves_model = ""
        model, ids, HAS_MASK, HAS_GLOVES_MAIN, _, _, base = load_models(args, CHECKS)
        args.gloves_model = _gm_arg
        print("[...] Warming up model (3 dummy frames)...", flush=True)
        for _ in range(3):
            model(np.zeros((480, 480, 3), dtype=np.uint8), verbose=False)
        print("[OK] Model hot.", flush=True)

        want_gloves = ("gloves" in CHECKS) and not args.ignore_gloves and bool(_gm_arg)
        gmodel, gids = None, {}

        today = datetime.now().strftime("%Y-%m-%d")
        visitors, ok_visitors = set(), set()
        warn_state = {}  # track_id -> last counted violation (rate-limit tally)
        streak = defaultdict(lambda: defaultdict(lambda: {"v": None, "n": 0, "p": None}))
        # per-track per-item LOCKED verdicts: {"v": None|bool(locked), "n": disagree run, "p": pending init dir}.
        # Alarm flips in `smooth` frames, clear flips need `clear_after` - brief
        # misreads can never move the board, siren, or tally.
        last_logged = {}  # tid -> (raw, locked) signature: DBG prints transitions only
        seen = defaultdict(int)  # track_id -> consecutive sightings (ghost filter for visitors)
        tally = {}
        if "helmet" in CHECKS:
            tally["no_helmet"] = 0
        if "vest" in CHECKS:
            tally["no_vest"] = 0
        judge_gloves_live = ("gloves" in CHECKS) and (HAS_GLOVES_MAIN or want_gloves)
        skin_capable = ("gloves" in CHECKS) and not HAS_GLOVES_MAIN and args.skin
        if skin_capable:  # classical fallback counts as a live glove check
            judge_gloves_live = True
        if judge_gloves_live:
            tally["no_gloves"] = 0
        if HAS_MASK:
            tally["no_mask"] = 0
        tally_name = f"ppe_stats_{today}.csv" if TAG == "all" else f"ppe_{TAG}_stats_{today}.csv"
        tally_path = BASE / "logs" / tally_name
        print(f"[INFO] profile checks={sorted(CHECKS)} tally={tally_name}", flush=True)
        if tally_path.exists():  # resume today's counts across restarts
            for line in tally_path.read_text().splitlines()[1:]:
                parts = line.split(",")
                if len(parts) == 2 and parts[0] in tally:
                    tally[parts[0]] = int(parts[1])
        with shared["lock"]:
            shared["models_ready"] = True
        print(f"[OK] masks live (+{time.time() - T0:.0f}s after click)", flush=True)
        if want_gloves and not HAS_GLOVES_MAIN:
            # slow 49MB backup loads now, in background: masks already judging.
            try:
                print("[...] loading gloves backup in background (masks already live)...", flush=True)
                _gpath = BASE / _gm_arg if not Path(_gm_arg).exists() else Path(_gm_arg)
                gmodel = YOLO(str(_gpath))
                gids = resolve_ids(gmodel)
                if gids["GLOVES"] is None and gids["NO_GLOVES"] is None:
                    print("[!] gloves backup has no glove classes - gloves stay off", flush=True)
                    gmodel = None
                else:
                    print(f"[OK] gloves backup live (+{time.time() - T0:.0f}s after click)", flush=True)
            except Exception as e:
                print(f"[!] gloves backup failed: {e}", flush=True)
                gmodel = None
        tracker_cfg = str(TRACKER_YAML) if TRACKER_YAML.exists() else "bytetrack.yaml"
        last_glove = []  # cached [(box, conf, cls)] from backup model
        last_face = ([], [])  # cached ([(box, conf)], [(box, conf)]) from face-zoom
        last_skin = []  # cached [(box, 0.90)] bare-hand pseudo-detections
        last_blue = []  # cached [(box, 0.85)] calibrated-blue glove hits
        blue_spec = None
        _hspec = BASE / "glove_hsv.json"
        if _hspec.exists() and ("gloves" in CHECKS):
            try:
                blue_spec = json.loads(_hspec.read_text())
                print(f"[INFO] calibrated glove color active (h={blue_spec['h']:.0f})", flush=True)
            except Exception as e:
                print(f"[!] glove_hsv.json unreadable: {e}", flush=True)
                blue_spec = None
        glove_mem = {"has": 0, "flag": 0}  # countdowns: glove evidence seen recently
        frame_n = 0
        alarm_on, last_siren = False, 0.0
        it0, itn, infer_fps = time.time(), 0, 0.0
        perf_t0 = time.time()
        glove_absence = HAS_GLOVES_MAIN  # backup-only gloves: explicit NO-Gloves only
        mask_name = model.names[ids["MASK"]] if ids["MASK"] is not None else "Mask"
        nomask_name = model.names[ids["NO_MASK"]] if ids["NO_MASK"] is not None else "NO-Mask"
        # auto-throttle: weak CPUs stay responsive by spacing out the extras
        face_every_cur, glove_every_cur = args.face_every, args.glove_every
        ema_dt, throttle_t0, throttled = 0.15, time.time(), False
        try:
            while not shared["stop"]:
                with shared["lock"]:
                    grab = None if shared["raw"] is None else shared["raw"].copy()
                if grab is None:
                    time.sleep(0.02)
                    continue
                frame_n += 1
                inf_t0 = time.time()
                try:
                    r = model.track(grab, persist=True, imgsz=args.imgsz, conf=base,
                                    verbose=False, tracker=tracker_cfg)[0]
                except Exception as e:
                    print(f"[!] track skipped: {e}", flush=True)
                    time.sleep(0.05)
                    continue

                persons, hats, nohats, vests, novests = [], [], [], [], []
                gloves, nogloves, masks, nomasks, face_boxes = [], [], [], [], []
                # NOTE: no overlay boxes are drawn live - the screen shows only
                # the side status board. PPE lists below feed judging only.
                if r.boxes is not None and len(r.boxes) > 0:
                    has_ids = r.boxes.id is not None
                    ids_arr = r.boxes.id.cpu().numpy().astype(int) if has_ids else \
                        [-1000 - i for i in range(len(r.boxes))]
                    for box, c, cf, tid in zip(r.boxes.xyxy.cpu().numpy(),
                                               r.boxes.cls.cpu().numpy().astype(int),
                                               r.boxes.conf.cpu().numpy(), ids_arr):
                        cf = float(cf)
                        if not box_passes(c, cf, args, ids):
                            continue
                        if c in (ids["MASK"], ids["NO_MASK"]):
                            face_boxes.append([float(v) for v in box])  # for skin exclusion
                        # profile gate: a station only ever SHOWS its own items
                        if c in (ids["HARDHAT"], ids["NO_HARDHAT"]) and "helmet" not in CHECKS:
                            continue
                        if c in (ids["VEST"], ids["NO_VEST"]) and "vest" not in CHECKS:
                            continue
                        if c in (ids["MASK"], ids["NO_MASK"]) and "mask" not in CHECKS:
                            continue
                        if c in (ids["GLOVES"], ids["NO_GLOVES"]) and "gloves" not in CHECKS:
                            continue
                        b = [float(v) for v in box]
                        if ids["PERSON"] is not None and c == ids["PERSON"]:
                            persons.append((b, int(tid), cf))
                        else:
                            if ids["HARDHAT"] is not None and c == ids["HARDHAT"]:
                                hats.append((b, cf))
                            elif ids["NO_HARDHAT"] is not None and c == ids["NO_HARDHAT"]:
                                nohats.append((b, cf))
                            elif ids["VEST"] is not None and c == ids["VEST"]:
                                vests.append((b, cf))
                            elif ids["NO_VEST"] is not None and c == ids["NO_VEST"]:
                                novests.append((b, cf))
                            elif ids["GLOVES"] is not None and c == ids["GLOVES"]:
                                gloves.append((b, cf))
                            elif ids["NO_GLOVES"] is not None and c == ids["NO_GLOVES"]:
                                nogloves.append((b, cf))
                            elif ids["MASK"] is not None and c == ids["MASK"]:
                                masks.append((b, cf))
                            elif ids["NO_MASK"] is not None and c == ids["NO_MASK"]:
                                nomasks.append((b, cf))

                # gloves-only backup (its person/helmet outputs are ignored -
                # that weight is blind there). Gated on persons present.
                if gmodel is not None and persons and frame_n % glove_every_cur == 0:
                    try:
                        rg = gmodel(grab, conf=args.glove_conf, imgsz=args.glove_imgsz,
                                    verbose=False)[0]
                        gb = []
                        if rg.boxes is not None and len(rg.boxes) > 0:
                            for box, c, cf in zip(rg.boxes.xyxy.cpu().numpy(),
                                                  rg.boxes.cls.cpu().numpy().astype(int),
                                                  rg.boxes.conf.cpu().numpy()):
                                cf = float(cf)
                                if c not in (gids.get("GLOVES"), gids.get("NO_GLOVES")) \
                                        or cf < args.glove_conf:
                                    continue
                                gb.append(([float(v) for v in box], cf, c))
                        last_glove = gb
                    except Exception as e:
                        print(f"[!] gloves-model frame skipped: {e}", flush=True)
                if gmodel is not None:
                    for (b, cf, c) in last_glove:
                        (gloves if c == gids.get("GLOVES") else nogloves).append((b, cf))

                # skin fallback: bare hands -> NO-Gloves pseudo-detections.
                # Runs only while no glove-capable model is loaded.
                skin_active = skin_capable and gmodel is None
                if skin_active and persons and frame_n % args.skin_every == 0:
                    try:
                        sb = []
                        for (pbox, _tid, _pc) in persons:
                            if pbox[3] - pbox[1] < args.min_glove_h:
                                continue
                            sb += bare_hand_boxes(grab, pbox, face_boxes)
                        last_skin = sb
                    except Exception as e:
                        print(f"[!] skin check skipped: {e}", flush=True)
                if skin_active:
                    nogloves = list(nogloves) + list(last_skin)

                # calibrated blue: YOUR glove color -> positive glove evidence.
                if blue_spec is not None and persons and frame_n % args.skin_every == 0:
                    try:
                        bb = []
                        for (pbox, _tid, _pc) in persons:
                            if pbox[3] - pbox[1] < args.min_glove_h:
                                continue
                            bb += blue_glove_boxes(grab, pbox, face_boxes, blue_spec)
                        last_blue = bb
                    except Exception as e:
                        print(f"[!] blue-glove check skipped: {e}", flush=True)
                if blue_spec is not None:
                    gloves = list(gloves) + list(last_blue)

                # face-zoom second look for tiny masks (the 2m+ fix)
                if args.face_zoom and not args.ignore_mask and persons \
                        and frame_n % face_every_cur == 0:
                    try:
                        last_face = face_zoom_boxes(grab, persons, model, ids, args)
                    except Exception as e:
                        print(f"[!] face-zoom frame skipped: {e}", flush=True)
                if args.face_zoom and not args.ignore_mask:
                    masks += last_face[0]
                    nomasks += last_face[1]
                    face_boxes += [b for b, _ in last_face[0]] + [b for b, _ in last_face[1]]

                # dedup: same object detected twice (seen live: double NO-Mask
                # boxes flickering with alternating track IDs)
                persons = dedup_persons(persons)
                hats = dedup_scored(hats)
                nohats = dedup_scored(nohats)
                vests = dedup_scored(vests)
                novests = dedup_scored(novests)
                gloves = dedup_scored(gloves)
                nogloves = dedup_scored(nogloves)
                masks = dedup_scored(masks)
                nomasks = dedup_scored(nomasks)

                violations = 0
                all_bad = set()  # confirmed violation tags across everyone in view
                mask_judged_any, glove_judged_any = False, False
                g_has_any, g_flag_any = False, False  # instant glove evidence, any person
                for pbox, tid, pconf in persons:
                    ph = pbox[3] - pbox[1]
                    judge_mask = HAS_MASK and ph >= args.min_face_h
                    judge_glove = judge_gloves_live and ph >= args.min_glove_h
                    mask_judged_any = mask_judged_any or judge_mask
                    glove_judged_any = glove_judged_any or judge_glove
                    raw_tags, _, _confs = judge_person(pbox, hats, nohats, vests, novests, gloves, nogloves,
                                               masks, nomasks, judge_mask, judge_glove,
                                               glove_absence_counts=glove_absence)
                    if judge_glove:  # close enough to read hands: record what was seen
                        if _confs["gloves"][0] > 0:
                            g_has_any = True
                        if _confs["gloves"][1] > 0:
                            g_flag_any = True
                    if "helmet" not in CHECKS:  # this station doesn't judge helmets
                        raw_tags = [t for t in raw_tags if t != "NO HELMET"]
                    if "vest" not in CHECKS:  # this station doesn't judge vests
                        raw_tags = [t for t in raw_tags if t != "NO VEST"]
                    # per-item locks: instant raw verdict -> locked verdict.
                    # init needs `smooth` bad frames (or 2 good) to set; a locked
                    # verdict flips only after `smooth` bad / `clear_after` good
                    # frames in a row. Flicker dies here, not on the board.
                    # (raw_tags are already profile-filtered upstream.)
                    locked_tags = []
                    for _item, _tag, _judged in (("helmet", "NO HELMET", True),
                                                 ("vest", "NO VEST", True),
                                                 ("mask", "NO MASK", judge_mask),
                                                 ("gloves", "NO GLOVES", judge_glove)):
                        _cur = (_tag in raw_tags) and _judged
                        _lk = streak[int(tid)][_item]
                        if _lk["v"] is None:
                            if _lk["p"] == _cur:
                                _lk["n"] += 1
                            else:
                                _lk["p"], _lk["n"] = _cur, 1
                            _need = args.smooth if _cur else 2
                            if _lk["n"] >= _need:
                                _lk["v"], _lk["n"], _lk["p"] = _cur, 0, None
                            _verdict = _lk["v"] if _lk["v"] is not None else _cur
                        else:
                            if _cur == _lk["v"]:
                                _lk["n"] = 0
                            else:
                                _lk["n"] += 1
                                _need = args.smooth if _cur else args.clear_after
                                if _lk["n"] >= _need:
                                    _lk["v"], _lk["n"] = _cur, 0
                            _verdict = _lk["v"]
                        if _verdict:
                            locked_tags.append(_tag)
                    confirmed = locked_tags
                    _sig = (tuple(raw_tags), tuple(confirmed))
                    if last_logged.get(int(tid)) != _sig:  # transition only: the flicker record
                        last_logged[int(tid)] = _sig
                        _c = {k: (round(a, 2), round(b, 2)) for k, (a, b) in _confs.items()}
                        print(f"[DBG] tid={tid} ph={int(pbox[3] - pbox[1])} pc={round(pconf, 2)} "
                              f"raw={raw_tags} locked={confirmed} "
                              f"hat={_c['helmet']} vest={_c['vest']} "
                              f"mask={_c['mask']} glove={_c['gloves']}", flush=True)
                    if confirmed:
                        all_bad.update(confirmed)
                    far_note = "FAR" if (HAS_MASK and ph < args.min_face_h) else ""
                    if confirmed:
                        violations += 1
                    seen[int(tid)] += 1
                    if len(seen) > 2000:
                        seen.clear()
                    if int(tid) >= 0 and seen[int(tid)] >= 3:  # stable track only, no ghosts
                        visitors.add(int(tid))
                    if confirmed:
                        vtype = "+".join(confirmed)
                        if warn_state.get(int(tid)) != vtype and int(tid) >= 0:  # once per episode
                            warn_state[int(tid)] = vtype
                            if "NO HELMET" in confirmed and "no_helmet" in tally:
                                tally["no_helmet"] += 1
                            if "NO VEST" in confirmed and "no_vest" in tally:
                                tally["no_vest"] += 1
                            if "NO GLOVES" in confirmed and "no_gloves" in tally:
                                tally["no_gloves"] += 1
                            if "NO MASK" in confirmed and "no_mask" in tally:
                                tally["no_mask"] += 1
                            save_tally(tally_path, visitors, ok_visitors, tally)
                            print(f"[WARNING] ID:{tid} {vtype}", flush=True)
                    elif not raw_tags:
                        if int(tid) >= 0:
                            ok_visitors.add(int(tid))
                            warn_state.pop(int(tid), None)

                # side status board: overall state per checked item (profile order).
                # Too-far faces are idle ("--"), never a false green OK.
                # Backup-model gloves get honest three-state: the model recall is
                # too poor for absence to mean anything, so YES needs a glove
                # actually seen in the last ~second, else "--".
                if g_has_any:
                    glove_mem["has"] = 5
                elif last_blue:
                    glove_mem["has"] = 5  # calibrated blue seen: green-able
                else:
                    glove_mem["has"] = max(0, glove_mem["has"] - 1)
                if g_flag_any:
                    glove_mem["flag"] = 5
                else:
                    glove_mem["flag"] = max(0, glove_mem["flag"] - 1)
                panel = []
                for _key, _label, _bad, _judged in (("no_helmet", "HELMET", "NO HELMET", True),
                                                    ("no_vest", "VEST", "NO VEST", True),
                                                    ("no_mask", "MASK", "NO MASK", mask_judged_any),
                                                    ("no_gloves", "GLOVES", "NO GLOVES", glove_judged_any)):
                    if _key not in tally:
                        # vg profile with no glove-capable weights yet: show the
                        # row as idle ("--") so the station promise stays visible
                        if _key == "no_gloves" and "gloves" in CHECKS:
                            panel.append((_label, "idle"))
                        continue
                    if _key == "no_gloves" and not HAS_GLOVES_MAIN:
                        if _bad in all_bad:
                            panel.append((_label, "bad"))
                        elif glove_mem["has"] > 0:
                            panel.append((_label, "ok"))
                        else:
                            panel.append((_label, "idle"))
                        continue
                    if not persons or not _judged:
                        panel.append((_label, "idle"))
                    elif _bad in all_bad:
                        panel.append((_label, "bad"))
                    else:
                        panel.append((_label, "ok"))

                now = time.time()
                itn += 1
                if now - it0 >= 1.0:
                    infer_fps = itn / (now - it0)
                    itn, it0 = 0, now
                if now - perf_t0 >= 10.0:
                    print(f"[PERF] infer {infer_fps:.1f}Hz persons={len(persons)} "
                          f"panel={[(l, s) for l, s in panel]} visitors={len(visitors)}", flush=True)
                    # auto-throttle: extras (zoom/gloves) back off when the CPU can't
                    # keep up, recover when it can. Hysteresis via 10s windows.
                    if ema_dt > 0.40 and not throttled:
                        face_every_cur, glove_every_cur = args.face_every * 2, args.glove_every * 2
                        throttled = True
                        print(f"[PERF] CPU struggling ({ema_dt * 1000:.0f}ms/inf) -> "
                              f"zoom every {face_every_cur}, gloves every {glove_every_cur}", flush=True)
                    elif ema_dt < 0.20 and throttled:
                        face_every_cur, glove_every_cur = args.face_every, args.glove_every
                        throttled = False
                        print("[PERF] CPU recovered -> full extras rate", flush=True)
                    perf_t0 = now

                if violations > 0:
                    if not args.silent and winsound is not None and ALARM_WAV.exists():
                        if (not alarm_on) or (now - last_siren > 4.0):
                            winsound.PlaySound(str(ALARM_WAV),
                                               winsound.SND_FILENAME | winsound.SND_ASYNC | winsound.SND_LOOP)
                            if not alarm_on:
                                print("[ALARM] siren ON", flush=True)
                            last_siren, alarm_on = now, True
                else:
                    if alarm_on and not args.silent and winsound is not None:
                        winsound.PlaySound(None, winsound.SND_PURGE)
                        print("[ALARM] siren OFF", flush=True)
                        alarm_on = False

                with shared["lock"]:
                    cam = shared["hud"]["cam_fps"]
                    shared["panel"] = panel
                    shared["hud"] = {"violations": violations, "persons": len(persons),
                                     "visitors": len(visitors), "ok": len(ok_visitors),
                                     "tally": dict(tally), "infer_fps": infer_fps, "cam_fps": cam}
                ema_dt = 0.9 * ema_dt + 0.1 * (time.time() - inf_t0)
        except Exception:
            traceback.print_exc()
            shared["stop"] = True
        finally:
            if alarm_on and winsound is not None:
                try:
                    winsound.PlaySound(None, winsound.SND_PURGE)
                except Exception:
                    pass
            save_tally(tally_path, visitors, ok_visitors, tally)
            print(f"[OK] AI worker ended | visitors={len(visitors)} ok={len(ok_visitors)} " +
                  " ".join(f"{k}={v}" for k, v in tally.items()), flush=True)

    worker = threading.Thread(target=infer_worker, daemon=True)
    worker.start()
    print(f"[OK] PPE_Check running on {args.source} | video=full camera rate, AI overlay ~5 Hz. "
          f"Press Q to quit.", flush=True)

    ct0, ctn, cam_fps = time.time(), 0, 0.0
    first_frame = True
    dead_n = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            dead_n += 1
            if dead_n > 150:  # ~5s of dead stream: exit loudly instead of spinning "stuck"
                print("[X] Camera stream died (5s, no frames) - exiting. "
                      "Check USB cable / close other camera apps, then Start again.", flush=True)
                shared["stop"] = True
                break
            time.sleep(0.02)
            continue
        dead_n = 0
        with shared["lock"]:
            shared["raw"] = frame  # full-res for the AI crops
            panel = list(shared["panel"])
            hud = shared["hud"]
            hud_tally = dict(hud["tally"])
            ready = shared["models_ready"]
        # display copy: capped at 960 wide so 1080p doesn't make a giant window.
        # Overlay metrics scale with display height (fs=1 at 480p).
        fh0, fw0 = frame.shape[:2]
        if fw0 > 960:
            frame = cv2.resize(frame, (960, int(fh0 * 960 / fw0)))
        fh, fw = frame.shape[:2]
        fs = fh / 480.0
        _fs = lambda base: base * fs
        _th = lambda base: max(1, int(round(base * fs)))
        if not ready:  # models still loading: live video + banner, AI states pop in later
            cv2.rectangle(frame, (8, int(88 * fs)), (int(330 * fs), int(118 * fs)), (25, 25, 25), -1)
            cv2.putText(frame, "LOADING AI MODELS...", (16, int(109 * fs)),
                        cv2.FONT_HERSHEY_SIMPLEX, _fs(0.6), (255, 255, 255), _th(2))
        v = hud["violations"]
        pill_color = (0, 0, 210) if v > 0 else (0, 170, 0)
        cv2.rectangle(frame, (8, 8), (int(248 * fs), int(38 * fs)), pill_color, -1)
        cv2.circle(frame, (int(26 * fs), int(23 * fs)), _th(6), (255, 255, 255), -1)
        cv2.putText(frame, "WARNING" if v > 0 else "ALL COMPLIANT", (int(40 * fs), int(28 * fs)),
                    cv2.FONT_HERSHEY_SIMPLEX, _fs(0.6), (255, 255, 255), _th(2))
        # bottom status bar: one dark strip, dot + LABEL + YES/NO per item.
        # green = worn, red = missing, gray = nobody in view. Tallies live on
        # the dashboard, not on the video.
        _bh = int(32 * fs)
        cv2.rectangle(frame, (0, fh - _bh), (fw, fh), (20, 20, 20), -1)
        bx = int(12 * fs)
        for (plabel, state) in panel:
            if state == "ok":
                dot, stxt, scol = (0, 200, 0), "YES", (0, 210, 0)
            elif state == "bad":
                dot, stxt, scol = (0, 0, 255), "NO", (0, 0, 255)
            else:
                dot, stxt, scol = (130, 130, 130), "--", (150, 150, 150)
            cv2.circle(frame, (int(bx + 6 * fs), fh - _bh // 2), _th(6), dot, -1)
            cv2.putText(frame, plabel, (int(bx + 17 * fs), fh - int(11 * fs)),
                        cv2.FONT_HERSHEY_SIMPLEX, _fs(0.5), (235, 235, 235), _th(1))
            (tw, _thx), _ = cv2.getTextSize(plabel, cv2.FONT_HERSHEY_SIMPLEX, _fs(0.5), _th(1))
            sx = bx + 17 * fs + tw + 8 * fs
            cv2.putText(frame, stxt, (int(sx), fh - int(11 * fs)),
                        cv2.FONT_HERSHEY_SIMPLEX, _fs(0.5), scol, _th(1))
            (tws, _ths), _ = cv2.getTextSize(stxt, cv2.FONT_HERSHEY_SIMPLEX, _fs(0.5), _th(1))
            bx = int(sx + tws + 24 * fs)
        cv2.putText(frame, f"{hud['persons']} in view", (int(bx), fh - int(11 * fs)),
                    cv2.FONT_HERSHEY_SIMPLEX, _fs(0.5), (235, 235, 235), _th(1))
        fps_txt = f"AI {hud['infer_fps']:.0f} + CAM {cam_fps:.0f} FPS"
        (twf, _thf), _ = cv2.getTextSize(fps_txt, cv2.FONT_HERSHEY_SIMPLEX, _fs(0.5), _th(1))
        cv2.putText(frame, fps_txt, (int(fw - twf - 12 * fs), fh - int(11 * fs)),
                    cv2.FONT_HERSHEY_SIMPLEX, _fs(0.5), (160, 160, 160), _th(1))

        now2 = time.time()
        ctn += 1
        if now2 - ct0 >= 1.0:
            cam_fps = ctn / (now2 - ct0)
            ctn, ct0 = 0, now2
            with shared["lock"]:
                shared["hud"]["cam_fps"] = cam_fps

        cv2.imshow(f"PPE_{TAG.upper()} - Q to quit", frame)
        if first_frame:
            print(f"[OK] First video frame shown (+{time.time() - T0:.0f}s after click)", flush=True)
            first_frame = False
        if cv2.waitKey(1) & 0xFF in (ord("q"), ord("Q")):
            shared["stop"] = True
            break

    worker.join(timeout=8)
    cap.release()
    cv2.destroyAllWindows()
    try:
        lock_path.unlink()
    except OSError:
        pass
    print("[OK] PPE session ended.", flush=True)


if __name__ == "__main__":
    main()

# FINE-TUNE NOTE (yellow V-44 mask / blue gloves, Phase 2):
# These weights never saw your exact PPE colors, so thresholds alone may not flip
# NO-Mask -> Mask. The real fix is 15 min of labeling + a free Colab GPU run:
#   1. Collect 50-100 phone photos of the yellow mask (and blue gloves) at 2-3 m,
#      varied angles/light; include worn + not-worn.
#   2. Label with labelImg / Roboflow (classes: Mask, NO-Mask, Gloves, NO-Gloves,
#      Hardhat, NO-Hardhat, Safety Vest, NO-Safety Vest, Person).
#   3. Colab: pip install ultralytics, then:
#        yolo detect train model=yolo26n.pt data=data.yaml epochs=80 imgsz=640
#      copy the resulting best.pt back as e.g. ppe_yellow.pt and run:
#        python mode_ppe.py --source 0 --model ppe_yellow.pt
