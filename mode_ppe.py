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
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def expand_person(pbox, top=0.25, side=0.10, bottom=0.05):
    """Expand a person box so helmets above the head and hands at the edge count."""
    x1, y1, x2, y2 = (float(v) for v in pbox)
    pw, ph = max(1.0, x2 - x1), max(1.0, y2 - y1)
    return (x1 - pw * side, y1 - ph * top, x2 + pw * side, y2 + ph * bottom)


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


def judge_person(pbox, hats, nohats, vests, novests, gloves, nogloves,
                 masks, nomasks, judge_mask, judge_gloves, glove_absence_counts=True):
    """Return (tags, detail). Absence alone only counts when close + persistent
    (persistence is handled by the caller streak); explicit NO-* always counts.
    glove_absence_counts=False when gloves come only from the backup model
    (its recall is poor, so absence means 'can't tell', not 'no gloves')."""
    has_hat = any(associated(b, pbox) for b in hats)
    flag_nohat = any(associated(b, pbox) for b in nohats)
    has_vest = any(associated(b, pbox) for b in vests)
    flag_novest = any(associated(b, pbox) for b in novests)
    has_mask = any(associated(b, pbox) for b in masks)
    flag_nomask = any(associated(b, pbox) for b in nomasks)
    has_gloves = any(associated(b, pbox) for b in gloves)
    flag_nogloves = any(associated(b, pbox) for b in nogloves)
    tags = []
    if flag_nohat or not has_hat:
        tags.append("NO HELMET")
    if flag_novest or not has_vest:
        tags.append("NO VEST")
    if judge_gloves and (flag_nogloves or ((not has_gloves) and glove_absence_counts)):
        tags.append("NO GLOVES")
    if judge_mask and (flag_nomask or not has_mask):
        tags.append("NO MASK")
    return tags, (has_hat, has_vest, has_mask, has_gloves)


# ---------------------------------------------------------------- image test
def run_image_test(args, model, ids, gmodel, gids):
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
            b = [float(v) for v in box]
            draw_ppe_box(frame, b, c, float(cf), model.names)
            if ids["PERSON"] is not None and c == ids["PERSON"]:
                persons.append((b, float(cf)))
            elif ids["HARDHAT"] is not None and c == ids["HARDHAT"]:
                hats.append(b)
            elif ids["NO_HARDHAT"] is not None and c == ids["NO_HARDHAT"]:
                nohats.append(b)
            elif ids["VEST"] is not None and c == ids["VEST"]:
                vests.append(b)
            elif ids["NO_VEST"] is not None and c == ids["NO_VEST"]:
                novests.append(b)
            elif ids["GLOVES"] is not None and c == ids["GLOVES"]:
                gloves.append(b)
            elif ids["NO_GLOVES"] is not None and c == ids["NO_GLOVES"]:
                nogloves.append(b)
            elif ids["MASK"] is not None and c == ids["MASK"]:
                masks.append(b)
            elif ids["NO_MASK"] is not None and c == ids["NO_MASK"]:
                nomasks.append(b)
    # optional gloves-only backup model
    if gmodel is not None and not args.ignore_gloves:
        rg = gmodel(frame, conf=args.glove_conf, imgsz=args.glove_imgsz, verbose=False)[0]
        if rg.boxes is not None and len(rg.boxes) > 0:
            for box, c, cf in zip(rg.boxes.xyxy.cpu().numpy(), rg.boxes.cls.cpu().numpy().astype(int),
                                  rg.boxes.conf.cpu().numpy()):
                if c not in (gids["GLOVES"], gids["NO_GLOVES"]) or float(cf) < args.glove_conf:
                    continue
                b = [float(v) for v in box]
                (gloves if c == gids["GLOVES"] else nogloves).append(b)
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
        masks += [b for b, _ in zm]
        nomasks += [b for b, _ in zn]
        print(f"       face-zoom added: mask={len(zm)} nomask={len(zn)}")
    if not persons:
        print("[TEST] NO Person box kept -> try --person-conf 0.25 --imgsz 640, move to 2m, center chest-up.")
    for i, (pbox, pconf) in enumerate(persons):
        ph = pbox[3] - pbox[1]
        judge_mask = not args.ignore_mask and (ids["MASK"] is not None or ids["NO_MASK"] is not None) \
            and ph >= args.min_face_h
        judge_gloves = not args.ignore_gloves and (gmodel is not None or ids["GLOVES"] is not None) \
            and ph >= args.min_glove_h
        tags, _ = judge_person(pbox, hats, nohats, vests, novests, gloves, nogloves,
                               masks, nomasks, judge_mask, judge_gloves,
                               glove_absence_counts=(ids["GLOVES"] is not None))
        x1, y1, x2, y2 = map(int, pbox)
        color = (0, 255, 0) if not tags else (0, 0, 255)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, "OK" if not tags else "WARNING " + "+".join(tags), (x1, y1 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        extra = []
        if not judge_mask:
            extra.append(f"mask skipped (h={ph:.0f}<{args.min_face_h})")
        if (ids["GLOVES"] is None and gmodel is None) or args.ignore_gloves:
            extra.append("gloves n/a")
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
    ap.add_argument("--gloves-model", default="", help="optional 2nd weights for gloves only, e.g. ppe_v8m.pt")
    ap.add_argument("--camera", default="CAM01")
    ap.add_argument("--imgsz", type=int, default=480, help="main pass size (480 = sweet spot: big items stay accurate, ~35%% faster than 640)")
    ap.add_argument("--conf", type=float, default=0.25, help="legacy global floor (inference uses the min of all thresholds)")
    ap.add_argument("--person-conf", type=float, default=0.35)
    ap.add_argument("--ppe-conf", type=float, default=0.30, help="helmet/vest + explicit NO-* boxes")
    ap.add_argument("--mask-conf", type=float, default=0.20, help="mask boxes are tiny -> keep low")
    ap.add_argument("--glove-conf", type=float, default=0.20, help="glove boxes are tiny -> keep low")
    ap.add_argument("--smooth", type=int, default=3, help="consecutive frames before a warning counts (kills flicker)")
    ap.add_argument("--min-face-h", type=int, default=130, help="px person height below which mask is NOT judged (FAR)")
    ap.add_argument("--min-glove-h", type=int, default=180, help="px person height below which gloves are NOT judged (FAR)")
    ap.add_argument("--ignore-mask", action="store_true", help="turn off mask checking (far-field cams)")
    ap.add_argument("--ignore-gloves", action="store_true", help="turn off glove checking")
    ap.add_argument("--glove-every", type=int, default=8, help="run 2nd gloves model every N inferences (CPU saver)")
    ap.add_argument("--glove-imgsz", type=int, default=320, help="inference size for gloves backup (320 is 3x faster than 640, fine at gate range)")
    ap.add_argument("--face-zoom", action=argparse.BooleanOptionalAction, default=True,
                    help="second close-up pass on face crop for tiny masks at 2m+ (use --no-face-zoom to disable)")
    ap.add_argument("--face-every", type=int, default=3, help="run face-zoom every N inferences (CPU saver)")
    ap.add_argument("--face-imgsz", type=int, default=256, help="inference size for the face crop")
    args = ap.parse_args()

    wpath = BASE / args.model if not Path(args.model).exists() else Path(args.model)
    model = YOLO(str(wpath))
    ids = resolve_ids(model)
    HAS_MASK = (ids["MASK"] is not None or ids["NO_MASK"] is not None) and not args.ignore_mask
    HAS_GLOVES_MAIN = (ids["GLOVES"] is not None or ids["NO_GLOVES"] is not None) and not args.ignore_gloves

    gmodel, gids = None, {}
    if args.gloves_model and not args.ignore_gloves:
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
            print(f"[!] --gloves-model {args.gloves_model} not found - gloves from primary model only", flush=True)

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
        print("[INFO] primary model has no glove classes -> glove check OFF "
              "(enable with --gloves-model ppe_v8m.pt at 2-3 m gate)", flush=True)

    # single-photo test path (no camera needed)
    if Path(str(args.source)).suffix.lower() in IMG_EXTS and Path(str(args.source)).exists():
        run_image_test(args, model, ids, gmodel, gids)
        return

    print("[...] Warming up model (3 dummy frames)...", flush=True)
    for _ in range(3):
        model(np.zeros((480, 480, 3), dtype=np.uint8), verbose=False)
    print("[OK] Model hot.", flush=True)

    src = int(args.source) if str(args.source).isdigit() else str(args.source)
    cap = None
    for attempt in range(1, 4):
        cap = cv2.VideoCapture(src)
        if cap.isOpened():
            break
        print(f"[!] Camera open failed (attempt {attempt}/3), retrying in 3s...", flush=True)
        time.sleep(3)
    if cap is None or not cap.isOpened():
        print(f"[X] Cannot open source {args.source} - check camera cable / RTSP URL", flush=True)
        return
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # never display stale buffered frames
    print(f"[OK] Camera opened: {args.source}", flush=True)

    today = datetime.now().strftime("%Y-%m-%d")
    visitors, ok_visitors = set(), set()
    warn_state = {}  # track_id -> last counted violation (rate-limit tally)
    streak = defaultdict(lambda: {"tags": "", "n": 0})  # consecutive-frame smoothing
    tally = {"no_helmet": 0, "no_vest": 0}
    judge_gloves_live = (HAS_GLOVES_MAIN or gmodel is not None)
    if judge_gloves_live:
        tally["no_gloves"] = 0
    if HAS_MASK:
        tally["no_mask"] = 0
    tally_path = BASE / "logs" / f"ppe_stats_{today}.csv"
    if tally_path.exists():  # resume today's counts across restarts
        for line in tally_path.read_text().splitlines()[1:]:
            parts = line.split(",")
            if len(parts) == 2 and parts[0] in tally:
                tally[parts[0]] = int(parts[1])

    # Shared display state: main thread captures + draws at camera rate,
    # worker thread runs AI behind at ~10 Hz and publishes overlay boxes.
    # items = [(x1, y1, x2, y2, color_bgr, label, thick)]
    shared = {
        "raw": None,
        "items": [],
        "hud": {"violations": 0, "persons": 0, "visitors": 0, "ok": 0,
                "tally": dict(tally), "infer_fps": 0.0, "cam_fps": 0.0},
        "stop": False,
        "lock": threading.Lock(),
    }

    def infer_worker():
        last_glove = []  # cached [(box, conf, cls)] from backup model
        last_face = ([], [])  # cached ([(box, conf)], [(box, conf)]) from face-zoom
        frame_n = 0
        alarm_on, last_siren = False, 0.0
        it0, itn, infer_fps = time.time(), 0, 0.0
        glove_absence = HAS_GLOVES_MAIN  # backup-only gloves: explicit NO-Gloves only
        mask_name = model.names[ids["MASK"]] if ids["MASK"] is not None else "Mask"
        nomask_name = model.names[ids["NO_MASK"]] if ids["NO_MASK"] is not None else "NO-Mask"
        try:
            while not shared["stop"]:
                with shared["lock"]:
                    grab = None if shared["raw"] is None else shared["raw"].copy()
                if grab is None:
                    time.sleep(0.02)
                    continue
                frame_n += 1
                try:
                    r = model.track(grab, persist=True, imgsz=args.imgsz, conf=base,
                                    verbose=False, tracker="bytetrack.yaml")[0]
                except Exception as e:
                    print(f"[!] track skipped: {e}", flush=True)
                    time.sleep(0.05)
                    continue

                items = []
                persons, hats, nohats, vests, novests = [], [], [], [], []
                gloves, nogloves, masks, nomasks = [], [], [], []
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
                        b = [float(v) for v in box]
                        if ids["PERSON"] is not None and c == ids["PERSON"]:
                            persons.append((b, int(tid), cf))
                        else:
                            items.append((int(b[0]), int(b[1]), int(b[2]), int(b[3]),
                                          BOX_COLORS.get(c, (255, 255, 0)),
                                          f"{model.names[c]} {cf:.2f}", 1))
                            if ids["HARDHAT"] is not None and c == ids["HARDHAT"]:
                                hats.append(b)
                            elif ids["NO_HARDHAT"] is not None and c == ids["NO_HARDHAT"]:
                                nohats.append(b)
                            elif ids["VEST"] is not None and c == ids["VEST"]:
                                vests.append(b)
                            elif ids["NO_VEST"] is not None and c == ids["NO_VEST"]:
                                novests.append(b)
                            elif ids["GLOVES"] is not None and c == ids["GLOVES"]:
                                gloves.append(b)
                            elif ids["NO_GLOVES"] is not None and c == ids["NO_GLOVES"]:
                                nogloves.append(b)
                            elif ids["MASK"] is not None and c == ids["MASK"]:
                                masks.append(b)
                            elif ids["NO_MASK"] is not None and c == ids["NO_MASK"]:
                                nomasks.append(b)

                # gloves-only backup (its person/helmet outputs are ignored -
                # that weight is blind there). Gated on persons present.
                if gmodel is not None and persons and frame_n % args.glove_every == 0:
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
                        (gloves if c == gids.get("GLOVES") else nogloves).append(b)
                        items.append((int(b[0]), int(b[1]), int(b[2]), int(b[3]), (255, 0, 0),
                                      f"{gmodel.names[c]} {cf:.2f} (glove-m)", 1))

                # face-zoom second look for tiny masks (the 2m+ fix)
                if args.face_zoom and not args.ignore_mask and persons \
                        and frame_n % args.face_every == 0:
                    try:
                        last_face = face_zoom_boxes(grab, persons, model, ids, args)
                    except Exception as e:
                        print(f"[!] face-zoom frame skipped: {e}", flush=True)
                if args.face_zoom and not args.ignore_mask:
                    for (b, cf) in last_face[0]:
                        masks.append(b)
                        items.append((int(b[0]), int(b[1]), int(b[2]), int(b[3]), (0, 255, 255),
                                      f"{mask_name} {cf:.2f} (zoom)", 1))
                    for (b, cf) in last_face[1]:
                        nomasks.append(b)
                        items.append((int(b[0]), int(b[1]), int(b[2]), int(b[3]), (0, 255, 255),
                                      f"{nomask_name} {cf:.2f} (zoom)", 1))

                violations = 0
                for pbox, tid, pconf in persons:
                    ph = pbox[3] - pbox[1]
                    judge_mask = HAS_MASK and ph >= args.min_face_h
                    judge_glove = judge_gloves_live and ph >= args.min_glove_h
                    raw_tags, _ = judge_person(pbox, hats, nohats, vests, novests, gloves, nogloves,
                                               masks, nomasks, judge_mask, judge_glove,
                                               glove_absence_counts=glove_absence)
                    raw_key = "+".join(raw_tags)
                    st = streak[int(tid)]
                    st["n"] = st["n"] + 1 if st["tags"] == raw_key else 1
                    st["tags"] = raw_key
                    confirmed = raw_tags if st["n"] >= args.smooth else []
                    far_note = "FAR" if (HAS_MASK and ph < args.min_face_h) else ""
                    x1, y1, x2, y2 = map(int, pbox)
                    if confirmed:
                        violations += 1
                        color = (0, 0, 255)
                        label = "WARNING " + "+".join(confirmed)
                    elif raw_tags:
                        color = (0, 255, 255)  # amber = verifying, not yet counted
                        label = f"VERIFYING {st['n']}/{args.smooth} " + "+".join(raw_tags)
                    else:
                        color = (0, 255, 0)
                        label = "OK" + (f" {far_note}" if far_note else "")
                    items.append((x1, y1, x2, y2, color, label, 2))
                    if int(tid) >= 0:
                        visitors.add(int(tid))
                    if confirmed:
                        vtype = "+".join(confirmed)
                        if warn_state.get(int(tid)) != vtype and int(tid) >= 0:  # once per episode
                            warn_state[int(tid)] = vtype
                            if "NO HELMET" in confirmed:
                                tally["no_helmet"] += 1
                            if "NO VEST" in confirmed:
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

                now = time.time()
                itn += 1
                if now - it0 >= 1.0:
                    infer_fps = itn / (now - it0)
                    itn, it0 = 0, now

                if violations > 0:
                    if winsound is not None and ALARM_WAV.exists():
                        if (not alarm_on) or (now - last_siren > 4.0):
                            winsound.PlaySound(str(ALARM_WAV),
                                               winsound.SND_FILENAME | winsound.SND_ASYNC | winsound.SND_LOOP)
                            if not alarm_on:
                                print("[ALARM] siren ON", flush=True)
                            last_siren, alarm_on = now, True
                else:
                    if alarm_on and winsound is not None:
                        winsound.PlaySound(None, winsound.SND_PURGE)
                        print("[ALARM] siren OFF", flush=True)
                        alarm_on = False

                with shared["lock"]:
                    cam = shared["hud"]["cam_fps"]
                    shared["items"] = items
                    shared["hud"] = {"violations": violations, "persons": len(persons),
                                     "visitors": len(visitors), "ok": len(ok_visitors),
                                     "tally": dict(tally), "infer_fps": infer_fps, "cam_fps": cam}
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
    print(f"[OK] PPE_Check running on {args.source} | video=full camera rate, AI overlay ~10 Hz. "
          f"Press Q to quit.", flush=True)

    ct0, ctn, cam_fps = time.time(), 0, 0.0
    while True:
        ok, frame = cap.read()
        if not ok:
            time.sleep(0.02)
            continue
        with shared["lock"]:
            shared["raw"] = frame
            items = list(shared["items"])
            hud = shared["hud"]
            hud_tally = dict(hud["tally"])
        for (x1, y1, x2, y2, color, label, thick) in items:
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, thick)
            cv2.putText(frame, label, (x1, max(0, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5 if thick == 1 else 0.6,
                        color, 1 if thick == 1 else 2)
        v = hud["violations"]
        pill_color = (0, 0, 210) if v > 0 else (0, 170, 0)
        cv2.rectangle(frame, (8, 8), (248, 38), pill_color, -1)
        cv2.circle(frame, (26, 23), 6, (255, 255, 255), -1)
        cv2.putText(frame, "WARNING" if v > 0 else "ALL COMPLIANT", (40, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, f"{hud['persons']} in view | {hud['visitors']} visitors | "
                           f"AI {hud['infer_fps']:.0f} + CAM {cam_fps:.0f} FPS",
                    (10, 54), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (230, 230, 230), 1)
        extra = ""
        if "no_gloves" in hud_tally:
            extra += f" | No gloves: {hud_tally['no_gloves']}"
        if "no_mask" in hud_tally:
            extra += f" | No mask: {hud_tally['no_mask']}"
        cv2.putText(frame, f"No helmet: {hud_tally['no_helmet']} | No vest: {hud_tally['no_vest']}"
                           f"{extra} | OK: {hud['ok']}",
                    (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

        now2 = time.time()
        ctn += 1
        if now2 - ct0 >= 1.0:
            cam_fps = ctn / (now2 - ct0)
            ctn, ct0 = 0, now2
            with shared["lock"]:
                shared["hud"]["cam_fps"] = cam_fps

        cv2.imshow("PPE_Check - Q to quit", frame)
        if cv2.waitKey(1) & 0xFF in (ord("q"), ord("Q")):
            shared["stop"] = True
            break

    worker.join(timeout=8)
    cap.release()
    cv2.destroyAllWindows()
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
