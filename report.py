"""Generate daily Excel report from incidents CSV.
Run: python report.py --mode zone   (zone intrusions)
     python report.py --mode ppe    (PPE violations)
     python report.py --mode zone --date 2026-09-12
"""
import argparse
from datetime import datetime
from pathlib import Path
import pandas as pd

BASE = Path(__file__).parent
PREFIX = {"zone": "incidents", "ppe": "ppe_incidents"}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="zone", choices=["zone", "ppe"])
    ap.add_argument("--date", default=datetime.now().strftime("%Y-%m-%d"))
    args = ap.parse_args()
    out = BASE / "reports" / f"{args.mode}_report_{args.date}.xlsx"
    out.parent.mkdir(parents=True, exist_ok=True)
    if args.mode == "ppe":  # counts-only tallies: merge stations (+ legacy files)
        frames = []
        for tag in ("ppe", "ppe_hv", "ppe_mg", "ppe_mh", "ppe_vg"):
            csv_path = BASE / "logs" / f"{tag}_stats_{args.date}.csv"
            if csv_path.exists():
                df = pd.read_csv(csv_path)
                df["station"] = tag
                frames.append(df)
        if not frames:
            print(f"[!] No log yet: logs/ppe_*_stats_{args.date}.csv")
            return
        tallies = pd.concat(frames, ignore_index=True)
        summary = tallies.groupby("metric", as_index=False)["count"].sum()
        with pd.ExcelWriter(out) as w:
            tallies.to_excel(w, sheet_name="Daily Stats", index=False)
            summary.to_excel(w, sheet_name="Combined", index=False)
        print(f"[OK] {out} (PPE tally, {len(frames)} station file(s))")
        return
    csv_path = BASE / "logs" / f"{PREFIX[args.mode]}_{args.date}.csv"
    if not csv_path.exists():
        print(f"[!] No log yet: {csv_path}")
        return
    df = pd.read_csv(csv_path)
    summary = df.groupby("violation_type").size().reset_index(name="count")
    hourly = df.copy()
    hourly["hour"] = pd.to_datetime(hourly["time"]).dt.strftime("%H:00")
    hourly = hourly.groupby(["hour", "violation_type"]).size().reset_index(name="count")
    with pd.ExcelWriter(out) as w:
        df.to_excel(w, sheet_name="Incidents", index=False)
        summary.to_excel(w, sheet_name="Summary", index=False)
        hourly.to_excel(w, sheet_name="Hourly", index=False)
    print(f"[OK] {out} ({len(df)} incidents)")

if __name__ == "__main__":
    main()
