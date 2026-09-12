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
    if args.mode == "ppe":
        csv_path = BASE / "logs" / f"ppe_stats_{args.date}.csv"
    else:
        csv_path = BASE / "logs" / f"{PREFIX[args.mode]}_{args.date}.csv"
    out = BASE / "reports" / f"{args.mode}_report_{args.date}.xlsx"
    out.parent.mkdir(parents=True, exist_ok=True)
    if not csv_path.exists():
        print(f"[!] No log yet: {csv_path}")
        return
    df = pd.read_csv(csv_path)
    if args.mode == "ppe":  # counts-only tally, no timestamps
        with pd.ExcelWriter(out) as w:
            df.to_excel(w, sheet_name="Daily Stats", index=False)
        print(f"[OK] {out} (PPE tally)")
        return
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
