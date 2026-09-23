"""Safety CCTV dashboard - stdlib only, no installs.
Run: python dashboard.py   -> open http://localhost:8000
Start/stop Zone + PPE modes, view live stats, download CSVs, browse evidence.
"""
import csv
import json
import subprocess
import sys
import threading
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse, parse_qs

BASE = Path(__file__).parent
PORT = 8000
procs = {}  # mode -> Popen

MODES = {
    "zone": {"file": "mode_zone.py", "name": "Zone Alert", "desc": "Restricted danger-zone intrusion alarm",
              "extra": []},
    "ppe_hv": {"file": "mode_ppe.py", "name": "PPE Helmet + Vest", "desc": "Hardhat & safety-vest station",
               "extra": ["--checks", "helmet,vest"]},
    "ppe_mg": {"file": "mode_ppe.py", "name": "PPE Mask + Gloves", "desc": "Face-mast & gloves station (2-3 m gate)",
               "extra": ["--checks", "mask,gloves", "--gloves-model", "ppe_v8m.pt", "--glove-every", "8"]},
}
CAM_MODES = ("zone", "ppe_hv", "ppe_mg")  # share one camera: only one runs at a time

PAGE = """<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Safety CCTV</title>
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:-apple-system,"Segoe UI",Roboto,Arial,sans-serif;background:#f4f6f8;color:#1c2530;max-width:1060px;margin:0 auto;padding:28px 18px 60px}
h1{font-size:22px;font-weight:700;margin-bottom:4px}
.sub{color:#6b7686;font-size:13px;margin-bottom:22px}
.grid{display:grid;grid-template-columns:1fr 1fr;gap:14px;margin-bottom:14px}
@media(max-width:700px){.grid{grid-template-columns:1fr}}
.card{background:#fff;border:1px solid #e3e8ee;border-radius:12px;padding:18px}
.card h2{font-size:16px;margin-bottom:4px}
.card p{font-size:13px;color:#6b7686;margin-bottom:12px}
.dot{display:inline-block;width:9px;height:9px;border-radius:50%;margin-right:6px;vertical-align:baseline}
.on{background:#16a34a}.off{background:#cbd5e1}
.status{font-size:13px;font-weight:600;margin-bottom:12px}
button{border:0;border-radius:8px;padding:9px 18px;font-size:14px;font-weight:600;cursor:pointer;margin-right:8px}
.start{background:#16a34a;color:#fff}.start:hover{background:#15803d}
.stop{background:#f1f5f9;color:#334155}.stop:hover{background:#e2e8f0}
button:disabled{opacity:.45;cursor:default}
.stats{display:grid;grid-template-columns:repeat(4,1fr);gap:14px;margin:14px 0}
@media(max-width:700px){.stats{grid-template-columns:1fr 1fr}}
.stat{background:#fff;border:1px solid #e3e8ee;border-radius:12px;padding:14px 16px}
.stat b{font-size:24px;display:block}.stat span{font-size:12px;color:#6b7686}
.panel{background:#fff;border:1px solid #e3e8ee;border-radius:12px;padding:18px;margin-top:14px}
.panel h2{font-size:15px;margin-bottom:10px}
a.btn{display:inline-block;background:#0f172a;color:#fff;text-decoration:none;border-radius:8px;padding:8px 16px;font-size:13px;font-weight:600;margin:0 8px 8px 0}
a.btn:hover{background:#1e293b}
.gal{display:grid;grid-template-columns:repeat(auto-fill,minmax(150px,1fr));gap:10px;margin-top:8px}
.gal a{display:block;border-radius:8px;overflow:hidden;border:1px solid #e3e8ee}
.gal img{width:100%;height:110px;object-fit:cover;display:block}
.gal span{display:block;font-size:11px;padding:5px 7px;color:#475569;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.empty{color:#94a3b8;font-size:13px}
</style></head><body>
<h1>Safety CCTV Console</h1>
<div class="sub">Workplace safety monitoring &mdash; zone intrusion + PPE compliance</div>
<div class="grid">
 <div class="card"><h2>Zone Alert</h2><p>Restricted danger-zone intrusion alarm</p>
  <div class="status" id="s-zone"><span class="dot off"></span>checking&hellip;</div>
  <button class="start" id="b-zone-start" onclick="ctl('zone','start')">Start</button><button class="stop" id="b-zone-stop" onclick="ctl('zone','stop')">Stop</button></div>
 <div class="card"><h2>PPE Check</h2><p>Helmet + vest live warnings</p>
  <div class="status" id="s-ppe"><span class="dot off"></span>checking&hellip;</div>
  <button class="start" id="b-ppe-start" onclick="ctl('ppe','start')">Start</button><button class="stop" id="b-ppe-stop" onclick="ctl('ppe','stop')">Stop</button></div>
</div>
<div class="stats">
 <div class="stat"><b id="st-entries">0</b><span>danger entries today</span></div>
 <div class="stat"><b id="st-visitors">0</b><span>visitors seen</span></div>
 <div class="stat"><b id="st-nohelmet">0</b><span>no-helmet events</span></div>
 <div class="stat"><b id="st-novest">0</b><span>no-vest events</span></div>
</div>
<div class="panel"><h2>Reports &amp; data</h2>
 <a class="btn" href="/files/logs/PLACEHOLDER_ZONE">Zone CSV</a>
 <a class="btn" href="/files/logs/PLACEHOLDER_PPE">PPE tally CSV</a>
 <a class="btn" href="/api/report?mode=zone">Zone Excel</a>
 <a class="btn" href="/api/report?mode=ppe">PPE Excel</a>
 <div class="sub" style="margin:6px 0 0">Excel is generated on demand from today's data.</div></div>
<div class="panel"><h2>Evidence &mdash; today</h2><div class="gal" id="gal"><span class="empty">loading&hellip;</span></div></div>
<script>
async function refresh(){
 const r = await fetch('/api/status'); const j = await r.json();
 for (const m of ['zone','ppe']){
  const on = j.modes[m];
  document.getElementById('s-'+m).innerHTML = '<span class="dot '+(on?'on':'off')+'"></span>'+(on?'Running':'Stopped');
  document.getElementById('b-'+m+'-start').disabled = on;
  document.getElementById('b-'+m+'-stop').disabled = !on;
 }
 document.getElementById('st-entries').textContent = j.stats.entries;
 document.getElementById('st-visitors').textContent = j.stats.visitors;
 document.getElementById('st-nohelmet').textContent = j.stats.no_helmet;
 document.getElementById('st-novest').textContent = j.stats.no_vest;
 const g = document.getElementById('gal');
 if (!j.evidence.length){ g.innerHTML = '<span class="empty">No evidence captured today.</span>'; }
 else { g.innerHTML = j.evidence.map(e=>'<a href="/files/evidence/'+e.day+'/'+e.name+'" target="_blank"><img src="/files/evidence/'+e.day+'/'+e.name+'" loading="lazy"><span>'+e.name+'</span></a>').join(''); }
}
async function ctl(m,a){ await fetch('/api/'+a+'?mode='+m,{method:'POST'}); setTimeout(refresh,800); }
refresh(); setInterval(refresh,3000);
</script></body></html>
"""
PAGE = (BASE / "console.html").read_text(encoding="utf-8")


def today_str():
    from datetime import datetime
    return datetime.now().strftime("%Y-%m-%d")


def read_stats():
    day = today_str()
    stats = {"entries": 0, "visitors": 0, "no_helmet": 0, "no_vest": 0, "no_mask": 0, "no_gloves": 0}
    zp = BASE / "logs" / f"incidents_{day}.csv"
    if zp.exists():
        with open(zp) as f:
            stats["entries"] = max(0, sum(1 for _ in f) - 1)
    for name in (f"ppe_stats_{day}.csv", f"ppe_hv_stats_{day}.csv", f"ppe_mg_stats_{day}.csv"):
        pp = BASE / "logs" / name
        if pp.exists():
            with open(pp) as f:
                for row in csv.DictReader(f):
                    if row["metric"] in stats and row["metric"] != "entries":
                        stats[row["metric"]] += int(row["count"])
    return stats


def evidence_list(limit=24):
    day = today_str()
    d = BASE / "evidence" / day
    if not d.exists():
        return []
    files = sorted(d.glob("*.jpg"), key=lambda p: p.stat().st_mtime, reverse=True)[:limit]
    out = []
    for f in files:
        kind = "ppe" if "_NO_HELMET" in f.name.upper() or "_NO_VEST" in f.name.upper() else "zone"
        out.append({"name": f.name, "day": day, "kind": kind})
    return out


class Handler(SimpleHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _json(self, obj):
        body = json.dumps(obj).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        u = urlparse(self.path)
        if u.path == "/":
            day = today_str()
            page = PAGE.replace("PLACEHOLDER_ZONE", f"incidents_{day}.csv") \
                       .replace("PLACEHOLDER_PPE_HV", f"ppe_hv_stats_{day}.csv") \
                       .replace("PLACEHOLDER_PPE_MG", f"ppe_mg_stats_{day}.csv")
            body = page.encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif u.path == "/api/status":
            alive = {m: (m in procs and procs[m].poll() is None) for m in MODES}
            self._json({"modes": alive, "stats": read_stats(), "evidence": evidence_list()})
        elif u.path == "/api/report":
            mode = parse_qs(u.query).get("mode", ["zone"])[0]
            if mode not in MODES:
                self.send_error(400);
                return
            subprocess.run([sys.executable, "report.py", "--mode", mode], cwd=BASE,
                           capture_output=True)
            day = today_str()
            xlsx = BASE / "reports" / f"{mode}_report_{day}.xlsx"
            if not xlsx.exists():
                self.send_error(404, "No data for report yet");
                return
            data = xlsx.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
            self.send_header("Content-Disposition", f"attachment; filename={xlsx.name}")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
        elif u.path.startswith("/files/"):
            self.path = u.path[len("/files"):] or "/"
            super().do_GET()
        else:
            self.send_error(404)

    def do_POST(self):
        u = urlparse(self.path)
        q = parse_qs(u.query)
        mode = q.get("mode", [""])[0]
        if u.path == "/api/start" and mode in MODES:
            old = procs.get(mode)
            if old is not None and old.poll() is None:
                self._json({"ok": True, "already": True});
                return
            for other in CAM_MODES:  # one camera: starting one station stops the others
                if other != mode:
                    p = procs.get(other)
                    if p is not None and p.poll() is None:
                        p.terminate()
            log = open(BASE / f"{mode}_dash.log", "ab")
            flags = 0
            if sys.platform == "win32":
                flags = 0x00000008 | 0x00000200  # DETACHED_PROCESS | NEW_PROCESS_GROUP
            procs[mode] = subprocess.Popen(
                [sys.executable, "-u", MODES[mode]["file"], "--source", "0", "--camera", "CAM01"]
                + MODES[mode].get("extra", []),
                cwd=BASE, stdout=log, stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL, close_fds=True,
                creationflags=flags)
            self._json({"ok": True, "pid": procs[mode].pid})
        elif u.path == "/api/stop" and mode in MODES:
            p = procs.get(mode)
            if p is not None and p.poll() is None:
                p.terminate()
            self._json({"ok": True})
        else:
            self.send_error(404)


if __name__ == "__main__":
    handler = partial(Handler, directory=str(BASE))
    srv = ThreadingHTTPServer(("127.0.0.1", PORT), handler)
    print(f"[OK] Dashboard at http://localhost:{PORT}")
    srv.serve_forever()
