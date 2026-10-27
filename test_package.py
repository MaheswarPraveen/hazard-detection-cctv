"""Automated verification test suite for built SafetyCCTV distribution."""
import subprocess
import time
import urllib.request
import urllib.error
import json
import shutil
import os
from pathlib import Path

dist_dir = Path(r"C:\Users\xczma\.gemini\antigravity\scratch\hazard-detection-cctv\dist\SafetyCCTV")
exe_path = dist_dir / "SafetyCCTV.exe"

print("--- TEST 1: Launch SafetyCCTV.exe and verify port 8000 ---")
proc = subprocess.Popen([str(exe_path)], cwd=dist_dir)
time.sleep(4)

try:
    req = urllib.request.Request("http://127.0.0.1:8000/api/status")
    with urllib.request.urlopen(req, timeout=5) as r:
        assert r.status == 200
        data = json.loads(r.read())
        print("API Status response:", data)
        print("[PASS] Dashboard is running and /api/status responded 200")

    print("\n--- TEST 2: Single-instance behavior ---")
    res = subprocess.run([str(exe_path)], cwd=dist_dir, timeout=10)
    assert res.returncode == 0
    print("[PASS] Single instance check: second launch exited cleanly with code 0")

    print("\n--- TEST 3: Security check on /files/ ---")
    for forbidden in ["config/settings.json", "alerts.json", "../config/settings.json"]:
        try:
            url = f"http://127.0.0.1:8000/files/{forbidden}"
            urllib.request.urlopen(url, timeout=3)
            raise AssertionError(f"Should have failed with 403: {forbidden}")
        except urllib.error.HTTPError as e:
            assert e.code == 403, f"Expected 403 but got {e.code}"
            print(f"[PASS] /files/{forbidden} blocked with HTTP 403 Forbidden")

    print("\n--- TEST 4: Generate Excel reports ---")
    today = time.strftime("%Y-%m-%d")
    logs_dir = dist_dir / "logs"
    (logs_dir / f"incidents_{today}.csv").write_text(
        "time,camera,violation_type,track_id,in_frame_count,photo_path\n12:00:00,CAM01,ZONE_INTRUSION,1,10,\n",
        encoding="utf-8"
    )
    (logs_dir / f"ppe_mh_stats_{today}.csv").write_text(
        "metric,count\nvisitors,5\nno_helmet,2\nno_mask,1\n",
        encoding="utf-8"
    )

    req_z = urllib.request.Request("http://127.0.0.1:8000/api/report?mode=zone")
    with urllib.request.urlopen(req_z, timeout=15) as r:
        assert r.status == 200
        content = r.read()
        assert len(content) > 1000
        print(f"[PASS] Zone Excel report generated ({len(content)} bytes)")

    req_p = urllib.request.Request("http://127.0.0.1:8000/api/report?mode=ppe")
    with urllib.request.urlopen(req_p, timeout=15) as r:
        assert r.status == 200
        content = r.read()
        assert len(content) > 1000
        print(f"[PASS] PPE Excel report generated ({len(content)} bytes)")

    print("\n--- TEST 5: Camera sharing / Mutual station exclusion ---")
    # Start zone mode
    req_start_zone = urllib.request.Request("http://127.0.0.1:8000/api/start?mode=zone", method="POST")
    with urllib.request.urlopen(req_start_zone, timeout=5) as r:
        res = json.loads(r.read())
        print("Start zone response:", res)
        assert res.get("ok") is True

    time.sleep(2)
    with urllib.request.urlopen("http://127.0.0.1:8000/api/status", timeout=5) as r:
        status_data = json.loads(r.read())
        print("Status while zone active:", status_data["modes"])
        assert status_data["modes"]["zone"] is True

    # Start ppe_mh -> must stop zone
    req_start_ppe = urllib.request.Request("http://127.0.0.1:8000/api/start?mode=ppe_mh", method="POST")
    with urllib.request.urlopen(req_start_ppe, timeout=5) as r:
        res = json.loads(r.read())
        print("Start ppe_mh response:", res)
        assert res.get("ok") is True

    time.sleep(2)
    with urllib.request.urlopen("http://127.0.0.1:8000/api/status", timeout=5) as r:
        status_data = json.loads(r.read())
        print("Status while ppe_mh active:", status_data["modes"])
        assert status_data["modes"]["zone"] is False
        assert status_data["modes"]["ppe_mh"] is True
        print("[PASS] Starting PPE cleanly stopped Zone Alert!")

    # Stop ppe_mh
    urllib.request.urlopen(urllib.request.Request("http://127.0.0.1:8000/api/stop?mode=ppe_mh", method="POST"))

finally:
    try:
        proc.terminate()
        proc.wait(timeout=5)
    except Exception:
        subprocess.run(["taskkill", "/F", "/IM", "SafetyCCTV.exe"], capture_output=True)

print("\n--- TEST 6: Test execution from a directory path with SPACES ---")
spaces_dir = Path(r"C:\Users\xczma\.gemini\antigravity\scratch\hazard-detection-cctv\dist\Safety CCTV Test Spaces")
if spaces_dir.exists():
    shutil.rmtree(spaces_dir, ignore_errors=True)

# Copy minimal layout to spaces directory (avoid copying heavy _internal just for CLI check)
spaces_dir.mkdir(parents=True, exist_ok=True)
for item in ["SafetyCCTV.exe", "models", "config", "assets", "tools", "logs", "reports", "evidence", "dataset"]:
    src = dist_dir / item
    dst = spaces_dir / item
    if src.is_dir():
        shutil.copytree(src, dst, dirs_exist_ok=True)
    elif src.is_file():
        shutil.copy2(src, dst)

# Create directory junction for _internal so copy is instantaneous
subprocess.run(["cmd", "/c", "mklink", "/J", str(spaces_dir / "_internal"), str(dist_dir / "_internal")], capture_output=True)

spaces_exe = spaces_dir / "SafetyCCTV.exe"
res = subprocess.run([str(spaces_exe), "--run", "report", "--help"], capture_output=True, text=True, timeout=15)
assert res.returncode == 0
print("[PASS] Runs flawlessly from folder path with spaces!")

# Clean up spaces test folder
subprocess.run(["cmd", "/c", "rmdir", str(spaces_dir / "_internal")], capture_output=True)
shutil.rmtree(spaces_dir, ignore_errors=True)

print("\n=======================================================")
print("  ALL VERIFICATION TESTS PASSED SUCCESSFULLY! (6/6)")
print("=======================================================")
