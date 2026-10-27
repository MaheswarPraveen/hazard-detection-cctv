"""Safety CCTV unified launcher and command dispatcher.
- Double-click: starts dashboard server + opens browser at http://localhost:<port>
- Single-instance: if server already active on port, opens browser and exits
- Subcommand dispatcher (--run):
    --run ppe <args>            -> mode_ppe.main()
    --run zone <args>           -> mode_zone.main()
    --run report <args>         -> report.main()
    --run define_zones <args>   -> define_zones.main()
    --run calibrate_glove <args>-> calibrate_glove.main()
    --run collect_data <args>   -> collect_data.main()
"""
import os
import socket
import sys
import threading
import time
import urllib.request
import webbrowser

os.environ["PYTHONUNBUFFERED"] = "1"

from paths import get_settings, ensure_dirs


def is_server_alive(port: int) -> bool:
    """Check if the dashboard server is already responding on this port."""
    try:
        url = f"http://127.0.0.1:{port}/api/status"
        req = urllib.request.Request(url, headers={"User-Agent": "SafetyCCTV-Launcher"})
        with urllib.request.urlopen(req, timeout=1.0) as resp:
            return resp.status == 200
    except Exception:
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.settimeout(0.5)
                return s.connect_ex(("127.0.0.1", port)) == 0
        except Exception:
            return False


def wait_and_open_browser(port: int, max_wait: float = 12.0):
    """Wait until local HTTP server is responding, then launch default browser."""
    url = f"http://localhost:{port}"
    start = time.time()
    while time.time() - start < max_wait:
        if is_server_alive(port):
            webbrowser.open(url)
            return
        time.sleep(0.2)
    webbrowser.open(url)


def dispatch_subcommand():
    """Handle `--run <command>` invocations."""
    run_idx = -1
    for i, arg in enumerate(sys.argv):
        if arg == "--run":
            run_idx = i
            break
    if run_idx == -1 or run_idx + 1 >= len(sys.argv):
        return False

    sub = sys.argv[run_idx + 1].lower()
    # Strip launcher arguments so sub-module argparse gets only the remaining flags
    sys.argv = [sys.argv[0]] + sys.argv[run_idx + 2:]

    if sub in ("ppe", "ppe_mh", "ppe_vg"):
        import mode_ppe
        mode_ppe.main()
        sys.exit(0)
    elif sub == "zone":
        import mode_zone
        mode_zone.main()
        sys.exit(0)
    elif sub == "report":
        import report
        report.main()
        sys.exit(0)
    elif sub in ("define_zones", "definezones", "zones"):
        import define_zones
        define_zones.main()
        sys.exit(0)
    elif sub in ("calibrate_glove", "calibrateglove", "glove"):
        import calibrate_glove
        calibrate_glove.main()
        sys.exit(0)
    elif sub in ("collect_data", "collectdata", "data"):
        import collect_data
        collect_data.main()
        sys.exit(0)
    else:
        print(f"[ERROR] Unknown subcommand: {sub}")
        sys.exit(1)


def main():
    ensure_dirs()
    if dispatch_subcommand():
        return

    settings = get_settings()
    port = int(settings.get("port", 8000))

    # Single-instance check
    if is_server_alive(port):
        print(f"[INFO] Safety CCTV already running on port {port}. Opening browser...")
        webbrowser.open(f"http://localhost:{port}")
        sys.exit(0)

    # Launch browser opener in background thread
    threading.Thread(target=wait_and_open_browser, args=(port,), daemon=True).start()

    # Start dashboard server on main thread
    import dashboard
    dashboard.run_server(port)


if __name__ == "__main__":
    main()
