"""Alert sender: email (SMTP, stdlib) + WhatsApp (CallMeBot, stdlib http).
Setup:
  1. copy alerts.example.json -> alerts.json, fill your values (see below)
  2. python alerts.py --test   (one test message per configured channel)
  3. stations pick it up automatically on next start (no code changes needed)

Email: any SMTP (Gmail needs an APP password, not your login password:
Settings -> Security -> 2-Step Verification -> App passwords).
WhatsApp (free, text only): message the CallMeBot bot from YOUR WhatsApp to
get an apikey - exact steps at https://www.callmebot.com/blog/free-api-whatsapp-messages/
"""
import json
import smtplib
import time
import urllib.parse
import urllib.request
from email.message import EmailMessage
from pathlib import Path

BASE = Path(__file__).parent
CFG = BASE / "alerts.json"


def load_cfg():
    if not CFG.exists():
        return {}
    try:
        return json.loads(CFG.read_text())
    except Exception as e:
        print(f"[ALERT] bad alerts.json: {e}", flush=True)
        return {}


class Throttle:
    """One message per key per cooldown window (violations repeat for minutes;
    your phone shouldn't)."""

    def __init__(self, seconds=300):
        self.seconds = seconds
        self.last = {}

    def ready(self, key):
        now = time.time()
        if now - self.last.get(key, 0) >= self.seconds:
            self.last[key] = now
            return True
        return False


def send_email(subject, body, attach=None):
    cfg = load_cfg().get("email", {})
    if not cfg.get("enabled"):
        return False
    try:
        msg = EmailMessage()
        msg["From"] = cfg["user"]
        msg["To"] = cfg["to"]
        msg["Subject"] = subject
        msg.set_content(body)
        if attach:
            p = Path(attach)
            if p.exists():
                msg.add_attachment(p.read_bytes(), maintype="image",
                                   subtype="jpeg", filename=p.name)
        ctx = smtplib.SMTP(cfg.get("host", "smtp.gmail.com"),
                           int(cfg.get("port", 587)), timeout=20)
        with ctx as s:
            s.starttls()
            s.login(cfg["user"], cfg["password"])
            s.send_message(msg)
        print(f"[ALERT] email sent: {subject}", flush=True)
        return True
    except Exception as e:
        print(f"[ALERT] email FAILED: {e}", flush=True)
        return False


def send_whatsapp(text):
    cfg = load_cfg().get("whatsapp", {})
    if not cfg.get("enabled"):
        return False
    try:
        url = ("https://api.callmebot.com/whatsapp.php?phone="
               + urllib.parse.quote(str(cfg["phone"]))
               + "&text=" + urllib.parse.quote(text)
               + "&apikey=" + urllib.parse.quote(str(cfg["apikey"])))
        with urllib.request.urlopen(url, timeout=20) as r:
            body = r.read().decode(errors="replace")
        ok = "queued" in body.lower() or "sent" in body.lower()
        print(f"[ALERT] whatsapp {'sent' if ok else 'FAILED: ' + body[:120]}", flush=True)
        return ok
    except Exception as e:
        print(f"[ALERT] whatsapp FAILED: {e}", flush=True)
        return False


if __name__ == "__main__":
    import sys
    if "--test" in sys.argv:
        cfg = load_cfg()
        if not cfg:
            print("[ALERT] no alerts.json - copy alerts.example.json first")
            raise SystemExit(1)
        send_email("[TEST] Safety CCTV alerts working",
                   "Test message from your Safety CCTV station.")
        send_whatsapp("TEST: Safety CCTV alerts working.")
