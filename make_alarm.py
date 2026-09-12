"""Generate alarm.wav - two-tone industrial klaxon (nee-naw), stdlib only. Run once."""
import math
import struct
import wave
from pathlib import Path

OUT = Path(__file__).parent / "alarm.wav"
SR = 22050
frames = []
for _ in range(3):  # 3 nee-naw cycles = ~2.1s seamless loop
    for freq, dur in ((880, 0.35), (660, 0.35)):
        n = int(SR * dur)
        for i in range(n):
            t = i / SR
            env = min(1.0, i / (SR * 0.01), (n - i) / (SR * 0.01))  # click-free edges
            v = math.sin(2 * math.pi * freq * t) + 0.4 * math.sin(4 * math.pi * freq * t)
            frames.append(struct.pack("<h", int(11000 * env * v)))

with wave.open(str(OUT), "wb") as w:
    w.setnchannels(1)
    w.setsampwidth(2)
    w.setframerate(SR)
    w.writeframes(b"".join(frames))
print(f"[OK] wrote {OUT} (two-tone klaxon)")
