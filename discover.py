"""Camera discovery: UPnP/SSDP + ONNX... ONVIF WS-Discovery probes (stdlib only)."""
import socket
import struct
import time
import urllib.request
import xml.etree.ElementTree as ET

SSDP = ("239.255.255.250", 1900)
MSEARCH = ("\r\n".join([
    "M-SEARCH * HTTP/1.1", "HOST: 239.255.255.250:1900", 'MAN: "ns=01; ns=01;"',
    "MX: 2", "ST: upnp:rootdevice", "", ""])).encode()

ONVIF = ("239.255.255.250", 3702)
PROBE = ("""<?xml version="1.0" encoding="UTF-8"?>"""
         """<e:Envelope xmlns:e="http://www.w3.org/2003/05/soap-envelope" """
         """xmlns:w="http://schemas.xmlsoap.org/ws/2004/08/addressing" """
         """xmlns:d="http://schemas.xmlsoap.org/ws/2005/04/discovery">"""
         """<e:Header><w:MessageID>uuid:2377a169-eb5f-47b5-9ff1-5a4a09d0e4b1</w:MessageID>"""
         """<w:To>urn:schemas-xmlsoap-org:ws:2005:04:discovery</w:To>"""
         """<w:Action>http://schemas.xmlsoap.org/ws/2005/04/discovery/Probe</w:Action></e:Header>"""
         """<e:Body><d:Probe><d:Types>dn:NetworkVideoTransmitter</d:Types></d:Probe></e:Body></e:Envelope>""").encode()

seen = {}


def listen(sock, seconds, tag):
    sock.settimeout(0.5)
    end = time.time() + seconds
    while time.time() < end:
        try:
            data, addr = sock.recvfrom(65535)
        except socket.timeout:
            continue
        key = (tag, addr[0])
        if key in seen:
            continue
        seen[key] = True
        txt = data.decode("utf-8", "ignore")
        name = ""
        for line in txt.splitlines():
            if line.lower().startswith("server:") or line.lower().startswith("location:"):
                name += " | " + line.strip()[:110]
        print(f"[{tag}] {addr[0]}:{addr[1]}{name}", flush=True)


print("--- SSDP (UPnP) ---", flush=True)
s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
s.bind(("", 0))
s.sendto(MSEARCH, SSDP)
listen(s, 5, "SSDP")
s.close()

print("--- ONVIF WS-Discovery ---", flush=True)
o = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
o.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_TTL, 2)
o.bind(("", 0))
o.sendto(PROBE, ONVIF)
o.settimeout(0.5)
end = time.time() + 6
got = set()
while time.time() < end:
    try:
        data, addr = o.recvfrom(65535)
    except socket.timeout:
        continue
    if addr[0] in got:
        continue
    got.add(addr[0])
    txt = data.decode("utf-8", "ignore")
    xaddrs = ""
    if "XAddrs" in txt:
        i = txt.find("XAddrs")
        xaddrs = txt[i:i + 160].replace("\n", " ")
    print(f"[ONVIF] camera at {addr[0]} {xaddrs}", flush=True)
o.close()
if not got:
    print("[ONVIF] no cameras answered", flush=True)
print("DONE", flush=True)
