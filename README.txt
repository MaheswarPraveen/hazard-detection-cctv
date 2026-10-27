================================================================================
                    SAFETY CCTV - WORKPLACE MONITORING SUITE
                     Zero-Install Standalone Windows Edition
================================================================================

QUICK START (3 EASY STEPS):
--------------------------------------------------------------------------------
1. CONFIGURE CAMERA (IF NOT USING DEFAULT):
   - Open config/settings.json in Notepad.
   - Set "camera": "0" (laptop webcam) or "1" (USB external camera) or RTSP URL:
     {"camera": "1", "port": 8000}
   - Save and close.

2. DEFINE DANGER ZONES:
   - Double-click tools/DefineZones.exe (or tools/DefineZones.bat).
   - Click points around your restricted area to draw a polygon.
   - Press 'S' to save the zone, 'R' to reset points, or 'Q' to quit and save.
   - Coordinates are automatically saved to config/zones.json.

3. LAUNCH THE SYSTEM:
   - Double-click SafetyCCTV.exe.
   - The web dashboard opens automatically in your browser at:
     http://localhost:8000
   - Click "Start" on any station:
       * Station 1: Zone Alert (Intrusion alarm & snapshot recorder)
       * Station 2: PPE Mask + Helmet (Entrance compliance station)
       * Station 3: PPE Helmet + Vest + Gloves (Fine-tuned 3-point gate check)

--------------------------------------------------------------------------------
STATION OPERATION & CAMERA SHARING:
--------------------------------------------------------------------------------
- All stations share one camera feed.
- Starting any station automatically stops the active station safely.
- Real-time stats, violation counts, and snapshots update automatically.

--------------------------------------------------------------------------------
ALERTS CONFIGURATION (OPTIONAL):
--------------------------------------------------------------------------------
To enable automatic Email (SMTP) and WhatsApp notifications on violations:
1. Copy config/alerts.example.json -> config/alerts.json
2. Open config/alerts.json in Notepad and fill in your details:
   - Email: Host, Port, Username, and App Password (e.g. Gmail App Password).
   - WhatsApp: Free CallMeBot Phone and API key.
3. Save the file. Stations will automatically pick it up on their next start.

--------------------------------------------------------------------------------
REPORTS & EXPORTS:
--------------------------------------------------------------------------------
- Real-time stats are logged daily in logs/ as CSV files.
- Click "Zone Excel" or "PPE Excel" on the dashboard to generate formatted Excel
  audit spreadsheets directly from the web interface.
- Violation snapshots are preserved organized by date inside evidence/<YYYY-MM-DD>/.

--------------------------------------------------------------------------------
UTILITY TOOLS (INSIDE tools/ FOLDER):
--------------------------------------------------------------------------------
- DefineZones.exe   : Interactive polygon drawing for restricted areas.
- CalibrateGlove.exe: 10-second glove color calibrator for custom colored gloves.
- CollectData.exe   : Capture custom PPE photos for model fine-tuning.

--------------------------------------------------------------------------------
TROUBLESHOOTING:
--------------------------------------------------------------------------------
- Camera busy or failed to open:
  Make sure no other software (Zoom, Teams, Camera app) is using the camera.
  Check USB cable connection or camera index in config/settings.json.
- Port 8000 in use:
  Change "port": 8080 in config/settings.json, save, and restart SafetyCCTV.exe.
- Refer to the full SafetyCCTV_Client_Guide.pdf for complete documentation.
================================================================================
