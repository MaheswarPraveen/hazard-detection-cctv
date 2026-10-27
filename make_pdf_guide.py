"""Generate SafetyCCTV_Client_Guide.pdf using ReportLab.
Produces a clean, highly professional, publication-quality PDF manual.
"""
import sys
from pathlib import Path
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, KeepTogether, HRFlowable
)
from reportlab.pdfgen import canvas

class NumberedCanvas(canvas.Canvas):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_decorations(num_pages)
            super().showPage()
        super().save()

    def draw_page_decorations(self, page_count):
        self.saveState()
        self.setFont("Helvetica", 9)
        self.setFillColor(colors.HexColor("#64748b"))
        
        # Header (Pages > 1)
        if self._pageNumber > 1:
            self.drawString(54, letter[1] - 36, "Safety CCTV - Client Deployment & User Manual")
            self.drawRightString(letter[0] - 54, letter[1] - 36, "Confidential / Operational Guide")
            self.setStrokeColor(colors.HexColor("#e2e8f0"))
            self.setLineWidth(0.75)
            self.line(54, letter[1] - 42, letter[0] - 54, letter[1] - 42)
        
        # Footer
        text = f"Page {self._pageNumber} of {page_count}"
        self.drawRightString(letter[0] - 54, 34, text)
        self.drawString(54, 34, "Safety CCTV Platform | Zero-Install Standalone Edition")
        self.setStrokeColor(colors.HexColor("#e2e8f0"))
        self.setLineWidth(0.75)
        self.line(54, 46, letter[0] - 54, 46)
        self.restoreState()


def build_pdf(filename="SafetyCCTV_Client_Guide.pdf"):
    doc = SimpleDocTemplate(
        str(filename),
        pagesize=letter,
        leftMargin=54,
        rightMargin=54,
        topMargin=54,
        bottomMargin=54
    )

    styles = getSampleStyleSheet()
    
    # Custom styles
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=24,
        leading=28,
        textColor=colors.HexColor('#0f172a'),
        spaceAfter=6
    )
    subtitle_style = ParagraphStyle(
        'DocSubtitle',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=12,
        leading=16,
        textColor=colors.HexColor('#475569'),
        spaceAfter=15
    )
    h1_style = ParagraphStyle(
        'Header1',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=15,
        leading=19,
        textColor=colors.HexColor('#0f172a'),
        spaceBefore=14,
        spaceAfter=8,
        keepWithNext=True
    )
    h2_style = ParagraphStyle(
        'Header2',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=12,
        leading=16,
        textColor=colors.HexColor('#1e293b'),
        spaceBefore=10,
        spaceAfter=5,
        keepWithNext=True
    )
    body_style = ParagraphStyle(
        'Body',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=10,
        leading=14.5,
        textColor=colors.HexColor('#334155'),
        spaceAfter=8
    )
    body_bold = ParagraphStyle(
        'BodyBold',
        parent=body_style,
        fontName='Helvetica-Bold'
    )
    callout_style = ParagraphStyle(
        'CalloutText',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9.5,
        leading=14,
        textColor=colors.HexColor('#1e293b')
    )
    code_inline = ParagraphStyle(
        'CodeInline',
        parent=styles['Normal'],
        fontName='Courier',
        fontSize=9,
        leading=12,
        textColor=colors.HexColor('#0f172a')
    )

    story = []

    # Title Banner
    story.append(Paragraph("SAFETY CCTV CLIENT MANUAL", title_style))
    story.append(Paragraph("AI-Powered Workplace Safety, Danger Zone Intrusion & PPE Compliance Suite", subtitle_style))
    
    meta_data = [
        [Paragraph("<b>Target Environment:</b> Windows 10 / 11 (64-bit)", body_style),
         Paragraph("<b>Runtime:</b> Standalone CPU-Optimized (No Python needed)", body_style)],
        [Paragraph("<b>Interface:</b> Local Web Dashboard (Port 8000)", body_style),
         Paragraph("<b>Deployment Status:</b> Production Standalone Release", body_style)]
    ]
    t_meta = Table(meta_data, colWidths=[240, 264])
    t_meta.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor('#f1f5f9')),
        ('PADDING', (0,0), (-1,-1), 8),
        ('BOX', (0,0), (-1,-1), 1, colors.HexColor('#cbd5e1')),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
    ]))
    story.append(t_meta)
    story.append(Spacer(1, 14))

    # Section 1: Overview
    story.append(Paragraph("1. System Overview & Core Capabilities", h1_style))
    story.append(Paragraph(
        "<b>Safety CCTV</b> is an offline computer vision monitoring system engineered to run on standard client "
        "laptops and workstations. Using optimized neural network models running directly on the host CPU, "
        "the application continuously monitors video streams to prevent workplace accidents and ensure regulatory compliance.",
        body_style
    ))
    story.append(Paragraph("The software provides three distinct operational stations:", body_style))
    
    station_points = [
        [Paragraph("<b>1. Zone Alert</b>", body_style),
         Paragraph("Monitors designated restricted danger zones. Fires an audio siren upon intrusion, tags the intruder ID, and preserves timestamped evidence snapshots in <code>evidence/</code>.", body_style)],
        [Paragraph("<b>2. PPE Mask + Helmet</b>", body_style),
         Paragraph("Entry-gate safety checkpoint checking for mandatory hardhats and facial masks in real time.", body_style)],
        [Paragraph("<b>3. PPE Helmet + Vest + Gloves</b>", body_style),
         Paragraph("Full 3-point compliance checkpoint combining body tracking, fine-tuned glove detection, and MediaPipe landmark heuristics for active work areas.", body_style)]
    ]
    t_stations = Table(station_points, colWidths=[150, 354])
    t_stations.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (0,-1), colors.HexColor('#f8fafc')),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#e2e8f0')),
        ('PADDING', (0,0), (-1,-1), 6),
        ('VALIGN', (0,0), (-1,-1), 'TOP'),
    ]))
    story.append(t_stations)
    story.append(Spacer(1, 14))

    # Section 2: Quick Start
    story.append(Paragraph("2. Quick-Start Guide (Ready in 3 Minutes)", h1_style))
    story.append(Paragraph(
        "This software requires <b>zero software installations</b> on the target laptop. "
        "Follow these three simple steps to start monitoring:",
        body_style
    ))

    steps_data = [
        [Paragraph("<b>Step 1</b>", body_bold),
         Paragraph("<b>Extract & Place Application:</b><br/>Unzip <code>SafetyCCTV_win64.zip</code> into any folder on your laptop (e.g. <code>C:\\SafetyCCTV</code> or Desktop). All dependencies and AI models are contained inside.", body_style)],
        [Paragraph("<b>Step 2</b>", body_bold),
         Paragraph("<b>Define Danger Zones:</b><br/>Run <code>tools/DefineZones.exe</code>. Your camera view will open. Click points to outline your danger zone polygon. Press <b>'S'</b> to save the zone, then <b>'Q'</b> to exit.", body_style)],
        [Paragraph("<b>Step 3</b>", body_bold),
         Paragraph("<b>Double-Click SafetyCCTV.exe:</b><br/>The web console automatically launches at <code>http://localhost:8000</code> in your default web browser. Click <b>'Start'</b> on any station.", body_style)]
    ]
    t_steps = Table(steps_data, colWidths=[55, 449])
    t_steps.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (0,-1), colors.HexColor('#e0f2fe')),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#cbd5e1')),
        ('PADDING', (0,0), (-1,-1), 8),
        ('VALIGN', (0,0), (-1,-1), 'TOP'),
    ]))
    story.append(t_steps)
    story.append(Spacer(1, 14))

    # Section 3: Package Layout
    story.append(Paragraph("3. Directory Layout & Components", h1_style))
    story.append(Paragraph("The software package is organized into clearly defined directories:", body_style))

    files_data = [
        [Paragraph("<b>Directory / File</b>", body_bold), Paragraph("<b>Function & Description</b>", body_bold)],
        [Paragraph("<code>SafetyCCTV.exe</code>", code_inline), Paragraph("Main double-click application launcher. Boots backend & opens browser.", body_style)],
        [Paragraph("<code>_internal/</code>", code_inline), Paragraph("Shared AI runtime and libraries (PyTorch CPU, OpenCV, Ultralytics). Do not edit.", body_style)],
        [Paragraph("<code>models/</code>", code_inline), Paragraph("Deep learning model weights (<code>ppe_v8n.pt</code>, <code>ppe_ft.pt</code>, <code>yolo26n.onnx</code>).", body_style)],
        [Paragraph("<code>config/</code>", code_inline), Paragraph("System settings (<code>settings.json</code>), danger zones (<code>zones.json</code>), and alert settings.", body_style)],
        [Paragraph("<code>assets/</code>", code_inline), Paragraph("Dashboard UI (<code>console.html</code>) and audio siren (<code>alarm.wav</code>).", body_style)],
        [Paragraph("<code>tools/</code>", code_inline), Paragraph("Standalone utilities: <code>DefineZones.exe</code>, <code>CalibrateGlove.exe</code>, <code>CollectData.exe</code>.", body_style)],
        [Paragraph("<code>logs/</code>", code_inline), Paragraph("Daily CSV compliance logs recording hourly statistics and incident timestamps.", body_style)],
        [Paragraph("<code>reports/</code>", code_inline), Paragraph("Formatted Microsoft Excel audit reports generated on-demand from the console.", body_style)],
        [Paragraph("<code>evidence/</code>", code_inline), Paragraph("Timestamped JPEG violation photos organized automatically by date (<code>YYYY-MM-DD</code>).", body_style)],
    ]
    t_files = Table(files_data, colWidths=[140, 364])
    t_files.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#0f172a')),
        ('TEXTCOLOR', (0,0), (-1,0), colors.white),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#cbd5e1')),
        ('PADDING', (0,0), (-1,-1), 5),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
    ]))
    story.append(t_files)
    
    story.append(PageBreak())

    # Section 4: Configuration Reference
    story.append(Paragraph("4. Hardware & Configuration Guide", h1_style))
    
    story.append(Paragraph("Configuring Camera Source (<code>config/settings.json</code>)", h2_style))
    story.append(Paragraph(
        "By default, the system uses external USB camera index <code>\"1\"</code>. If you are using the laptop's built-in "
        "webcam or an RTSP network CCTV stream, open <code>config/settings.json</code> in Notepad and set:",
        body_style
    ))
    
    cfg_code = [
        [Paragraph("<b>Built-in Laptop Webcam:</b>", body_style), Paragraph('<code>{"camera": "0", "port": 8000}</code>', code_inline)],
        [Paragraph("<b>External USB Camera:</b>", body_style), Paragraph('<code>{"camera": "1", "port": 8000}</code>', code_inline)],
        [Paragraph("<b>IP CCTV Network Stream:</b>", body_style), Paragraph('<code>{"camera": "rtsp://admin:pass@192.168.1.100:554/ch1", "port": 8000}</code>', code_inline)],
    ]
    t_cfg = Table(cfg_code, colWidths=[170, 334])
    t_cfg.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (0,-1), colors.HexColor('#f8fafc')),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#cbd5e1')),
        ('PADDING', (0,0), (-1,-1), 6),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
    ]))
    story.append(t_cfg)
    story.append(Spacer(1, 8))
    story.append(Paragraph("<i>Note: Any changes saved to <code>settings.json</code> take effect immediately on the next station start without restarting the launcher.</i>", callout_style))
    story.append(Spacer(1, 12))

    story.append(Paragraph("Automated Email & WhatsApp Alerts (Optional)", h2_style))
    story.append(Paragraph(
        "To receive instant alerts when danger intrusions or safety non-compliance occurs:<br/>"
        "1. Copy <code>config/alerts.example.json</code> and rename it to <code>config/alerts.json</code>.<br/>"
        "2. Open <code>config/alerts.json</code> in Notepad and enter your credentials:<br/>"
        "&nbsp;&nbsp;&nbsp;&bull; <b>Email:</b> Enter your SMTP host, port, username, and password (for Gmail, generate a 16-character App Password).<br/>"
        "&nbsp;&nbsp;&nbsp;&bull; <b>WhatsApp:</b> Enter your phone number and free CallMeBot API key.<br/>"
        "3. Set <code>\"enabled\": true</code>. Alerts will automatically trigger with cool-downs to prevent inbox spam.",
        body_style
    ))
    story.append(Spacer(1, 10))

    # Section 5: Web Console & Reports
    story.append(Paragraph("5. Operating the Web Dashboard & Generating Reports", h1_style))
    story.append(Paragraph(
        "When <code>SafetyCCTV.exe</code> runs, it serves a real-time reactive console at <code>http://localhost:8000</code>. "
        "The interface is optimized for both desktop and wall-mounted monitoring screens, featuring Light/Dark modes, "
        "live status indicators, hourly metrics, and an evidence photo wall.",
        body_style
    ))

    dash_features = [
        [Paragraph("<b>Feature</b>", body_bold), Paragraph("<b>Operational Details</b>", body_bold)],
        [Paragraph("<b>Station Start / Stop</b>", body_bold),
         Paragraph("Only one station uses the camera at any moment. Clicking 'Start' on Station B cleanly terminates Station A, releases the camera lock, and initializes the new neural network within seconds.", body_style)],
        [Paragraph("<b>Real-time Metrics</b>", body_bold),
         Paragraph("Six live counters tally daily statistics: Danger Entries, Total Visitors Seen, No-Helmet Incidents, No-Vest Incidents, No-Mask Incidents, and No-Gloves Incidents.", body_style)],
        [Paragraph("<b>Live Evidence Wall</b>", body_bold),
         Paragraph("Violation photos are captured automatically and displayed as visual thumbnails. Clicking any thumbnail opens the full-resolution high-definition captured frame.", body_style)],
        [Paragraph("<b>Excel Audit Reports</b>", body_bold),
         Paragraph("Click <b>'Zone Excel'</b> or <b>'PPE Excel'</b> to instantly download complete Microsoft Excel reports compiled from today's CSV logs, broken down by hour and incident type.", body_style)],
    ]
    t_dash = Table(dash_features, colWidths=[130, 374])
    t_dash.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#0f172a')),
        ('TEXTCOLOR', (0,0), (-1,0), colors.white),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#cbd5e1')),
        ('PADDING', (0,0), (-1,-1), 6),
        ('VALIGN', (0,0), (-1,-1), 'TOP'),
    ]))
    story.append(t_dash)
    story.append(Spacer(1, 14))

    # Section 6: Troubleshooting & FAQs
    story.append(Paragraph("6. Troubleshooting & Operational FAQs", h1_style))
    
    faq_data = [
        [Paragraph("<b>Issue Observed</b>", body_bold), Paragraph("<b>Resolution Steps</b>", body_bold)],
        [Paragraph("<b>Camera open failed / black screen</b>", body_bold),
         Paragraph("1. Check if another program (Zoom, Teams, Skype, Windows Camera) is holding the webcam.<br/>2. Verify whether your camera is index <code>\"0\"</code> or <code>\"1\"</code> in <code>config/settings.json</code>.<br/>3. Replug the USB cable and click 'Start' again.", body_style)],
        [Paragraph("<b>Web page says 'Unable to connect'</b>", body_bold),
         Paragraph("If port 8000 is occupied by another application, open <code>config/settings.json</code>, change <code>\"port\": 8000</code> to <code>\"port\": 8080</code>, and restart <code>SafetyCCTV.exe</code>.", body_style)],
        [Paragraph("<b>App opens browser but does not launch twice</b>", body_bold),
         Paragraph("This is expected. Safety CCTV has single-instance protection. Double-clicking the icon while already active simply brings up the existing dashboard without creating duplicate processes.", body_style)],
        [Paragraph("<b>Zone alarms fire unexpectedly</b>", body_bold),
         Paragraph("Run <code>tools/DefineZones.exe</code> to verify the polygon boundary. Ensure it only covers restricted floor regions and does not overlap safe pedestrian walkways.", body_style)],
        [Paragraph("<b>Custom glove colors not recognized</b>", body_bold),
         Paragraph("Run <code>tools/CalibrateGlove.exe</code> under standard facility lighting. Hold the glove in the green square and press 'S' to calibrate the color filter into <code>config/glove_hsv.json</code>.", body_style)],
    ]
    t_faq = Table(faq_data, colWidths=[150, 354])
    t_faq.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#0f172a')),
        ('TEXTCOLOR', (0,0), (-1,0), colors.white),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#cbd5e1')),
        ('PADDING', (0,0), (-1,-1), 6),
        ('VALIGN', (0,0), (-1,-1), 'TOP'),
    ]))
    story.append(t_faq)
    story.append(Spacer(1, 14))

    # Footer note
    story.append(Paragraph(
        "<b>Customer Support & Updates:</b> Safety CCTV is designed for self-contained, air-gapped operation. "
        "All data, models, and incident evidence remain strictly confidential on the local computer.",
        callout_style
    ))

    doc.build(story, canvasmaker=NumberedCanvas)
    print(f"[OK] Generated PDF: {filename}")

if __name__ == "__main__":
    out_path = Path(__file__).parent / "SafetyCCTV_Client_Guide.pdf"
    build_pdf(out_path)
