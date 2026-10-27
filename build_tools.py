import subprocess
from pathlib import Path

base = Path(r"C:\Users\xczma\.gemini\antigravity\scratch\hazard-detection-cctv")
tools_dir = base / "tools"
tools_dir.mkdir(parents=True, exist_ok=True)

csc = r"C:\Windows\Microsoft.NET\Framework64\v4.0.30319\csc.exe"

tools = [
    ("DefineZones", "define_zones"),
    ("CalibrateGlove", "calibrate_glove"),
    ("CollectData", "collect_data")
]

cs_template = """using System;
using System.Diagnostics;
using System.IO;

namespace SafetyCCTVTools {{
    class Program {{
        static int Main(string[] args) {{
            string baseDir = AppDomain.CurrentDomain.BaseDirectory;
            string parentDir = Directory.GetParent(baseDir.TrimEnd('\\\\', '/')).FullName;
            string targetExe = Path.Combine(parentDir, "SafetyCCTV.exe");
            if (!File.Exists(targetExe)) {{
                targetExe = Path.Combine(baseDir, "SafetyCCTV.exe");
            }}
            if (!File.Exists(targetExe)) {{
                Console.ForegroundColor = ConsoleColor.Red;
                Console.WriteLine("[ERROR] SafetyCCTV.exe not found at: " + targetExe);
                Console.ResetColor();
                Console.WriteLine("Ensure this tool is placed in the tools/ folder of Safety CCTV.");
                Console.WriteLine("Press any key to exit...");
                Console.ReadKey();
                return 1;
            }}

            string argString = "--run {sub}";
            if (args.Length > 0) {{
                argString += " " + string.Join(" ", args);
            }}

            ProcessStartInfo psi = new ProcessStartInfo(targetExe, argString);
            psi.UseShellExecute = false;
            psi.WorkingDirectory = parentDir;

            try {{
                using (Process p = Process.Start(psi)) {{
                    p.WaitForExit();
                    return p.ExitCode;
                }}
            }} catch (Exception ex) {{
                Console.WriteLine("[ERROR] Failed to launch: " + ex.Message);
                return 1;
            }}
        }}
    }}
}}
"""

for name, sub in tools:
    cs_code = cs_template.format(sub=sub)
    cs_file = tools_dir / f"{name}.cs"
    exe_file = tools_dir / f"{name}.exe"
    bat_file = tools_dir / f"{name}.bat"

    cs_file.write_text(cs_code, encoding="utf-8")
    res = subprocess.run([csc, "/nologo", f"/out:{exe_file}", str(cs_file)], capture_output=True, text=True)
    if res.returncode != 0:
        print(f"Failed {name}: {res.stderr}")
    else:
        print(f"Compiled {name}.exe successfully")
        try:
            cs_file.unlink()
        except OSError:
            pass

    bat_file.write_text(f'@echo off\r\ncd /d "%~dp0.."\r\nSafetyCCTV.exe --run {sub} %*\r\n', encoding="utf-8")
    print(f"Created {name}.bat")
