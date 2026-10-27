# -*- mode: python ; coding: utf-8 -*-
from pathlib import Path
from PyInstaller.utils.hooks import collect_all

block_cipher = None

# Collect all resources and binaries for heavy computer vision & AI dependencies
datas_u, binaries_u, hidden_u = collect_all('ultralytics')
datas_m, binaries_m, hidden_m = collect_all('mediapipe')
datas_c, binaries_c, hidden_c = collect_all('cv2')
datas_s, binaries_s, hidden_s = collect_all('supervision')
datas_l, binaries_l, hidden_l = collect_all('lap')

all_datas = datas_u + datas_m + datas_c + datas_s + datas_l
all_binaries = binaries_u + binaries_m + binaries_c + binaries_s + binaries_l

all_hidden = list(set(
    hidden_u + hidden_m + hidden_c + hidden_s + hidden_l + [
        'lap',
        'pandas',
        'openpyxl',
        'paths',
        'launcher',
        'dashboard',
        'mode_ppe',
        'mode_zone',
        'report',
        'define_zones',
        'calibrate_glove',
        'collect_data',
        'alerts',
        'winsound',
    ]
))

a = Analysis(
    ['launcher.py'],
    pathex=['.'],
    binaries=all_binaries,
    datas=all_datas,
    hiddenimports=all_hidden,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=['tkinter', 'matplotlib', 'IPython', 'pytest', 'tensorboard'],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='SafetyCCTV',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,  # Windowless launcher for clean user experience
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='SafetyCCTV',
)
