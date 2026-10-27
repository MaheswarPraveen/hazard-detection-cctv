"""Central path and directory management for Safety CCTV.
Supports execution both from source tree and PyInstaller frozen runtime.
"""
import json
import sys
from pathlib import Path

if getattr(sys, "frozen", False):
    APP_DIR = Path(sys.executable).resolve().parent
else:
    APP_DIR = Path(__file__).resolve().parent

MODELS_DIR = APP_DIR / "models"
CONFIG_DIR = APP_DIR / "config"
ASSETS_DIR = APP_DIR / "assets"
LOGS_DIR = APP_DIR / "logs"
REPORTS_DIR = APP_DIR / "reports"
EVIDENCE_DIR = APP_DIR / "evidence"
DATASET_DIR = APP_DIR / "dataset"
TOOLS_DIR = APP_DIR / "tools"


def ensure_dirs():
    """Ensure all required runtime directories exist and are writable."""
    for d in (MODELS_DIR, CONFIG_DIR, ASSETS_DIR, LOGS_DIR, REPORTS_DIR, EVIDENCE_DIR, DATASET_DIR):
        d.mkdir(parents=True, exist_ok=True)


def get_model_path(name: str) -> Path:
    """Find a model file: check explicit path, then models/, then app root."""
    if not name:
        return Path("")
    p = Path(name)
    if p.is_file():
        return p.resolve()
    in_models = MODELS_DIR / name
    if in_models.is_file():
        return in_models
    in_app = APP_DIR / name
    if in_app.is_file():
        return in_app
    return in_models


def get_config_path(name: str) -> Path:
    """Find a configuration file: check config/, then app root."""
    in_cfg = CONFIG_DIR / name
    if in_cfg.is_file():
        return in_cfg
    in_app = APP_DIR / name
    if in_app.is_file():
        return in_app
    return in_cfg


def get_asset_path(name: str) -> Path:
    """Find an asset file: check assets/, then app root."""
    in_assets = ASSETS_DIR / name
    if in_assets.is_file():
        return in_assets
    in_app = APP_DIR / name
    if in_app.is_file():
        return in_app
    return in_assets


def get_settings() -> dict:
    """Load settings from config/settings.json, auto-creating with defaults if missing."""
    settings_file = get_config_path("settings.json")
    defaults = {"camera": "1", "port": 8000}
    if not settings_file.exists():
        CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        try:
            settings_file.write_text(json.dumps(defaults, indent=2), encoding="utf-8")
        except Exception:
            pass
        return defaults.copy()
    try:
        data = json.loads(settings_file.read_text(encoding="utf-8"))
        for k, v in defaults.items():
            data.setdefault(k, v)
        return data
    except Exception:
        return defaults.copy()
