from pathlib import Path

# Project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"

RAW_DATA_DIR = DATA_DIR / "raw"

PROCESSED_DATA_DIR = DATA_DIR / "processed"

FEATURES_DATA_DIR = DATA_DIR / "features"

REPORTS_DIR = PROJECT_ROOT / "reports"

MODELS_DIR = PROJECT_ROOT / "models"

CONFIG_DIR = PROJECT_ROOT / "config"

NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"