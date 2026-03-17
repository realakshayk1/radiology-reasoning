"""
paths.py — Path constants and helpers for the RadReason project.
"""

import os
from pathlib import Path

def get_project_root() -> Path:
    """Returns the absolute path to the project root."""
    return Path(__file__).parent.parent.parent

# Common paths
ROOT = get_project_root()
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
DATA_SPLITS = ROOT / "data" / "splits"
ARTIFACTS = ROOT / "artifacts"
MODELS = ARTIFACTS / "models"
FIGURES = ARTIFACTS / "figures"
CONFIGS = ROOT / "configs"
