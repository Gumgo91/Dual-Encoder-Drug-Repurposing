"""Project paths. Set L1000_DATA_DIR, L1000_CACHE_DIR or L1000_RESULTS_DIR to use other locations."""

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = Path(os.environ.get("L1000_DATA_DIR", ROOT / "data"))
CACHE_DIR = Path(os.environ.get("L1000_CACHE_DIR", ROOT / "cache"))
RESULTS_DIR = Path(os.environ.get("L1000_RESULTS_DIR", ROOT / "results"))
MODELS_DIR = ROOT / "models"
FIGURES_DIR = ROOT / "figures"
