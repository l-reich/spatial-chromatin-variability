# src/config.py
import os
from pathlib import Path

# Get the directory where this file is located (src/)
SRC_DIR = Path(os.path.dirname(os.path.abspath(__file__)))

# Get project root directory (one level up from src/)
PROJECT_ROOT = SRC_DIR.parent

# Define paths relative to project root
DATA_DIR = PROJECT_ROOT / "data"
H5AD_DIR = PROJECT_ROOT / "h5ad_files"
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = PROJECT_ROOT / "results" / "figures"

# Helper function to get paths
def get_h5ad_path(filename):
    return H5AD_DIR / filename