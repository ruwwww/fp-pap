"""
src/__init__.py
Time Series Project - Package Initializer
"""
from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent
DATA_DIR = ROOT_DIR / "data"
CONFIG_DIR = ROOT_DIR / "config"
RESULTS_DIR = ROOT_DIR / "results"
