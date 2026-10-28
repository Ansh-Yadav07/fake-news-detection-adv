# Run from the repo root: python scripts/experiments/check_mock.py
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from app import extract_features
try:
    print(extract_features("heloo"))
    print("Success")
except Exception as e:
    print(f"Error: {e}")
