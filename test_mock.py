import sys
from app import extract_features
try:
    print(extract_features("heloo"))
    print("Success")
except Exception as e:
    print(f"Error: {e}")
