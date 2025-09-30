# utils.py
import json
from pathlib import Path

UPLOAD_DIR = Path("uploads")

def load_latest_results() -> list | str:
    import glob
    json_files = sorted(glob.glob(str(UPLOAD_DIR / "*.json")), reverse=True)
    if not json_files:
        return "No JSON files found."
    latest_json_path = Path(json_files[0])
    with open(latest_json_path, "r") as f:
        return json.load(f)
