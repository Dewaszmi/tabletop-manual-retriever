import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

UPLOADS_DIR = Path(
    os.environ.get("UPLOADS_DIR", PROJECT_ROOT / "data" / "uploads")
).resolve()
