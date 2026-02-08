from __future__ import annotations
import json
from pathlib import Path
from datetime import datetime

def safe_mkdir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p

def write_json(path: str | Path, obj) -> None:
    Path(path).write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")

def now_iso() -> str:
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"
