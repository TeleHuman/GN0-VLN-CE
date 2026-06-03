from __future__ import annotations

import json
from pathlib import Path


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def write_episode_log(log_dir: Path, episode_id: str, payload: dict) -> None:
    log_dir.mkdir(parents=True, exist_ok=True)
    out = log_dir / f"{episode_id}.json"
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
