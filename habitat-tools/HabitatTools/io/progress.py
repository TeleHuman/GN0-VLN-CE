from __future__ import annotations

import json
from pathlib import Path


def load_done_episodes(progress_path: Path) -> tuple[set[tuple[str, int]], list[dict]]:
    done: set[tuple[str, int]] = set()
    rows: list[dict] = []

    if not progress_path.exists():
        return done, rows

    with progress_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            rows.append(row)
            done.add((str(row["scene_id"]), int(row["episode_id"])))

    return done, rows


def append_progress_row(progress_path: Path, row: dict) -> None:
    progress_path.parent.mkdir(parents=True, exist_ok=True)
    with progress_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
