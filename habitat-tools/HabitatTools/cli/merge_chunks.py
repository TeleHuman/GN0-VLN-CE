#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from HabitatTools.metrics import write_ce_outputs


def parse_args():
    parser = argparse.ArgumentParser(description="Merge chunk outputs for Habitat CE eval")
    parser.add_argument("--task", default="vlnce", choices=["vlnce"])
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--chunks", required=True, type=int)
    parser.add_argument("--eval-split", default="val_unseen")
    return parser.parse_args()


def _load_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def main() -> int:
    args = parse_args()
    root = Path(args.output_root)
    merged_dir = root / "merged"
    merged_dir.mkdir(parents=True, exist_ok=True)

    merged_map: dict[tuple[str, int], dict] = {}
    for idx in range(args.chunks):
        chunk_dir = root / f"chunk_{idx:03d}"
        rows = _load_rows(chunk_dir / "progress.jsonl")
        for row in rows:
            key = (str(row.get("scene_id", "")), int(row.get("episode_id", 0)))
            merged_map[key] = row

    rows = list(merged_map.values())
    rows.sort(key=lambda x: (str(x.get("scene_id", "")), int(x.get("episode_id", 0))))

    progress_path = merged_dir / "progress.jsonl"
    with progress_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    output_log, output_json, summary = write_ce_outputs(
        merged_dir,
        rows,
        split_name=args.eval_split,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"saved={output_log}")
    print(f"saved={output_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
