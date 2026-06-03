#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze aggregate CE metrics from current GN0-VLN-CE outputs."
    )
    parser.add_argument(
        "--path",
        required=True,
        help=(
            "Run directory, chunk directory, merged directory, progress.jsonl, or log "
            "directory. Auto mode understands eval runs and dagger ce_run outputs."
        ),
    )
    parser.add_argument(
        "--source",
        choices=["auto", "merged", "progress", "chunks", "logs"],
        default="auto",
        help="Input discovery mode. Default: auto.",
    )
    parser.add_argument(
        "--output-json",
        default="",
        help="Optional path to write the computed summary JSON.",
    )
    parser.add_argument(
        "--no-dedup",
        action="store_true",
        help="Do not deduplicate rows by scene_id/episode_id/id.",
    )
    parser.add_argument(
        "--top-scenes",
        type=int,
        default=10,
        help="Number of worst/best scene rows to print.",
    )
    return parser.parse_args()


def safe_float(value, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def metric(row: dict, *keys: str, default: float = 0.0) -> float:
    for key in keys:
        if row.get(key) is not None:
            return safe_float(row.get(key), default=default)
    return default


def row_key(row: dict) -> tuple:
    scene_id = row.get("scene_id")
    episode_id = row.get("episode_id")
    if scene_id is not None and episode_id is not None:
        return ("scene_episode", str(scene_id), str(episode_id))
    row_id = row.get("id")
    if row_id:
        return ("id", str(row_id))
    return ("object", id(row))


def sort_key(row: dict) -> tuple:
    return (
        str(row.get("scene_id", "")),
        safe_float(row.get("episode_id"), default=-1.0),
        str(row.get("id", "")),
    )


def load_progress_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise SystemExit(f"Invalid JSONL at {path}:{line_no}: {exc}") from exc
            if isinstance(row, dict):
                rows.append(row)
    return rows


def load_log_dir(path: Path) -> list[dict]:
    rows: list[dict] = []
    for item in sorted(path.glob("*.json")):
        try:
            row = json.loads(item.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise SystemExit(f"Invalid JSON at {item}: {exc}") from exc
        if isinstance(row, dict):
            rows.append(row)
    return rows


def discover_sources(path: Path, source: str) -> tuple[str, list[Path]]:
    if path.is_file():
        if path.name.endswith(".jsonl"):
            return "progress", [path]
        if path.suffix == ".json":
            return "json", [path]
        raise SystemExit(f"Unsupported input file: {path}")

    if not path.exists():
        raise SystemExit(f"Path not found: {path}")

    if source == "progress":
        progress = path / "progress.jsonl"
        if progress.exists():
            return "progress", [progress]
        raise SystemExit(f"No progress.jsonl found under {path}")

    if source == "merged":
        candidates = [
            path / "merged" / "progress.jsonl",
            path / "ce_run" / "merged" / "progress.jsonl",
            path / "progress.jsonl",
        ]
        hits = [p for p in candidates if p.exists()]
        if hits:
            return "merged", hits[:1]
        raise SystemExit(f"No merged progress.jsonl found under {path}")

    if source == "chunks":
        hits = sorted(path.glob("chunk_*/progress.jsonl"))
        hits.extend(sorted((path / "ce_run").glob("chunk_*/progress.jsonl")))
        if hits:
            return "chunks", hits
        raise SystemExit(f"No chunk progress.jsonl files found under {path}")

    if source == "logs":
        candidates = [
            path,
            path / "log",
            path / "merged" / "log",
            path / "ce_run" / "merged" / "log",
        ]
        hits = [p for p in candidates if p.is_dir() and list(p.glob("*.json"))]
        if hits:
            return "logs", hits[:1]
        raise SystemExit(f"No log/*.json files found under {path}")

    auto_candidates = [
        ("merged", path / "merged" / "progress.jsonl"),
        ("merged", path / "ce_run" / "merged" / "progress.jsonl"),
        ("progress", path / "progress.jsonl"),
    ]
    for mode, candidate in auto_candidates:
        if candidate.exists():
            return mode, [candidate]

    chunk_hits = sorted(path.glob("chunk_*/progress.jsonl"))
    chunk_hits.extend(sorted((path / "ce_run").glob("chunk_*/progress.jsonl")))
    if chunk_hits:
        return "chunks", chunk_hits

    log_candidates = [
        path / "log",
        path / "merged" / "log",
        path / "ce_run" / "merged" / "log",
    ]
    if path.name == "log":
        log_candidates.append(path)
    for candidate in log_candidates:
        if candidate.is_dir() and list(candidate.glob("*.json")):
            return "logs", [candidate]

    raise SystemExit(
        f"No CE rows found under {path}. Expected progress.jsonl, merged/progress.jsonl, "
        "ce_run/merged/progress.jsonl, chunk_*/progress.jsonl, or log/*.json."
    )


def load_json_file(path: Path) -> list[dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return [row for row in payload if isinstance(row, dict)]
    if isinstance(payload, dict):
        rows = payload.get("rows") or payload.get("episodes")
        if isinstance(rows, list):
            return [row for row in rows if isinstance(row, dict)]
        raise SystemExit(
            f"{path} is a summary JSON, not per-episode rows. Use a progress.jsonl or log directory."
        )
    raise SystemExit(f"Unsupported JSON payload in {path}")


def load_rows(paths: Iterable[Path], mode: str) -> list[dict]:
    rows: list[dict] = []
    for path in paths:
        if mode == "logs":
            rows.extend(load_log_dir(path))
        elif mode == "json":
            rows.extend(load_json_file(path))
        else:
            rows.extend(load_progress_jsonl(path))
    return rows


def dedup_rows(rows: list[dict]) -> list[dict]:
    merged: dict[tuple, dict] = {}
    for row in rows:
        merged[row_key(row)] = row
    return sorted(merged.values(), key=sort_key)


def mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def median(values: list[float]) -> float | None:
    return statistics.median(values) if values else None


def pct(value: float | None) -> str:
    return "-" if value is None else f"{value * 100:.2f}%"


def num(value: float | None, digits: int = 4) -> str:
    return "-" if value is None else f"{value:.{digits}f}"


def summarize_group(rows: list[dict]) -> dict:
    count = len(rows)
    success_count = sum(1 for row in rows if metric(row, "success") >= 0.5)
    return {
        "episodes": count,
        "success_count": success_count,
        "failure_count": count - success_count,
        "sr": (success_count / count) if count else None,
        "spl": mean([metric(row, "spl") for row in rows]),
        "os": mean([metric(row, "os", "oracle_success") for row in rows]),
        "ne": mean([metric(row, "ne", "distance_to_goal") for row in rows]),
        "ne_median": median([metric(row, "ne", "distance_to_goal") for row in rows]),
        "path_length": mean([metric(row, "path_length") for row in rows]),
        "steps": mean([metric(row, "steps") for row in rows]),
    }


def summarize(rows: list[dict], source_paths: list[Path], mode: str) -> dict:
    total = summarize_group(rows)
    split_counts = Counter(str(row.get("split", "unknown")) for row in rows)
    early_stop_counts = Counter(
        str(row.get("early_stop_reason"))
        for row in rows
        if row.get("early_stop_reason")
    )

    per_scene: list[dict] = []
    scene_rows: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        scene_rows[str(row.get("scene_id", "unknown"))].append(row)

    for scene_id, items in scene_rows.items():
        scene_summary = summarize_group(items)
        scene_summary["scene_id"] = scene_id
        per_scene.append(scene_summary)

    per_scene.sort(
        key=lambda item: (
            1.0 if item["sr"] is None else item["sr"],
            -(item["episodes"]),
            str(item["scene_id"]),
        )
    )

    success_rows = [row for row in rows if metric(row, "success") >= 0.5]
    fail_rows = [row for row in rows if metric(row, "success") < 0.5]

    return {
        "source_mode": mode,
        "source_paths": [str(p) for p in source_paths],
        "splits": dict(sorted(split_counts.items())),
        "total": total,
        "success_cases": summarize_group(success_rows),
        "failure_cases": summarize_group(fail_rows),
        "early_stop_reasons": dict(sorted(early_stop_counts.items())),
        "per_scene": per_scene,
    }


def print_summary(summary: dict, top_scenes: int) -> None:
    total = summary["total"]
    print(f"Source mode: {summary['source_mode']}")
    for path in summary["source_paths"]:
        print(f"Source: {path}")
    print(f"Splits: {summary['splits']}")
    print()
    print("Total")
    print(f"  Episodes: {total['episodes']}")
    print(f"  Success:  {total['success_count']} / {total['episodes']} = {pct(total['sr'])}")
    print(f"  SPL:      {pct(total['spl'])}")
    print(f"  OS:       {pct(total['os'])}")
    print(f"  NE:       {num(total['ne'])}  median={num(total['ne_median'])}")
    print(f"  TL:       {num(total['path_length'])}")
    print(f"  Steps:    {num(total['steps'], digits=2)}")

    if summary["early_stop_reasons"]:
        print()
        print("Early Stops")
        for reason, count in summary["early_stop_reasons"].items():
            print(f"  {reason}: {count}")

    scenes = summary["per_scene"]
    if scenes and top_scenes > 0:
        print()
        print(f"Worst Scenes ({min(top_scenes, len(scenes))})")
        for item in scenes[:top_scenes]:
            print(
                "  "
                f"{item['scene_id']} "
                f"ep={item['episodes']} sr={pct(item['sr'])} "
                f"spl={pct(item['spl'])} ne={num(item['ne'])}"
            )

        best = sorted(
            scenes,
            key=lambda item: (
                -1.0 if item["sr"] is None else -item["sr"],
                -item["episodes"],
                str(item["scene_id"]),
            ),
        )
        print()
        print(f"Best Scenes ({min(top_scenes, len(best))})")
        for item in best[:top_scenes]:
            print(
                "  "
                f"{item['scene_id']} "
                f"ep={item['episodes']} sr={pct(item['sr'])} "
                f"spl={pct(item['spl'])} ne={num(item['ne'])}"
            )


def main() -> int:
    args = parse_args()
    mode, source_paths = discover_sources(Path(args.path), args.source)
    rows = load_rows(source_paths, mode)
    if not args.no_dedup:
        rows = dedup_rows(rows)
    if not rows:
        raise SystemExit("No per-episode rows found.")

    summary = summarize(rows, source_paths, mode)
    print_summary(summary, top_scenes=max(args.top_scenes, 0))

    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(summary, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print()
        print(f"Saved: {output_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
