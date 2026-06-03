from __future__ import annotations

import json
from pathlib import Path


def _safe_float(value, default=0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def summarize_ce(rows: list[dict]) -> dict:
    if not rows:
        return {
            "success": 0.0,
            "spl": 0.0,
            "os": 0.0,
            "ne": 0.0,
            "path_length": 0.0,
            "length": 0,
        }

    count = len(rows)
    return {
        "success": sum(float(r.get("success", 0.0)) for r in rows) / count,
        "spl": sum(float(r.get("spl", 0.0)) for r in rows) / count,
        "os": sum(float(r.get("os", 0.0)) for r in rows) / count,
        "ne": sum(float(r.get("ne", 0.0)) for r in rows) / count,
        "path_length": sum(float(r.get("path_length", 0.0)) for r in rows) / count,
        "length": count,
    }


def _calc_lines(rows: list[dict], label: str) -> list[str]:
    lines = [f"### {label} ({len(rows)}) ###"]
    if len(rows) == 0:
        return lines

    total_steps = 0.0
    total_tl = 0.0
    total_ne = 0.0
    total_os = 0.0
    total_sr = 0.0
    total_spl = 0.0

    for row in rows:
        total_steps += _safe_float(row.get("steps", 0.0))
        total_tl += _safe_float(row.get("path_length", 0.0))
        total_ne += _safe_float(row.get("ne", row.get("distance_to_goal", 0.0)))
        total_os += _safe_float(row.get("os", row.get("oracle_success", 0.0)))
        total_sr += _safe_float(row.get("success", 0.0))
        total_spl += _safe_float(row.get("spl", 0.0))

    count = len(rows)
    lines.append(f"Steps = {int(total_steps)} / {count} = {round(total_steps / count, 2)}")
    lines.append(f"TL = {total_tl} / {count} = {round(total_tl / count, 4)}")
    lines.append(f"NE = {total_ne} / {count} = {round(total_ne / count, 4)}")
    lines.append(f"OS = {total_os} / {count} = {round((total_os / count), 4) * 100}%")
    lines.append(f"SR = {total_sr} / {count} = {round((total_sr / count), 4) * 100}%")
    lines.append(f"SPL = {total_spl} / {count} = {round((total_spl / count), 4) * 100}%")
    return lines


def summarize_ce_lines(rows: list[dict], split_name: str = "default") -> list[str]:
    lines: list[str] = []
    lines.append(f"[split:{split_name}] total {len(rows)} data")
    lines.extend(_calc_lines(rows, "Total"))
    lines.extend(_calc_lines([r for r in rows if _safe_float(r.get("success", 0.0)) > 0], "Success Case"))
    lines.extend(_calc_lines([r for r in rows if _safe_float(r.get("success", 0.0)) == 0], "Fail Case"))
    lines.append(f"############[{split_name}]#############")
    lines.append("##########################")
    return lines


def write_ce_outputs(output_dir: Path, rows: list[dict], split_name: str):
    output_log = output_dir / "eval_result.log"
    output_json = output_dir / "result.json"

    lines = summarize_ce_lines(rows, split_name=split_name)
    output_log.write_text("\n".join(lines), encoding="utf-8")

    summary = summarize_ce(rows)
    summary["split"] = split_name
    summary["length"] = len(rows)
    output_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return output_log, output_json, summary
