#!/usr/bin/env python3
import argparse
import json
import os
import shlex
import statistics
import subprocess
import sys
import time
from datetime import datetime


DEFAULT_PYTHON = os.environ.get("PYTHON_BIN", sys.executable or "python")


REMOTE_AGG_CODE = r"""
import json
import statistics
from pathlib import Path

run_dir = Path(RUN_DIR)
root = run_dir / "ce_run"
rows = []
per_scene = {}

for p in sorted(root.glob("chunk_*/progress.jsonl")):
    try:
        with p.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except Exception:
                    continue
                rows.append(row)
                scene = str(row.get("scene_id", "unknown"))
                per_scene.setdefault(scene, []).append(row)
    except Exception:
        pass

def mean(xs):
    return sum(xs) / len(xs) if xs else None

def median(xs):
    return statistics.median(xs) if xs else None

n = len(rows)
succ = [r for r in rows if float(r.get("success", 0.0)) >= 0.5]
fail = n - len(succ)
ne_vals = [float(r["ne"]) for r in rows if r.get("ne") is not None]
spl_vals = [float(r["spl"]) for r in rows if r.get("spl") is not None]
os_vals = [float(r["os"]) for r in rows if r.get("os") is not None]
pl_vals = [float(r["path_length"]) for r in rows if r.get("path_length") is not None]
st_vals = [float(r["steps"]) for r in rows if r.get("steps") is not None]

scene_rows = []
for scene, items in per_scene.items():
    total = len(items)
    s = sum(1 for r in items if float(r.get("success", 0.0)) >= 0.5)
    scene_rows.append({
        "scene_id": scene,
        "episodes": total,
        "success": s,
        "failure": total - s,
        "sr": (s / total) if total else None,
        "ne_mean": mean([float(r["ne"]) for r in items if r.get("ne") is not None]),
    })

scene_rows.sort(key=lambda x: ((1e9 if x["sr"] is None else x["sr"]), -x["episodes"], x["scene_id"]))

out = {
    "episodes": n,
    "success": len(succ),
    "failure": fail,
    "sr": (len(succ) / n) if n else None,
    "ne_mean": mean(ne_vals),
    "ne_median": median(ne_vals),
    "spl_mean": mean(spl_vals),
    "os_mean": mean(os_vals),
    "path_length_mean": mean(pl_vals),
    "steps_mean": mean(st_vals),
    "worst_scenes": scene_rows[:5],
}
print(json.dumps(out, ensure_ascii=False))
"""


def build_remote_cmd(run_dir: str, python_bin: str) -> str:
    code = REMOTE_AGG_CODE.replace("RUN_DIR", json.dumps(run_dir))
    return f"{shlex.quote(python_bin)} -c {shlex.quote(code)}"


def compute_local_summary(run_dir: str) -> dict:
    from pathlib import Path

    root = Path(run_dir) / "ce_run"
    rows = []
    per_scene = {}

    for p in sorted(root.glob("chunk_*/progress.jsonl")):
        try:
            with p.open() as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        row = json.loads(line)
                    except Exception:
                        continue
                    rows.append(row)
                    scene = str(row.get("scene_id", "unknown"))
                    per_scene.setdefault(scene, []).append(row)
        except Exception:
            pass

    def mean(xs):
        return sum(xs) / len(xs) if xs else None

    def median(xs):
        return statistics.median(xs) if xs else None

    n = len(rows)
    succ = [r for r in rows if float(r.get("success", 0.0)) >= 0.5]
    fail = n - len(succ)
    ne_vals = [float(r["ne"]) for r in rows if r.get("ne") is not None]
    spl_vals = [float(r["spl"]) for r in rows if r.get("spl") is not None]
    os_vals = [float(r["os"]) for r in rows if r.get("os") is not None]
    pl_vals = [float(r["path_length"]) for r in rows if r.get("path_length") is not None]
    st_vals = [float(r["steps"]) for r in rows if r.get("steps") is not None]

    scene_rows = []
    for scene, items in per_scene.items():
        total = len(items)
        s = sum(1 for r in items if float(r.get("success", 0.0)) >= 0.5)
        scene_rows.append(
            {
                "scene_id": scene,
                "episodes": total,
                "success": s,
                "failure": total - s,
                "sr": (s / total) if total else None,
                "ne_mean": mean(
                    [float(r["ne"]) for r in items if r.get("ne") is not None]
                ),
            }
        )

    scene_rows.sort(
        key=lambda x: ((1e9 if x["sr"] is None else x["sr"]), -x["episodes"], x["scene_id"])
    )

    return {
        "episodes": n,
        "success": len(succ),
        "failure": fail,
        "sr": (len(succ) / n) if n else None,
        "ne_mean": mean(ne_vals),
        "ne_median": median(ne_vals),
        "spl_mean": mean(spl_vals),
        "os_mean": mean(os_vals),
        "path_length_mean": mean(pl_vals),
        "steps_mean": mean(st_vals),
        "worst_scenes": scene_rows[:5],
    }


def fetch_summary(host: str, port: int, run_dir: str, python_bin: str) -> dict:
    cmd = [
        "ssh",
        "-p",
        str(port),
        f"root@{host}",
        build_remote_cmd(run_dir, python_bin),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip() or proc.stdout.strip() or f"ssh exit {proc.returncode}")
    return json.loads(proc.stdout.strip())


def fmt(v, nd=4):
    if v is None:
        return "-"
    if isinstance(v, float):
        return f"{v:.{nd}f}"
    return str(v)


def print_summary(summary: dict) -> None:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = (
        f"[{ts}] episodes={summary['episodes']} "
        f"success={summary['success']} failure={summary['failure']} "
        f"sr={fmt(summary['sr'])} ne_mean={fmt(summary['ne_mean'])} "
        f"ne_median={fmt(summary['ne_median'])} spl={fmt(summary['spl_mean'])} "
        f"os={fmt(summary['os_mean'])} steps={fmt(summary['steps_mean'], 2)}"
    )
    print(line, flush=True)
    worst = summary.get("worst_scenes") or []
    if worst:
        parts = []
        for row in worst:
            parts.append(
                f"{row['scene_id']} sr={fmt(row['sr'])} ep={row['episodes']} ne={fmt(row['ne_mean'])}"
            )
        print("  worst_scenes: " + " | ".join(parts), flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="116.238.240.2")
    ap.add_argument("--port", type=int, default=30524)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--python-bin", default=DEFAULT_PYTHON)
    ap.add_argument("--interval", type=float, default=30.0)
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--local-read", action="store_true")
    args = ap.parse_args()

    while True:
        try:
            if args.local_read:
                summary = compute_local_summary(args.run_dir)
            else:
                summary = fetch_summary(args.host, args.port, args.run_dir, args.python_bin)
            print_summary(summary)
        except KeyboardInterrupt:
            return 130
        except Exception as e:
            ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(f"[{ts}] ERROR {e}", file=sys.stderr, flush=True)

        if args.once:
            return 0
        time.sleep(args.interval)


if __name__ == "__main__":
    raise SystemExit(main())
