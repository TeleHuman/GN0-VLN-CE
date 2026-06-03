#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable


ERROR_PATTERNS = (
    "Fatal Python error",
    "Traceback (most recent call last)",
    "RuntimeError:",
    "ValueError:",
    "AssertionError:",
    "Exception:",
    "Segmentation fault",
    "Aborted (core dumped)",
    "core dumped",
    "Killed",
    "No such file or directory",
    "Resource temporarily unavailable",
    "PanicException",
    "TypeError:",
)

LAUNCH_RE = re.compile(r"launch split_id=(?P<split_id>\d+)\s+gpu=(?P<gpu>\S+)")
SINGLE_WORKER_EXIT_RE = re.compile(
    r"single_worker_exit split_id=(?P<split_id>\d+)\s+status=(?P<status>\d+)\s+attempt=(?P<attempt>\d+)\s+restart_count=(?P<restart_count>\d+)"
)
CHUNK_RE = re.compile(r"chunk_(\d+)$")


@dataclass(frozen=True)
class ChunkStatus:
    key: str
    run_name: str
    chunk_name: str
    chunk_index: int
    split_id: int | None
    gpu: str | None
    output_dir: str
    eval_pid: int | None
    pid_alive: bool
    runner_size: int
    runner_mtime: float
    progress_size: int
    progress_mtime: float
    state: str
    reason: str | None
    progress_count: int
    result_exists: bool
    sample_dir_count: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Monitor chunk-based CE eval or DAgger runs for crashes, exits, and stalls."
    )
    parser.add_argument("--run-dir", required=True, help="Run root, e.g. .../ce_dagger_xxx")
    parser.add_argument("--interval", type=float, default=20.0, help="Polling interval in seconds")
    parser.add_argument("--chunk-start", type=int, default=None, help="Optional first chunk index")
    parser.add_argument("--local-chunks", type=int, default=None, help="Optional number of chunks")
    parser.add_argument(
        "--stale-seconds",
        type=float,
        default=1800.0,
        help="Alert if both runner.log and progress.jsonl stay unchanged beyond this many seconds while process is alive.",
    )
    parser.add_argument("--summary-every", type=int, default=6, help="Print summary every N rounds")
    parser.add_argument("--tail-bytes", type=int, default=131072, help="Tail bytes to inspect")
    parser.add_argument("--report-file", default=None, help="Optional jsonl report file")
    parser.add_argument("--once", action="store_true", help="Print one snapshot and exit")
    parser.add_argument("--exit-on-problem", action="store_true", help="Exit code 1 on failed/died/stale")
    return parser.parse_args()


def now_ts() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def log(level: str, msg: str) -> None:
    print(f"[{now_ts()}] [{level}] {msg}", flush=True)


def write_report(path: Path, event: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(event, ensure_ascii=False) + "\n")


def iter_proc_cmdlines() -> Iterable[tuple[int, list[str]]]:
    proc_root = Path("/proc")
    for entry in proc_root.iterdir():
        if not entry.name.isdigit():
            continue
        cmdline_path = entry / "cmdline"
        try:
            raw = cmdline_path.read_bytes()
        except OSError:
            continue
        if not raw:
            continue
        argv = [part.decode("utf-8", errors="replace") for part in raw.split(b"\x00") if part]
        if argv:
            yield int(entry.name), argv


def get_arg_value(argv: list[str], key: str) -> str | None:
    for idx, token in enumerate(argv[:-1]):
        if token == key:
            return argv[idx + 1]
    return None


def looks_like_eval_worker(argv: list[str]) -> bool:
    joined = " ".join(argv)
    return (
        "HabitatTools.cli.eval_ce_aligned" in joined
        or "eval_ce_aligned.py" in joined
    )


def find_eval_pid(output_dir: Path) -> int | None:
    output_dir_str = str(output_dir)
    matches: list[int] = []
    for pid, argv in iter_proc_cmdlines():
        if not looks_like_eval_worker(argv):
            continue
        if get_arg_value(argv, "--output-dir") == output_dir_str:
            matches.append(pid)
    if not matches:
        return None
    return min(matches)


def count_lines(path: Path) -> int:
    if not path.exists():
        return 0
    count = 0
    with path.open("rb") as f:
        for _ in f:
            count += 1
    return count


def read_tail(path: Path, tail_bytes: int) -> tuple[str, int, float]:
    if not path.exists():
        return "", 0, 0.0
    stat = path.stat()
    size = stat.st_size
    mtime = stat.st_mtime
    with path.open("rb") as f:
        if size > tail_bytes:
            f.seek(size - tail_bytes)
        tail = f.read()
    return tail.decode("utf-8", errors="replace"), size, mtime


def read_head(path: Path, head_bytes: int = 8192) -> tuple[str, int, float]:
    if not path.exists():
        return "", 0, 0.0
    stat = path.stat()
    size = stat.st_size
    mtime = stat.st_mtime
    with path.open("rb") as f:
        head = f.read(head_bytes)
    return head.decode("utf-8", errors="replace"), size, mtime


def extract_reason(tail_text: str) -> str | None:
    lines = [line.strip() for line in tail_text.splitlines() if line.strip()]
    if not lines:
        return None
    for line in reversed(lines):
        for pattern in ERROR_PATTERNS:
            if pattern in line:
                return line[:500]
    return lines[-1][:500]


def parse_chunk_index(chunk_dir: Path) -> int | None:
    match = CHUNK_RE.fullmatch(chunk_dir.name)
    if match is None:
        return None
    return int(match.group(1))


def should_keep_chunk(chunk_index: int | None, chunk_start: int | None, local_chunks: int | None) -> bool:
    if chunk_index is None:
        return True
    if chunk_start is None and local_chunks is None:
        return True
    start = 0 if chunk_start is None else chunk_start
    if local_chunks is None:
        return chunk_index >= start
    return start <= chunk_index < start + local_chunks


def iter_chunk_dirs(run_dir: Path) -> list[Path]:
    ce_run = run_dir / "ce_run"
    paths: list[Path] = []
    for root in (ce_run,):
        if root.exists():
            paths.extend(sorted(root.glob("chunk_*")))
    if paths:
        return paths
    paths.extend(sorted(run_dir.glob("chunk_*")))
    return paths


def sample_dir_count(chunk_dir: Path) -> int:
    return sum(1 for p in chunk_dir.iterdir() if p.is_dir() and "_" in p.name) if chunk_dir.exists() else 0


def parse_chunk_status(chunk_dir: Path, stale_seconds: float, tail_bytes: int) -> ChunkStatus:
    chunk_index = parse_chunk_index(chunk_dir)
    runner_log = chunk_dir / "runner.log"
    progress_path = chunk_dir / "progress.jsonl"
    result_path = chunk_dir / "result.json"

    head_text, runner_size, runner_mtime = read_head(runner_log)
    tail_text, _, _ = read_tail(runner_log, tail_bytes=tail_bytes)
    _, progress_size, progress_mtime = read_head(progress_path)
    eval_pid = find_eval_pid(chunk_dir)
    pid_alive = eval_pid is not None
    progress_count = count_lines(progress_path)
    result_exists = result_path.exists()
    run_name = chunk_dir.parent.parent.name if chunk_dir.parent.parent.exists() else chunk_dir.parent.name

    split_id = None
    gpu = None
    launch_match = LAUNCH_RE.search(head_text) or LAUNCH_RE.search(tail_text)
    if launch_match is not None:
        split_id = int(launch_match.group("split_id"))
        gpu = launch_match.group("gpu")

    reason = None
    state = "unknown"
    now = time.time()
    last_activity = max(runner_mtime, progress_mtime)

    if pid_alive:
        if stale_seconds > 0 and last_activity > 0 and now - last_activity > stale_seconds:
            state = "stale"
            reason = (
                f"no runner/progress update for {int(now - last_activity)}s "
                f"(progress_count={progress_count}, sample_dirs={sample_dir_count(chunk_dir)})"
            )
        else:
            state = "running"
    else:
        exit_match = None
        for m in SINGLE_WORKER_EXIT_RE.finditer(tail_text):
            exit_match = m
        if result_exists:
            state = "completed"
        elif exit_match is not None:
            exit_status = int(exit_match.group("status"))
            state = "completed" if exit_status == 0 else "failed"
            if exit_status != 0:
                reason = extract_reason(tail_text)
        elif runner_log.exists():
            state = "died"
            reason = extract_reason(tail_text)
        else:
            state = "missing"

    key = f"{run_name}/{chunk_dir.name}"
    return ChunkStatus(
        key=key,
        run_name=run_name,
        chunk_name=chunk_dir.name,
        chunk_index=-1 if chunk_index is None else chunk_index,
        split_id=split_id,
        gpu=gpu,
        output_dir=str(chunk_dir),
        eval_pid=eval_pid,
        pid_alive=pid_alive,
        runner_size=runner_size,
        runner_mtime=runner_mtime,
        progress_size=progress_size,
        progress_mtime=progress_mtime,
        state=state,
        reason=reason,
        progress_count=progress_count,
        result_exists=result_exists,
        sample_dir_count=sample_dir_count(chunk_dir),
    )


def collect_statuses(run_dir: Path, chunk_start: int | None, local_chunks: int | None, stale_seconds: float, tail_bytes: int) -> list[ChunkStatus]:
    statuses: list[ChunkStatus] = []
    for chunk_dir in iter_chunk_dirs(run_dir):
        chunk_index = parse_chunk_index(chunk_dir)
        if not should_keep_chunk(chunk_index, chunk_start=chunk_start, local_chunks=local_chunks):
            continue
        statuses.append(parse_chunk_status(chunk_dir, stale_seconds=stale_seconds, tail_bytes=tail_bytes))
    return statuses


def format_summary(statuses: list[ChunkStatus]) -> str:
    counter = Counter(status.state for status in statuses)
    pieces = [f"total={len(statuses)}"]
    for state in ("running", "completed", "stale", "failed", "died", "missing", "unknown"):
        if counter.get(state, 0):
            pieces.append(f"{state}={counter[state]}")
    return " ".join(pieces)


def transition_signature(status: ChunkStatus) -> tuple[object, ...]:
    return (
        status.state,
        status.eval_pid,
        status.progress_count,
        status.result_exists,
        status.sample_dir_count,
        status.reason,
        status.runner_size,
        status.progress_size,
    )


def main() -> int:
    args = parse_args()
    run_dir = Path(args.run_dir).expanduser().resolve()
    if not run_dir.exists():
        log("ERROR", f"run_dir not found: {run_dir}")
        return 2

    report_file = Path(args.report_file).expanduser().resolve() if args.report_file else None
    seen: dict[str, tuple[object, ...]] = {}
    iteration = 0

    while True:
        statuses = collect_statuses(
            run_dir,
            chunk_start=args.chunk_start,
            local_chunks=args.local_chunks,
            stale_seconds=args.stale_seconds,
            tail_bytes=args.tail_bytes,
        )
        if not statuses:
            log("WARN", f"no chunk_* found under {run_dir}")
            if args.once:
                return 1
            time.sleep(args.interval)
            continue

        problem = False
        for status in statuses:
            signature = transition_signature(status)
            previous = seen.get(status.key)
            if previous != signature:
                prev_state = previous[0] if previous else None
                msg = (
                    f"{status.key} split_id={status.split_id} gpu={status.gpu} "
                    f"state={status.state} pid={status.eval_pid} progress={status.progress_count} "
                    f"samples={status.sample_dir_count} result={int(status.result_exists)}"
                )
                if prev_state is not None:
                    msg += f" prev={prev_state}"
                if status.reason:
                    msg += f" reason={status.reason}"
                level = "ALERT" if status.state in {"failed", "died", "stale"} else "INFO"
                log(level, msg)
                if report_file is not None:
                    write_report(
                        report_file,
                        {
                            "ts": now_ts(),
                            "previous_state": prev_state,
                            "status": asdict(status),
                            "message": msg,
                        },
                    )
            seen[status.key] = signature
            if status.state in {"failed", "died", "stale"}:
                problem = True

        if iteration % max(args.summary_every, 1) == 0 or args.once:
            log("SUMMARY", format_summary(statuses))

        if args.once:
            return 1 if problem and args.exit_on_problem else 0
        if problem and args.exit_on_problem:
            log("ERROR", "detected failed/died/stale chunk, exiting due to --exit-on-problem")
            return 1

        iteration += 1
        time.sleep(args.interval)


if __name__ == "__main__":
    raise SystemExit(main())
