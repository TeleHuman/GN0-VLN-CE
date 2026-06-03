#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Validate collected DAgger sample directories and summarize health."
    )
    parser.add_argument("--run-path", required=True, help="Chunk/run directory containing episode folders.")
    parser.add_argument("--task", default="vlnce", choices=["vlnce"])
    parser.add_argument("--out", required=True, help="Path to write JSON summary.")
    parser.add_argument("--expect-episodes", type=int, default=0)
    parser.add_argument("--require-gt-rollout-valid-rate", type=float, default=1.0)
    return parser.parse_args()


def _read_json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def main() -> int:
    args = parse_args()
    run_path = Path(args.run_path)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if not run_path.exists():
        payload = {
            "status": "fail",
            "reason": "run_path_missing",
            "run_path": str(run_path),
            "task": args.task,
        }
        out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        print(json.dumps(payload, ensure_ascii=False))
        return 1

    episode_dirs = sorted([p for p in run_path.iterdir() if p.is_dir()])
    episode_count = len(episode_dirs)
    sample_files = 0
    meta_files = 0
    gt_pairs = 0
    gt_rollout_valid_true = 0
    gt_rollout_valid_total = 0
    prompt_type_counter = Counter()
    bad_sample_json = []
    missing_meta = []
    missing_sample = []

    for ep_dir in episode_dirs:
        sample_path = ep_dir / "sample.json"
        meta_path = ep_dir / "meta.json"
        if sample_path.exists():
            sample_files += 1
            sample = _read_json(sample_path)
            if isinstance(sample, list):
                for row in sample:
                    if isinstance(row, dict):
                        pt = row.get("prompt_type")
                        if isinstance(pt, str):
                            prompt_type_counter[pt] += 1
                        gt_pair = row.get("gt_true_pair")
                        if isinstance(gt_pair, dict):
                            gt_pairs += 1
                        validation = row.get("validation")
                        if isinstance(validation, dict) and "gt_rollout_valid" in validation:
                            gt_rollout_valid_total += 1
                            if bool(validation.get("gt_rollout_valid")):
                                gt_rollout_valid_true += 1
            else:
                bad_sample_json.append(str(sample_path))
        else:
            missing_sample.append(str(sample_path))

        if meta_path.exists():
            meta_files += 1
        else:
            missing_meta.append(str(meta_path))

    gt_rollout_valid_rate = (
        float(gt_rollout_valid_true) / float(gt_rollout_valid_total)
        if gt_rollout_valid_total > 0
        else None
    )

    status = "pass"
    failures = []

    if args.expect_episodes > 0 and episode_count != int(args.expect_episodes):
        status = "fail"
        failures.append(
            f"episode_count_mismatch:{episode_count}!={int(args.expect_episodes)}"
        )
    if missing_meta:
        status = "fail"
        failures.append(f"missing_meta:{len(missing_meta)}")
    if missing_sample:
        status = "fail"
        failures.append(f"missing_sample:{len(missing_sample)}")
    if bad_sample_json:
        status = "fail"
        failures.append(f"bad_sample_json:{len(bad_sample_json)}")
    if gt_rollout_valid_rate is not None and gt_rollout_valid_rate < float(
        args.require_gt_rollout_valid_rate
    ):
        status = "fail"
        failures.append(
            "gt_rollout_valid_rate_below_threshold:"
            f"{gt_rollout_valid_rate:.6f}<{float(args.require_gt_rollout_valid_rate):.6f}"
        )

    payload = {
        "status": status,
        "task": args.task,
        "run_path": str(run_path),
        "episode_count": episode_count,
        "sample_files": sample_files,
        "meta_files": meta_files,
        "gt_true_pair_rows": gt_pairs,
        "gt_rollout_valid_true": gt_rollout_valid_true,
        "gt_rollout_valid_total": gt_rollout_valid_total,
        "gt_rollout_valid_rate": gt_rollout_valid_rate,
        "prompt_type_counter": dict(prompt_type_counter),
        "missing_meta_count": len(missing_meta),
        "missing_sample_count": len(missing_sample),
        "bad_sample_json_count": len(bad_sample_json),
        "failures": failures,
    }
    if missing_meta:
        payload["missing_meta_examples"] = missing_meta[:10]
    if missing_sample:
        payload["missing_sample_examples"] = missing_sample[:10]
    if bad_sample_json:
        payload["bad_sample_json_examples"] = bad_sample_json[:10]

    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False))
    return 0 if status == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
