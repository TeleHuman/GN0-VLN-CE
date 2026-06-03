#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from HabitatTools.agents import (
    BAECEDaggerAlignedAgent,
    BAECEEvalAlignedAgent,
)
from HabitatTools.env import build_habitat_env
from HabitatTools.evaluators import evaluate_ce, evaluate_ce_queue_worker


FROZEN_PROTOCOL = {
    "engine_profile": "ce_aligned_protocol_v1",
    "task": "vlnce",
    "eval_split": "val_unseen",
    "dataset_data_path": "/mnt/data/InternRobotics/data/vln_ce/r2r/{split}/{split}.json.gz",
    "max_steps_per_episode": 500,
    "device_map": "none",
    "image_order": "current_first",
    "current_resize_w": 480,
    "current_resize_h": 360,
    "history_len": 16,
    "history_grid_size": 4,
    "history_tile_w": 160,
    "history_tile_h": 120,
    "max_new_tokens": 24,
    "plan_horizon": 6,
    "ignore_first_model_stop": True,
    "forbid_stop_before_goal": True,
    "auto_stop_distance": 3.0,
    "early_stop_rotation": -1,
    "stuck_turnaround_tricks": False,
    "stuck_pose_stop_steps": 10,
}


def _parse_astar_margins(raw: str) -> tuple[int, ...]:
    txt = str(raw or "").strip()
    if not txt:
        return (9, 8, 7, 6, 5, 4)
    vals = []
    for token in txt.split(","):
        token = token.strip()
        if not token:
            continue
        vals.append(int(token))
    if not vals:
        return (9, 8, 7, 6, 5, 4)
    return tuple(vals)


def parse_args():
    parser = argparse.ArgumentParser(description="CE aligned evaluator.")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--habitat-config-path", required=True)
    parser.add_argument("--scenes-dir", default="")
    parser.add_argument("--dataset-data-path", default="")

    parser.add_argument("--split-num", type=int, default=1)
    parser.add_argument("--split-id", type=int, default=0)
    parser.add_argument("--start-idx", type=int, default=0)
    parser.add_argument("--end-idx", type=int, default=-1)
    parser.add_argument("--max-episodes", type=int, default=0)
    parser.add_argument("--episode-keys-json", default="")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--queue-worker-id", type=int, default=0)
    parser.add_argument("--queue-worker-count", type=int, default=1)

    parser.add_argument("--load-dtype", default="bf16", choices=["bf16", "fp16"])
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--action-num", type=int, default=1)
    parser.add_argument("--fallback-action", type=int, default=0)
    parser.add_argument("--mode", default=None, choices=["collect", "eval"])
    parser.add_argument("--dagger", action="store_true")
    parser.add_argument(
        "--occupancy-root",
        default="/mnt/data/GN0-VLN-CE/data/scene_datasets/mp3d_ce_occ",
    )
    parser.add_argument(
        "--prompt-type",
        default="V3HF",
        choices=["V3HF"],
    )
    parser.set_defaults(allow_sliding=None)
    parser.add_argument("--allow-sliding", dest="allow_sliding", action="store_true")
    parser.add_argument("--no-allow-sliding", dest="allow_sliding", action="store_false")
    parser.add_argument("--gt-strict-coverage", action="store_true")
    parser.add_argument("--astar-margins", default="9,8,7,6,5,4")
    parser.add_argument("--skip-multifloor", action="store_true")
    parser.add_argument("--multifloor-height-threshold", type=float, default=0.75)
    parser.add_argument(
        "--stuck-pose-stop-steps",
        type=int,
        default=None,
        help="Eval-only early stop after N consecutive unchanged Habitat poses. "
        "Disabled for dagger collect runs unless wired explicitly.",
    )

    return parser.parse_args()


def _ensure_import_paths() -> Path:
    gn0_vln_ce_root = Path(__file__).resolve().parents[4]
    if str(gn0_vln_ce_root) not in sys.path:
        sys.path.insert(0, str(gn0_vln_ce_root))
    return gn0_vln_ce_root


def _register_habitat_extensions() -> None:
    try:
        from HabitatTools.habitat_extensions import measures as _measures  # noqa: F401
    except Exception as exc:
        print(
            f"[Warn] failed to import HabitatTools habitat extensions: {exc}. "
            "Custom Habitat measures may be unavailable.",
            file=sys.stderr,
        )


def main() -> int:
    args = parse_args()
    gn0_vln_ce_root = _ensure_import_paths()
    _register_habitat_extensions()

    agent_branch = "dagger" if args.dagger else "eval"
    run_mode = args.mode or ("collect" if args.dagger else "eval")
    allow_sliding_effective = (
        bool(run_mode == "collect")
        if args.allow_sliding is None
        else bool(args.allow_sliding)
    )
    dataset_data_path_effective = (
        args.dataset_data_path.strip()
        if str(args.dataset_data_path).strip()
        else FROZEN_PROTOCOL["dataset_data_path"]
    )
    gt_strict_effective = bool(
        args.gt_strict_coverage and args.dagger and run_mode == "collect"
    )
    astar_margins_effective = _parse_astar_margins(args.astar_margins)
    stuck_pose_stop_steps_effective = max(
        0,
        int(
            FROZEN_PROTOCOL["stuck_pose_stop_steps"]
            if args.stuck_pose_stop_steps is None
            else args.stuck_pose_stop_steps
        ),
    )
    if not args.dagger:
        stuck_pose_stop_steps_effective = 0
    if args.gt_strict_coverage and not gt_strict_effective:
        print(
            "[Warn] --gt-strict-coverage is only honored for dagger collect runs.",
            file=sys.stderr,
        )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    run_args = {
        "engine_profile": FROZEN_PROTOCOL["engine_profile"],
        "task": FROZEN_PROTOCOL["task"],
        "model_path": args.model_path,
        "output_dir": str(output_dir),
        "habitat_config_path": args.habitat_config_path,
        "eval_split": FROZEN_PROTOCOL["eval_split"],
        "dataset_data_path": dataset_data_path_effective,
        "dataset_data_path_override": (
            args.dataset_data_path if str(args.dataset_data_path).strip() else ""
        ),
        "scenes_dir": args.scenes_dir,
        "split_num": args.split_num,
        "split_id": args.split_id,
        "start_idx": args.start_idx,
        "end_idx": args.end_idx,
        "max_episodes": args.max_episodes,
        "max_steps_per_episode": FROZEN_PROTOCOL["max_steps_per_episode"],
        "early_stop_rotation": FROZEN_PROTOCOL["early_stop_rotation"],
        "stuck_pose_stop_steps_requested": args.stuck_pose_stop_steps,
        "stuck_pose_stop_steps": int(stuck_pose_stop_steps_effective),
        "episode_keys_json": args.episode_keys_json,
        "resume": bool(args.resume),
        "queue_worker_id": int(args.queue_worker_id),
        "queue_worker_count": int(args.queue_worker_count),
        "load_dtype": args.load_dtype,
        "device": args.device,
        "device_map": FROZEN_PROTOCOL["device_map"],
        "action_num": int(args.action_num),
        "fallback_action": args.fallback_action,
        "mode": run_mode,
        "agent_branch": agent_branch,
        "allow_sliding_requested": args.allow_sliding,
        "allow_sliding": allow_sliding_effective,
        "dagger": bool(args.dagger),
        "eval_branch": bool(not args.dagger),
        "occupancy_root": args.occupancy_root,
        "prompt_type": args.prompt_type,
        "gt_strict_coverage_requested": bool(args.gt_strict_coverage),
        "gt_strict_coverage": gt_strict_effective,
        "astar_margins": args.astar_margins,
        "astar_margins_effective": [int(m) for m in astar_margins_effective],
        "skip_multifloor": bool(args.skip_multifloor),
        "multifloor_height_threshold": float(args.multifloor_height_threshold),
        "protocol": dict(FROZEN_PROTOCOL),
    }
    (output_dir / "run_args.json").write_text(
        json.dumps(run_args, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    env = build_habitat_env(
        habitat_config_path=args.habitat_config_path,
        eval_split=FROZEN_PROTOCOL["eval_split"],
        dataset_data_path=dataset_data_path_effective,
        scenes_dir=args.scenes_dir,
        repo_root=str(gn0_vln_ce_root),
        task=FROZEN_PROTOCOL["task"],
        allow_sliding=allow_sliding_effective,
        split_num=int(args.split_num),
        split_id=int(args.split_id),
        start_idx=int(args.start_idx),
        end_idx=int(args.end_idx),
        max_episodes=int(args.max_episodes),
        episode_keys_json=args.episode_keys_json,
        progress_path=str(output_dir / "progress.jsonl"),
        resume=bool(args.resume),
    )

    if args.dagger:
        agent = BAECEDaggerAlignedAgent(
            env=env,
            model_path=args.model_path,
            output_dir=str(output_dir),
            occupancy_root=args.occupancy_root,
            prompt_type=args.prompt_type,
            action_num=args.action_num,
            load_dtype=args.load_dtype,
            gt_strict_coverage=gt_strict_effective,
            astar_margins=astar_margins_effective,
        )
    else:
        agent = BAECEEvalAlignedAgent(
            env=env,
            model_path=args.model_path,
            output_dir=str(output_dir),
            occupancy_root=args.occupancy_root,
            prompt_type=args.prompt_type,
            action_num=args.action_num,
            load_dtype=args.load_dtype,
            astar_margins=astar_margins_effective,
        )

    try:
        evaluate_fn = evaluate_ce_queue_worker if int(args.queue_worker_count) > 1 else evaluate_ce
        eval_kwargs = dict(
            env=env,
            agent=agent,
            output_dir=output_dir,
            eval_split=FROZEN_PROTOCOL["eval_split"],
            start_idx=args.start_idx,
            end_idx=args.end_idx,
            max_episodes=args.max_episodes,
            split_num=args.split_num,
            split_id=args.split_id,
            max_steps_per_episode=FROZEN_PROTOCOL["max_steps_per_episode"],
            early_stop_rotation=FROZEN_PROTOCOL["early_stop_rotation"],
            stuck_pose_stop_steps=int(stuck_pose_stop_steps_effective),
            resume=bool(args.resume),
            episode_keys_json=args.episode_keys_json,
            skip_multifloor=bool(args.skip_multifloor),
            multifloor_height_threshold=float(args.multifloor_height_threshold),
        )
        if evaluate_fn is evaluate_ce_queue_worker:
            eval_kwargs.update(
                queue_worker_id=int(args.queue_worker_id),
                queue_worker_count=int(args.queue_worker_count),
            )
        rows, summary = evaluate_fn(**eval_kwargs)
        print(json.dumps(summary, indent=2, ensure_ascii=False))
        print(f"rows={len(rows)} output={output_dir}")
        return 0
    finally:
        env.close()


if __name__ == "__main__":
    raise SystemExit(main())
