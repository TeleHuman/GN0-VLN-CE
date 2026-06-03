#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from HabitatTools.agents import BAEV3Agent
from HabitatTools.config import EvalRuntimeConfig
from HabitatTools.env import build_habitat_env
from HabitatTools.evaluators import evaluate_ce


def parse_args() -> EvalRuntimeConfig:
    parser = argparse.ArgumentParser(description="Habitat CE evaluator for GN0-VLN-CE")
    parser.add_argument("--task", default="vlnce", choices=["vlnce"])
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--habitat-config-path", required=True)
    parser.add_argument("--eval-split", default="val_unseen")
    parser.add_argument("--dataset-data-path", default="")
    parser.add_argument("--scenes-dir", default="")
    parser.add_argument("--split-num", type=int, default=1)
    parser.add_argument("--split-id", type=int, default=0)
    parser.add_argument("--start-idx", type=int, default=0)
    parser.add_argument("--end-idx", type=int, default=-1)
    parser.add_argument("--max-episodes", type=int, default=0)
    parser.add_argument("--max-steps-per-episode", type=int, default=500)
    parser.add_argument("--early-stop-rotation", type=int, default=25)
    parser.add_argument("--episode-keys-json", default="")
    parser.add_argument("--resume", action="store_true")

    parser.add_argument("--prompt-type", default="V3HF", choices=["V3HF"])
    parser.add_argument("--action-num", type=int, default=1)
    parser.add_argument("--history-len", type=int, default=16)
    parser.add_argument("--fallback-action", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--load-dtype", default="bf16", choices=["bf16", "fp16"])
    parser.add_argument("--device", default="cuda:0")

    args = parser.parse_args()

    args.prompt_type = "V3HF"

    return EvalRuntimeConfig(
        task=args.task,
        model_path=args.model_path,
        output_dir=args.output_dir,
        habitat_config_path=args.habitat_config_path,
        eval_split=args.eval_split,
        dataset_data_path=args.dataset_data_path,
        scenes_dir=args.scenes_dir,
        split_num=args.split_num,
        split_id=args.split_id,
        start_idx=args.start_idx,
        end_idx=args.end_idx,
        max_episodes=args.max_episodes,
        max_steps_per_episode=args.max_steps_per_episode,
        early_stop_rotation=args.early_stop_rotation,
        resume=bool(args.resume),
        prompt_type=args.prompt_type,
        action_num=args.action_num,
        history_len=args.history_len,
        max_new_tokens=args.max_new_tokens,
        load_dtype=args.load_dtype,
        device=args.device,
        fallback_action=args.fallback_action,
    ), args.episode_keys_json


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
    cfg, episode_keys_json = parse_args()
    gn0_vln_ce_root = _ensure_import_paths()
    _register_habitat_extensions()

    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "run_args.json").write_text(
        json.dumps(cfg.__dict__, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    env = build_habitat_env(
        habitat_config_path=cfg.habitat_config_path,
        eval_split=cfg.eval_split,
        dataset_data_path=cfg.dataset_data_path,
        scenes_dir=cfg.scenes_dir,
        repo_root=str(gn0_vln_ce_root),
        task=cfg.task,
    )

    agent = BAEV3Agent(
        model_path=cfg.model_path,
        output_dir=cfg.output_dir,
        load_dtype=cfg.load_dtype,
        max_new_tokens=cfg.max_new_tokens,
        action_num=cfg.action_num,
        history_len=cfg.history_len,
        fallback_action=cfg.fallback_action,
        ignore_stop_on_step0=True,
        step0_override_action=1,
    )

    try:
        rows, summary = evaluate_ce(
            env=env,
            agent=agent,
            output_dir=output_dir,
            eval_split=cfg.eval_split,
            start_idx=cfg.start_idx,
            end_idx=cfg.end_idx,
            max_episodes=cfg.max_episodes,
            split_num=cfg.split_num,
            split_id=cfg.split_id,
            max_steps_per_episode=cfg.max_steps_per_episode,
            early_stop_rotation=cfg.early_stop_rotation,
            resume=cfg.resume,
            episode_keys_json=episode_keys_json,
        )
        print(json.dumps(summary, indent=2, ensure_ascii=False))
        print(f"rows={len(rows)} output={output_dir}")
        return 0
    finally:
        env.close()


if __name__ == "__main__":
    raise SystemExit(main())
