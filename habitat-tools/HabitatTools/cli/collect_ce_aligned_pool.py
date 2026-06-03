#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import socket
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

from HabitatTools.agents import BAECEDaggerAlignedAgent
from HabitatTools.cli.eval_ce_aligned import (
    FROZEN_PROTOCOL,
    _ensure_import_paths,
    _parse_astar_margins,
    _register_habitat_extensions,
)
from HabitatTools.datasets import (
    extract_distance_to_goal,
    get_episode_instruction,
    get_scene_episode_key,
    load_episode_key_filter,
)
from HabitatTools.env import build_habitat_env
from HabitatTools.evaluators.ce_evaluator import (
    _extract_metrics,
    _get_runtime_episode,
    _is_multifloor_episode,
    _reset_on_target_episode,
    _step_env,
)
from HabitatTools.io.progress import load_done_episodes
from HabitatTools.io.results import write_episode_log, write_json
from HabitatTools.metrics import summarize_ce, write_ce_outputs
from HabitatTools.utils.actions import ACTION_STOP


def parse_args():
    parser = argparse.ArgumentParser(
        description="CE aligned pooled DAgger collector. "
        "Each GPU process keeps one model loaded and claims scene-grouped bundles "
        "from a shared task pool."
    )
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--worker-name", required=True)
    parser.add_argument("--habitat-config-path", required=True)
    parser.add_argument("--scenes-dir", default="")
    parser.add_argument("--dataset-data-path", default="")
    parser.add_argument("--start-idx", type=int, default=0)
    parser.add_argument("--end-idx", type=int, default=-1)
    parser.add_argument("--max-episodes", type=int, default=0)
    parser.add_argument("--episode-keys-json", default="")
    parser.add_argument("--load-dtype", default="bf16", choices=["bf16", "fp16"])
    parser.add_argument("--occupancy-root", default="/mnt/data/GN0-VLN-CE/data/scene_datasets/mp3d_ce_occ")
    parser.add_argument("--prompt-type", default="V3HF", choices=["V3HF"])
    parser.set_defaults(allow_sliding=None)
    parser.add_argument("--allow-sliding", dest="allow_sliding", action="store_true")
    parser.add_argument("--no-allow-sliding", dest="allow_sliding", action="store_false")
    parser.add_argument("--astar-margins", default="9,8,7,6,5,4")
    parser.add_argument("--skip-multifloor", action="store_true")
    parser.add_argument("--multifloor-height-threshold", type=float, default=0.75)
    parser.add_argument("--bundle-size", type=int, default=10)
    parser.add_argument("--claim-timeout-sec", type=float, default=1800.0)
    parser.add_argument("--init-timeout-sec", type=float, default=300.0)
    parser.add_argument("--claim-same-scene-first", action="store_true", default=True)
    parser.add_argument("--no-claim-same-scene-first", dest="claim_same_scene_first", action="store_false")
    return parser.parse_args()


def _atomic_lock(lock_path: Path, timeout_sec: float):
    deadline = time.time() + float(timeout_sec)
    while True:
        try:
            fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            payload = {
                "pid": os.getpid(),
                "host": socket.gethostname(),
                "created_at": time.time(),
            }
            os.write(fd, json.dumps(payload).encode("utf-8"))
            os.close(fd)
            return
        except FileExistsError:
            if time.time() > deadline:
                raise TimeoutError(f"lock timeout: {lock_path}")
            time.sleep(0.1)


def _unlock(lock_path: Path) -> None:
    try:
        lock_path.unlink()
    except FileNotFoundError:
        pass


def _ensure_pool_initialized(
    pool_dir: Path,
    episodes: list[Any],
    bundle_size: int,
    worker_name: str,
    timeout_sec: float,
) -> dict:
    pool_dir.mkdir(parents=True, exist_ok=True)
    bundles_dir = pool_dir / "bundles"
    claims_dir = pool_dir / "claims"
    done_dir = pool_dir / "done"
    manifest_path = pool_dir / "manifest.json"
    ready_path = pool_dir / "READY"
    init_lock = pool_dir / "INIT.lock"

    if ready_path.exists() and manifest_path.exists():
        return json.loads(manifest_path.read_text(encoding="utf-8"))

    try:
        _atomic_lock(init_lock, timeout_sec=timeout_sec)
        acquired = True
    except TimeoutError:
        acquired = False

    if not acquired:
        deadline = time.time() + float(timeout_sec)
        while time.time() <= deadline:
            if ready_path.exists() and manifest_path.exists():
                return json.loads(manifest_path.read_text(encoding="utf-8"))
            time.sleep(0.2)
        raise TimeoutError("timed out waiting for pool initialization")

    try:
        if ready_path.exists() and manifest_path.exists():
            return json.loads(manifest_path.read_text(encoding="utf-8"))

        bundles_dir.mkdir(parents=True, exist_ok=True)
        claims_dir.mkdir(parents=True, exist_ok=True)
        done_dir.mkdir(parents=True, exist_ok=True)

        grouped: dict[str, list[tuple[str, int]]] = defaultdict(list)
        for episode in episodes:
            grouped[get_scene_episode_key(episode)[0]].append(get_scene_episode_key(episode))

        manifests: list[dict[str, Any]] = []
        for scene_id in sorted(grouped):
            scene_keys = grouped[scene_id]
            for bundle_idx, start in enumerate(range(0, len(scene_keys), int(bundle_size))):
                chunk = scene_keys[start : start + int(bundle_size)]
                bundle_id = f"{scene_id}__{bundle_idx:04d}"
                payload = {
                    "bundle_id": bundle_id,
                    "scene_id": scene_id,
                    "episode_keys": [
                        {"scene_id": s, "episode_id": int(eid)} for s, eid in chunk
                    ],
                    "episode_count": len(chunk),
                }
                (bundles_dir / f"{bundle_id}.json").write_text(
                    json.dumps(payload, indent=2, ensure_ascii=False),
                    encoding="utf-8",
                )
                manifests.append(payload)

        manifest = {
            "worker_initialized_by": worker_name,
            "bundle_size": int(bundle_size),
            "bundle_count": len(manifests),
            "episode_count": len(episodes),
            "bundles": [
                {
                    "bundle_id": item["bundle_id"],
                    "scene_id": item["scene_id"],
                    "episode_count": item["episode_count"],
                }
                for item in manifests
            ],
        }
        manifest_path.write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        ready_path.write_text(
            json.dumps(
                {
                    "worker_name": worker_name,
                    "initialized_at": time.time(),
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        return manifest
    finally:
        _unlock(init_lock)


def _candidate_episodes(
    env,
    start_idx: int,
    end_idx: int,
    max_episodes: int,
    episode_keys_json: str,
    skip_multifloor: bool,
    multifloor_height_threshold: float,
) -> list[Any]:
    all_eps = list(env.episodes)
    if end_idx < 0:
        end_idx = len(all_eps)
    scoped = all_eps[start_idx:end_idx]
    if max_episodes > 0:
        scoped = scoped[:max_episodes]

    episode_filter = load_episode_key_filter(episode_keys_json)
    selected: list[Any] = []
    for episode in scoped:
        key = get_scene_episode_key(episode)
        if episode_filter is not None and key not in episode_filter:
            continue
        if skip_multifloor:
            is_multifloor, _ = _is_multifloor_episode(episode, multifloor_height_threshold)
            if is_multifloor:
                continue
        selected.append(episode)
    return selected


def _claim_next_bundle(
    pool_dir: Path,
    bundle_specs: list[dict[str, Any]],
    worker_name: str,
    claim_timeout_sec: float,
    preferred_scene_id: str | None,
    claim_same_scene_first: bool,
) -> dict[str, Any] | None:
    bundles_dir = pool_dir / "bundles"
    claims_dir = pool_dir / "claims"
    done_dir = pool_dir / "done"

    def iter_specs():
        if preferred_scene_id and claim_same_scene_first:
            for item in bundle_specs:
                if item["scene_id"] == preferred_scene_id:
                    yield item
        for item in bundle_specs:
            if preferred_scene_id and claim_same_scene_first and item["scene_id"] == preferred_scene_id:
                continue
            yield item

    for spec in iter_specs():
        bundle_id = str(spec["bundle_id"])
        done_path = done_dir / f"{bundle_id}.done"
        claim_path = claims_dir / f"{bundle_id}.claim"
        bundle_path = bundles_dir / f"{bundle_id}.json"
        if done_path.exists() or not bundle_path.exists():
            continue
        if claim_path.exists():
            try:
                age = time.time() - claim_path.stat().st_mtime
                if age > float(claim_timeout_sec):
                    claim_path.unlink()
                else:
                    continue
            except FileNotFoundError:
                pass
        try:
            fd = os.open(str(claim_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(
                fd,
                json.dumps(
                    {
                        "worker_name": worker_name,
                        "host": socket.gethostname(),
                        "pid": os.getpid(),
                        "claimed_at": time.time(),
                        "scene_id": spec["scene_id"],
                    },
                    ensure_ascii=False,
                ).encode("utf-8"),
            )
            os.close(fd)
        except FileExistsError:
            continue
        return json.loads(bundle_path.read_text(encoding="utf-8"))

    return None


def _complete_bundle(pool_dir: Path, bundle_id: str, worker_name: str, processed_rows: int) -> None:
    claims_dir = pool_dir / "claims"
    done_dir = pool_dir / "done"
    claim_path = claims_dir / f"{bundle_id}.claim"
    done_path = done_dir / f"{bundle_id}.done"
    done_path.write_text(
        json.dumps(
            {
                "worker_name": worker_name,
                "host": socket.gethostname(),
                "pid": os.getpid(),
                "completed_at": time.time(),
                "processed_rows": int(processed_rows),
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    try:
        claim_path.unlink()
    except FileNotFoundError:
        pass


def _append_shared_row(output_dir: Path, row: dict) -> bool:
    progress_path = output_dir / "progress.jsonl"
    log_dir = output_dir / "log"
    lock_path = output_dir / "_pool" / "progress.lock"
    log_path = log_dir / f"{row['id']}.json"

    _atomic_lock(lock_path, timeout_sec=300.0)
    try:
        if log_path.exists():
            return False
        progress_path.parent.mkdir(parents=True, exist_ok=True)
        with progress_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        write_episode_log(log_dir, row["id"], row)
        return True
    finally:
        _unlock(lock_path)


def _maybe_finalize(output_dir: Path, pool_dir: Path) -> None:
    manifest_path = pool_dir / "manifest.json"
    done_dir = pool_dir / "done"
    finalized_path = pool_dir / "FINALIZED.json"
    finalize_lock = pool_dir / "FINALIZE.lock"

    if finalized_path.exists() or not manifest_path.exists():
        return
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected = int(manifest.get("bundle_count", 0))
    done_count = len(list(done_dir.glob("*.done")))
    if done_count < expected:
        return

    try:
        _atomic_lock(finalize_lock, timeout_sec=1.0)
    except TimeoutError:
        return

    try:
        if finalized_path.exists():
            return
        _, rows = load_done_episodes(output_dir / "progress.jsonl")
        _, _, summary = write_ce_outputs(output_dir, rows, split_name=FROZEN_PROTOCOL["eval_split"])
        summary["processed_episodes"] = len(rows)
        summary["pool"] = {
            "bundle_count": expected,
            "done_count": done_count,
        }
        write_json(output_dir / "result.json", summary)
        finalized_path.write_text(
            json.dumps(
                {
                    "finalized_at": time.time(),
                    "host": socket.gethostname(),
                    "pid": os.getpid(),
                    "summary": summary,
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
    finally:
        _unlock(finalize_lock)


def _process_bundle(
    bundle: dict[str, Any],
    env,
    agent,
    episode_map: dict[tuple[str, int], Any],
    output_dir: Path,
    worker_status_path: Path,
) -> int:
    processed_rows = 0
    status_payload = {
        "worker_name": worker_status_path.parent.name,
        "bundle_id": bundle["bundle_id"],
        "scene_id": bundle["scene_id"],
        "updated_at": time.time(),
        "processed_rows": 0,
    }
    worker_status_path.write_text(
        json.dumps(status_payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    for item in bundle["episode_keys"]:
        key = (str(item["scene_id"]), int(item["episode_id"]))
        episode = episode_map.get(key)
        if episode is None:
            continue

        observations, runtime_episode = _reset_on_target_episode(env, episode)
        scene_id, episode_id = get_scene_episode_key(runtime_episode)
        instruction = get_episode_instruction(runtime_episode)
        agent.reset(scene_id=scene_id, episode_id=episode_id, instruction=instruction)

        done = False
        step_id = 0
        last_dtg = None
        continuous_rotation_count = 0
        while (not done) and (step_id < int(FROZEN_PROTOCOL["max_steps_per_episode"])):
            info = _extract_metrics(env)
            curr_dtg = extract_distance_to_goal(info)
            if last_dtg is None or curr_dtg is None or curr_dtg != last_dtg:
                continuous_rotation_count = 0
            else:
                continuous_rotation_count += 1
            last_dtg = curr_dtg

            if (
                int(FROZEN_PROTOCOL["early_stop_rotation"]) >= 0
                and continuous_rotation_count > int(FROZEN_PROTOCOL["early_stop_rotation"])
            ):
                action = ACTION_STOP
            else:
                if isinstance(observations, dict):
                    agent_obs = dict(observations)
                    agent_obs["metrics"] = info
                else:
                    agent_obs = {"rgb": observations, "metrics": info}
                action = agent.act(agent_obs, instruction=instruction, step_id=step_id)

            observations, done = _step_env(env, action)
            step_id += 1

        metrics = _extract_metrics(env)
        row = {
            "id": f"{scene_id}_{episode_id:04d}",
            "split": FROZEN_PROTOCOL["eval_split"],
            "scene_id": scene_id,
            "episode_id": episode_id,
            "success": float(metrics.get("success", 0.0)),
            "spl": float(metrics.get("spl", 0.0)),
            "os": float(metrics.get("oracle_success", 0.0)),
            "ne": float(metrics.get("distance_to_goal", 0.0)),
            "path_length": float(metrics.get("path_length", 0.0)),
            "steps": int(step_id),
            "instruction": instruction,
        }
        if _append_shared_row(output_dir, row):
            processed_rows += 1
            status_payload["processed_rows"] = processed_rows
            status_payload["updated_at"] = time.time()
            worker_status_path.write_text(
                json.dumps(status_payload, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )

    return processed_rows


def main() -> int:
    args = parse_args()
    gn0_vln_ce_root = _ensure_import_paths()
    _register_habitat_extensions()

    allow_sliding_effective = True if args.allow_sliding is None else bool(args.allow_sliding)
    dataset_data_path_effective = (
        args.dataset_data_path.strip()
        if str(args.dataset_data_path).strip()
        else FROZEN_PROTOCOL["dataset_data_path"]
    )
    astar_margins_effective = _parse_astar_margins(args.astar_margins)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    pool_dir = output_dir / "_pool"
    worker_dir = output_dir / "_workers" / args.worker_name
    worker_dir.mkdir(parents=True, exist_ok=True)

    run_args = {
        "engine_profile": FROZEN_PROTOCOL["engine_profile"],
        "task": FROZEN_PROTOCOL["task"],
        "model_path": args.model_path,
        "output_dir": str(output_dir),
        "worker_name": args.worker_name,
        "habitat_config_path": args.habitat_config_path,
        "eval_split": FROZEN_PROTOCOL["eval_split"],
        "dataset_data_path": dataset_data_path_effective,
        "dataset_data_path_override": (
            args.dataset_data_path if str(args.dataset_data_path).strip() else ""
        ),
        "scenes_dir": args.scenes_dir,
        "start_idx": args.start_idx,
        "end_idx": args.end_idx,
        "max_episodes": args.max_episodes,
        "load_dtype": args.load_dtype,
        "mode": "collect",
        "allow_sliding": allow_sliding_effective,
        "dagger": True,
        "occupancy_root": args.occupancy_root,
        "prompt_type": args.prompt_type,
        "astar_margins": args.astar_margins,
        "astar_margins_effective": [int(m) for m in astar_margins_effective],
        "skip_multifloor": bool(args.skip_multifloor),
        "multifloor_height_threshold": float(args.multifloor_height_threshold),
        "bundle_size": int(args.bundle_size),
        "claim_timeout_sec": float(args.claim_timeout_sec),
        "claim_same_scene_first": bool(args.claim_same_scene_first),
        "protocol": dict(FROZEN_PROTOCOL),
    }
    (output_dir / "run_args.json").write_text(
        json.dumps(run_args, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (worker_dir / "run_args.json").write_text(
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
    )

    try:
        episodes = _candidate_episodes(
            env=env,
            start_idx=args.start_idx,
            end_idx=args.end_idx,
            max_episodes=args.max_episodes,
            episode_keys_json=args.episode_keys_json,
            skip_multifloor=bool(args.skip_multifloor),
            multifloor_height_threshold=float(args.multifloor_height_threshold),
        )
        episode_map = {get_scene_episode_key(ep): ep for ep in episodes}
        manifest = _ensure_pool_initialized(
            pool_dir=pool_dir,
            episodes=episodes,
            bundle_size=int(args.bundle_size),
            worker_name=args.worker_name,
            timeout_sec=float(args.init_timeout_sec),
        )

        agent = BAECEDaggerAlignedAgent(
            env=env,
            model_path=args.model_path,
            output_dir=str(output_dir),
            occupancy_root=args.occupancy_root,
            prompt_type=args.prompt_type,
            action_num=1,
            load_dtype=args.load_dtype,
            gt_strict_coverage=False,
            astar_margins=astar_margins_effective,
        )

        worker_status_path = worker_dir / "status.json"
        total_processed = 0
        claimed_bundles = 0
        last_scene_id: str | None = None
        bundle_specs = list(manifest.get("bundles", []))

        while True:
            bundle = _claim_next_bundle(
                pool_dir=pool_dir,
                bundle_specs=bundle_specs,
                worker_name=args.worker_name,
                claim_timeout_sec=float(args.claim_timeout_sec),
                preferred_scene_id=last_scene_id,
                claim_same_scene_first=bool(args.claim_same_scene_first),
            )
            if bundle is None:
                break

            processed_rows = _process_bundle(
                bundle=bundle,
                env=env,
                agent=agent,
                episode_map=episode_map,
                output_dir=output_dir,
                worker_status_path=worker_status_path,
            )
            _complete_bundle(
                pool_dir=pool_dir,
                bundle_id=str(bundle["bundle_id"]),
                worker_name=args.worker_name,
                processed_rows=processed_rows,
            )
            total_processed += processed_rows
            claimed_bundles += 1
            last_scene_id = str(bundle.get("scene_id") or "")

        summary = {
            "worker_name": args.worker_name,
            "host": socket.gethostname(),
            "pid": os.getpid(),
            "claimed_bundles": claimed_bundles,
            "processed_rows": total_processed,
            "bundle_count_total": int(manifest.get("bundle_count", 0)),
            "episode_count_total": int(manifest.get("episode_count", 0)),
        }
        write_json(worker_dir / "summary.json", summary)
        _maybe_finalize(output_dir=output_dir, pool_dir=pool_dir)
        print(json.dumps(summary, indent=2, ensure_ascii=False))
        return 0
    finally:
        env.close()


if __name__ == "__main__":
    raise SystemExit(main())
