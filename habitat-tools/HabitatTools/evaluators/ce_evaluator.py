from __future__ import annotations

import fcntl
import json
import os
import shutil
import socket
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from tqdm import tqdm

from HabitatTools.datasets import (
    extract_distance_to_goal,
    get_episode_instruction,
    get_scene_episode_key,
    load_episode_key_filter,
    select_chunked_episodes,
)
from HabitatTools.io.progress import append_progress_row, load_done_episodes
from HabitatTools.io.results import write_episode_log, write_json
from HabitatTools.metrics import summarize_ce, write_ce_outputs
from HabitatTools.utils.actions import ACTION_STOP

POSE_UNCHANGED_POSITION_EPS = 1e-4
POSE_UNCHANGED_ROTATION_EPS = 1e-4


def _step_env(env, action: int):
    step_result = env.step(int(action))
    if isinstance(step_result, tuple) and len(step_result) == 5:
        observations, _, terminated, truncated, _ = step_result
        done = bool(terminated) or bool(truncated)
    elif isinstance(step_result, tuple) and len(step_result) == 4:
        observations, _, done, _ = step_result
        done = bool(done)
    else:
        observations = step_result
        done = bool(getattr(env, "episode_over", False))

    if isinstance(observations, tuple):
        observations = observations[0]
    return observations, done


def _extract_metrics(env) -> dict:
    metrics = env.get_metrics()
    return metrics if isinstance(metrics, dict) else {}


def _as_float_tuple(value: Any) -> tuple[float, ...] | None:
    if value is None:
        return None
    try:
        return tuple(float(x) for x in value)
    except Exception:
        return None


def _rotation_to_tuple(rotation: Any) -> tuple[float, ...] | None:
    if rotation is None:
        return None

    components = getattr(rotation, "components", None)
    if components is not None:
        values = _as_float_tuple(components)
        if values is not None:
            return values

    vector = getattr(rotation, "vector", None)
    scalar = getattr(rotation, "scalar", None)
    if vector is not None and scalar is not None:
        vector_values = _as_float_tuple(vector)
        if vector_values is not None:
            try:
                return vector_values + (float(scalar),)
            except Exception:
                pass

    if all(hasattr(rotation, attr) for attr in ("w", "x", "y", "z")):
        try:
            return (
                float(rotation.w),
                float(rotation.x),
                float(rotation.y),
                float(rotation.z),
            )
        except Exception:
            pass

    return _as_float_tuple(rotation)


def _extract_agent_pose(env) -> tuple[tuple[float, ...], tuple[float, ...]] | None:
    candidates = [
        env,
        getattr(env, "_env", None),
        getattr(env, "_task", None),
    ]
    for owner in candidates:
        if owner is None:
            continue
        for attr in ("sim", "_sim"):
            sim = getattr(owner, attr, None)
            if sim is None or not hasattr(sim, "get_agent_state"):
                continue
            try:
                state = sim.get_agent_state()
            except Exception:
                continue
            position = _as_float_tuple(getattr(state, "position", None))
            rotation = _rotation_to_tuple(getattr(state, "rotation", None))
            if position is not None and rotation is not None:
                return position, rotation
    return None


def _poses_match(
    pose_a: tuple[tuple[float, ...], tuple[float, ...]] | None,
    pose_b: tuple[tuple[float, ...], tuple[float, ...]] | None,
) -> bool:
    if pose_a is None or pose_b is None:
        return False
    pos_a, rot_a = pose_a
    pos_b, rot_b = pose_b
    if len(pos_a) != len(pos_b) or len(rot_a) != len(rot_b):
        return False
    pos_same = all(
        abs(a - b) <= POSE_UNCHANGED_POSITION_EPS for a, b in zip(pos_a, pos_b)
    )
    rot_same = all(
        abs(a - b) <= POSE_UNCHANGED_ROTATION_EPS for a, b in zip(rot_a, rot_b)
    )
    return pos_same and rot_same


def _reset_on_target_episode(env, episode):
    try:
        from habitat.core.dataset import EpisodeIterator

        env.episode_iterator = EpisodeIterator([episode], cycle=False, shuffle=False)
    except Exception:
        # Fallback for environments that do not expose EpisodeIterator.
        env.current_episode = episode

    observations = env.reset()
    if isinstance(observations, tuple):
        observations = observations[0]
    runtime_episode = _get_runtime_episode(env, fallback_episode=episode)
    return observations, runtime_episode


def _get_runtime_episode(env, fallback_episode):
    runtime_episode = getattr(env, "current_episode", None)
    return runtime_episode if runtime_episode is not None else fallback_episode


def _episode_output_id(scene_id: str, episode_id: int) -> str:
    return f"{scene_id}_{int(episode_id):04d}"


def _cleanup_incomplete_episode_output(output_dir: Path, log_dir: Path, scene_id: str, episode_id: int) -> None:
    episode_output_id = _episode_output_id(scene_id, episode_id)
    episode_dir = output_dir / episode_output_id
    episode_log = log_dir / f"{episode_output_id}.json"

    if episode_dir.exists():
        shutil.rmtree(episode_dir)
        print(f"[info][ce] removed incomplete episode dir before resume: {episode_dir}")
    if episode_log.exists():
        episode_log.unlink()
        print(f"[info][ce] removed incomplete episode log before resume: {episode_log}")


def _episode_log_has_summary(log_dir: Path, scene_id: str, episode_id: int) -> bool:
    episode_output_id = _episode_output_id(scene_id, episode_id)
    episode_log = log_dir / f"{episode_output_id}.json"
    if not episode_log.exists():
        return False
    try:
        payload = json.loads(episode_log.read_text(encoding="utf-8"))
    except Exception:
        return False
    required_keys = {"id", "scene_id", "episode_id", "success", "spl", "os", "ne", "steps"}
    if not isinstance(payload, dict) or not required_keys.issubset(payload.keys()):
        return False
    try:
        return (
            str(payload.get("scene_id")) == str(scene_id)
            and int(payload.get("episode_id")) == int(episode_id)
            and str(payload.get("id")) == episode_output_id
        )
    except Exception:
        return False


def _filter_resume_rows(
    output_dir: Path,
    log_dir: Path,
    progress_path: Path,
    rows: list[dict],
) -> tuple[set[tuple[str, int]], list[dict]]:
    kept_rows: list[dict] = []
    kept_done: set[tuple[str, int]] = set()
    mutated = False

    for row in rows:
        try:
            scene_id = str(row["scene_id"])
            episode_id = int(row["episode_id"])
        except Exception:
            mutated = True
            continue

        if _episode_log_has_summary(log_dir, scene_id, episode_id):
            kept_rows.append(row)
            kept_done.add((scene_id, episode_id))
            continue

        mutated = True
        _cleanup_incomplete_episode_output(
            output_dir=output_dir,
            log_dir=log_dir,
            scene_id=scene_id,
            episode_id=episode_id,
        )

    if mutated:
        progress_path.parent.mkdir(parents=True, exist_ok=True)
        with progress_path.open("w", encoding="utf-8") as f:
            for row in kept_rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    return kept_done, kept_rows


def _extract_start_goal_height(episode: Any) -> tuple[float | None, float | None]:
    start_position = getattr(episode, "start_position", None)
    if start_position is None and isinstance(episode, dict):
        start_position = episode.get("start_position")

    goals = getattr(episode, "goals", None)
    if goals is None and isinstance(episode, dict):
        goals = episode.get("goals")

    goal_position = None
    if isinstance(goals, list) and goals:
        goal0 = goals[0]
        if isinstance(goal0, dict):
            goal_position = goal0.get("position")
        else:
            goal_position = getattr(goal0, "position", None)

    start_h = None
    goal_h = None
    try:
        if start_position is not None and len(start_position) >= 2:
            start_h = float(start_position[1])
    except Exception:
        start_h = None
    try:
        if goal_position is not None and len(goal_position) >= 2:
            goal_h = float(goal_position[1])
    except Exception:
        goal_h = None

    return start_h, goal_h


def _is_multifloor_episode(
    episode: Any,
    height_threshold: float,
) -> tuple[bool, float | None]:
    start_h, goal_h = _extract_start_goal_height(episode)
    if start_h is None or goal_h is None:
        return False, None
    dy = abs(goal_h - start_h)
    return bool(dy > float(height_threshold)), float(dy)


@contextmanager
def _chunk_write_lock(output_dir: Path):
    queue_dir = output_dir / ".queue"
    queue_dir.mkdir(parents=True, exist_ok=True)
    lock_path = queue_dir / "write.lock"
    with lock_path.open("a", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _safe_marker_name(scene_id: str, episode_id: int) -> str:
    return _episode_output_id(scene_id, episode_id).replace("/", "__")


def _ensure_episode_queue(output_dir: Path, episodes: list[Any]) -> Path:
    queue_dir = output_dir / ".queue"
    claims_dir = queue_dir / "claims"
    done_dir = queue_dir / "done"
    queue_dir.mkdir(parents=True, exist_ok=True)
    claims_dir.mkdir(parents=True, exist_ok=True)
    done_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = queue_dir / "episodes.jsonl"
    if manifest_path.exists():
        return manifest_path

    init_lock = queue_dir / "init.lock"
    try:
        init_lock.mkdir()
    except FileExistsError:
        for _ in range(600):
            if manifest_path.exists():
                return manifest_path
            time.sleep(0.5)
        raise TimeoutError(f"timed out waiting for queue manifest: {manifest_path}")

    try:
        tmp_path = queue_dir / f"episodes.jsonl.tmp.{os.getpid()}"
        with tmp_path.open("w", encoding="utf-8") as handle:
            for idx, episode in enumerate(episodes):
                scene_id, episode_id = get_scene_episode_key(episode)
                item = {
                    "index": idx,
                    "scene_id": scene_id,
                    "episode_id": int(episode_id),
                }
                handle.write(json.dumps(item, ensure_ascii=False) + "\n")
        tmp_path.replace(manifest_path)
    finally:
        try:
            init_lock.rmdir()
        except OSError:
            pass
    return manifest_path


def _load_queue_items(manifest_path: Path) -> list[dict]:
    items: list[dict] = []
    with manifest_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
    return items


def _pid_is_alive(pid: int | None) -> bool:
    if pid is None or pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _claim_path_is_active(claim_path: Path) -> bool:
    try:
        payload = json.loads(claim_path.read_text(encoding="utf-8"))
    except Exception:
        return False
    claim_host = str(payload.get("host") or "")
    current_host = socket.gethostname()
    if claim_host and claim_host != current_host:
        stale_sec = float(os.environ.get("HABITAT_TOOLS_CLAIM_STALE_SEC", "86400"))
        try:
            age = time.time() - claim_path.stat().st_mtime
        except OSError:
            return False
        return bool(age < stale_sec)
    pid = payload.get("pid")
    try:
        pid = int(pid)
    except Exception:
        return False
    return _pid_is_alive(pid)


def _claim_next_episode(output_dir: Path, manifest_path: Path, episode_by_key: dict[tuple[str, int], Any]):
    claims_dir = output_dir / ".queue" / "claims"
    done_dir = output_dir / ".queue" / "done"
    claims_dir.mkdir(parents=True, exist_ok=True)
    done_dir.mkdir(parents=True, exist_ok=True)

    for item in _load_queue_items(manifest_path):
        scene_id = str(item["scene_id"])
        episode_id = int(item["episode_id"])
        key = (scene_id, episode_id)
        episode = episode_by_key.get(key)
        if episode is None:
            continue
        marker_name = _safe_marker_name(scene_id, episode_id)
        done_marker = done_dir / f"{marker_name}.done"
        if done_marker.exists():
            continue
        claim_path = claims_dir / f"{marker_name}.claim"
        try:
            fd = os.open(str(claim_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            if not _claim_path_is_active(claim_path):
                try:
                    claim_path.unlink()
                except FileNotFoundError:
                    pass
                except OSError:
                    continue
            continue
        except OSError:
            continue
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(
                {
                    "pid": os.getpid(),
                    "host": socket.gethostname(),
                    "time": time.time(),
                    "item": item,
                },
                handle,
            )
        return item, episode, claim_path, done_marker

    return None, None, None, None


def _write_done_marker(done_marker: Path, payload: dict) -> None:
    done_marker.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = done_marker.with_suffix(done_marker.suffix + f".tmp.{os.getpid()}")
    tmp_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    tmp_path.replace(done_marker)


def _finalize_ce_row_locked(output_dir: Path, log_dir: Path, progress_path: Path, eval_split: str, key: tuple[str, int], row: dict) -> tuple[list[dict], bool]:
    with _chunk_write_lock(output_dir):
        done_episodes, rows = load_done_episodes(progress_path)
        if key in done_episodes:
            return rows, False
        append_progress_row(progress_path, row)
        write_episode_log(log_dir, row["id"], row)
        rows.append(row)
        write_ce_outputs(output_dir, rows, split_name=eval_split)
        return rows, True


def evaluate_ce_queue_worker(
    env,
    agent,
    output_dir: Path,
    eval_split: str,
    start_idx: int,
    end_idx: int,
    max_episodes: int,
    split_num: int,
    split_id: int,
    max_steps_per_episode: int,
    early_stop_rotation: int,
    resume: bool,
    episode_keys_json: str,
    queue_worker_id: int,
    queue_worker_count: int,
    skip_multifloor: bool = False,
    multifloor_height_threshold: float = 0.75,
    stuck_pose_stop_steps: int = 10,
):
    output_dir.mkdir(parents=True, exist_ok=True)
    log_dir = output_dir / "log"
    progress_path = output_dir / "progress.jsonl"
    result_path = output_dir / "result.json"

    done_episodes, rows = load_done_episodes(progress_path)
    episode_filter = load_episode_key_filter(episode_keys_json)
    if resume:
        done_episodes, rows = _filter_resume_rows(output_dir, log_dir, progress_path, rows)
        for row in rows:
            row_id = row.get("id")
            if isinstance(row_id, str) and row_id:
                write_episode_log(log_dir, row_id, row)
    write_ce_outputs(output_dir, rows, split_name=eval_split)

    episodes = select_chunked_episodes(
        env.episodes,
        start_idx=start_idx,
        end_idx=end_idx,
        max_episodes=max_episodes,
        split_num=split_num,
        split_id=split_id,
    )
    if episode_filter is not None:
        episodes = [ep for ep in episodes if get_scene_episode_key(ep) in episode_filter]
    episode_by_key = {get_scene_episode_key(ep): ep for ep in episodes}
    manifest_path = _ensure_episode_queue(output_dir, episodes)

    processed = 0
    desc = f"ce-{split_id}-w{queue_worker_id}"
    pbar = tqdm(desc=desc)
    try:
        while True:
            done_episodes, _ = load_done_episodes(progress_path)
            item, episode, claim_path, done_marker = _claim_next_episode(output_dir, manifest_path, episode_by_key)
            if episode is None:
                break

            expected_scene_id = str(item["scene_id"])
            expected_episode_id = int(item["episode_id"])
            expected_key = (expected_scene_id, expected_episode_id)
            if expected_key in done_episodes:
                _write_done_marker(done_marker, {"status": "already_done", "key": list(expected_key)})
                continue

            _cleanup_incomplete_episode_output(
                output_dir=output_dir,
                log_dir=log_dir,
                scene_id=expected_scene_id,
                episode_id=expected_episode_id,
            )

            observations, runtime_episode = _reset_on_target_episode(env, episode)
            scene_id, episode_id = get_scene_episode_key(runtime_episode)
            key = (scene_id, episode_id)
            if key != expected_key:
                print(
                    f"[warn][ce] episode switched after reset: "
                    f"expected={expected_key} actual={key}"
                )
                _cleanup_incomplete_episode_output(
                    output_dir=output_dir,
                    log_dir=log_dir,
                    scene_id=scene_id,
                    episode_id=episode_id,
                )

            if skip_multifloor:
                is_multifloor, dy = _is_multifloor_episode(
                    runtime_episode, multifloor_height_threshold
                )
                if is_multifloor:
                    dy_text = "unknown" if dy is None else f"{dy:.3f}"
                    print(
                        f"[info][ce] skip multifloor episode key={key} "
                        f"dy={dy_text} threshold={float(multifloor_height_threshold):.3f}"
                    )
                    _write_done_marker(done_marker, {"status": "skipped_multifloor", "key": list(expected_key)})
                    continue

            instruction = get_episode_instruction(runtime_episode)
            agent.reset(scene_id=scene_id, episode_id=episode_id, instruction=instruction)

            done = False
            step_id = 0
            last_dtg = None
            continuous_rotation_count = 0
            last_pose = _extract_agent_pose(env)
            unchanged_pose_count = 0
            max_unchanged_pose_count = 0
            early_stop_reason = None

            while (not done) and (step_id < max_steps_per_episode):
                info = _extract_metrics(env)
                curr_dtg = extract_distance_to_goal(info)
                if last_dtg is None or curr_dtg is None or curr_dtg != last_dtg:
                    continuous_rotation_count = 0
                else:
                    continuous_rotation_count += 1
                last_dtg = curr_dtg

                if early_stop_rotation >= 0 and continuous_rotation_count > early_stop_rotation:
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
                current_pose = _extract_agent_pose(env)
                if _poses_match(current_pose, last_pose):
                    unchanged_pose_count += 1
                else:
                    unchanged_pose_count = 0
                max_unchanged_pose_count = max(
                    max_unchanged_pose_count, unchanged_pose_count
                )
                last_pose = current_pose
                if (
                    int(stuck_pose_stop_steps) > 0
                    and unchanged_pose_count >= int(stuck_pose_stop_steps)
                ):
                    early_stop_reason = f"stuck_pose_{int(stuck_pose_stop_steps)}"
                    print(
                        f"[info][ce] early stop unchanged pose "
                        f"key={key} step={step_id} count={unchanged_pose_count}"
                    )
                    done = True

            metrics = _extract_metrics(env)
            row = {
                "id": f"{scene_id}_{episode_id:04d}",
                "split": eval_split,
                "scene_id": scene_id,
                "episode_id": episode_id,
                "success": float(metrics.get("success", 0.0)),
                "spl": float(metrics.get("spl", 0.0)),
                "os": float(metrics.get("oracle_success", 0.0)),
                "ne": float(metrics.get("distance_to_goal", 0.0)),
                "path_length": float(metrics.get("path_length", 0.0)),
                "steps": int(step_id),
                "instruction": instruction,
                "queue_worker_id": int(queue_worker_id),
                "queue_worker_count": int(queue_worker_count),
                "early_stop_reason": early_stop_reason,
                "max_unchanged_pose_count": int(max_unchanged_pose_count),
            }

            rows, wrote = _finalize_ce_row_locked(output_dir, log_dir, progress_path, eval_split, key, row)
            _write_done_marker(done_marker, {"status": "done", "key": list(key), "row_id": row["id"]})
            if wrote:
                processed += 1
                pbar.update(1)
    finally:
        pbar.close()

    _, rows = load_done_episodes(progress_path)
    summary = summarize_ce(rows)
    summary["processed_episodes"] = len(rows)
    summary["split"] = eval_split
    summary["chunk"] = {"split_num": split_num, "split_id": split_id}
    summary["queue_worker"] = {"id": queue_worker_id, "count": queue_worker_count, "processed_by_this_worker": processed}
    with _chunk_write_lock(output_dir):
        write_json(result_path, summary)
    return rows, summary


def evaluate_ce(
    env,
    agent,
    output_dir: Path,
    eval_split: str,
    start_idx: int,
    end_idx: int,
    max_episodes: int,
    split_num: int,
    split_id: int,
    max_steps_per_episode: int,
    early_stop_rotation: int,
    resume: bool,
    episode_keys_json: str,
    skip_multifloor: bool = False,
    multifloor_height_threshold: float = 0.75,
    stuck_pose_stop_steps: int = 10,
):
    output_dir.mkdir(parents=True, exist_ok=True)
    log_dir = output_dir / "log"
    progress_path = output_dir / "progress.jsonl"
    result_path = output_dir / "result.json"

    done_episodes, rows = load_done_episodes(progress_path) if resume else (set(), [])
    episode_filter = load_episode_key_filter(episode_keys_json)
    if resume:
        done_episodes, rows = _filter_resume_rows(output_dir, log_dir, progress_path, rows)
        for row in rows:
            row_id = row.get("id")
            if isinstance(row_id, str) and row_id:
                write_episode_log(log_dir, row_id, row)
    write_ce_outputs(output_dir, rows, split_name=eval_split)

    episodes = select_chunked_episodes(
        env.episodes,
        start_idx=start_idx,
        end_idx=end_idx,
        max_episodes=max_episodes,
        split_num=split_num,
        split_id=split_id,
    )

    processed = 0
    for episode in tqdm(episodes, desc=f"ce-{split_id}"):
        expected_scene_id, expected_episode_id = get_scene_episode_key(episode)
        expected_key = (expected_scene_id, expected_episode_id)

        if episode_filter is not None and expected_key not in episode_filter:
            continue
        if expected_key in done_episodes:
            continue

        if resume:
            _cleanup_incomplete_episode_output(
                output_dir=output_dir,
                log_dir=log_dir,
                scene_id=expected_scene_id,
                episode_id=expected_episode_id,
            )

        observations, runtime_episode = _reset_on_target_episode(env, episode)
        scene_id, episode_id = get_scene_episode_key(runtime_episode)
        key = (scene_id, episode_id)
        if key != expected_key:
            print(
                f"[warn][ce] episode switched after reset: "
                f"expected={expected_key} actual={key}"
            )

        if episode_filter is not None and key not in episode_filter:
            continue
        if key in done_episodes:
            continue
        if resume and key != expected_key:
            _cleanup_incomplete_episode_output(
                output_dir=output_dir,
                log_dir=log_dir,
                scene_id=scene_id,
                episode_id=episode_id,
            )

        if skip_multifloor:
            is_multifloor, dy = _is_multifloor_episode(
                runtime_episode, multifloor_height_threshold
            )
            if is_multifloor:
                dy_text = "unknown" if dy is None else f"{dy:.3f}"
                print(
                    f"[info][ce] skip multifloor episode key={key} "
                    f"dy={dy_text} threshold={float(multifloor_height_threshold):.3f}"
                )
                continue

        instruction = get_episode_instruction(runtime_episode)
        agent.reset(scene_id=scene_id, episode_id=episode_id, instruction=instruction)

        done = False
        step_id = 0
        last_dtg = None
        continuous_rotation_count = 0
        last_pose = _extract_agent_pose(env)
        unchanged_pose_count = 0
        max_unchanged_pose_count = 0
        early_stop_reason = None

        while (not done) and (step_id < max_steps_per_episode):
            info = _extract_metrics(env)
            curr_dtg = extract_distance_to_goal(info)
            if last_dtg is None or curr_dtg is None or curr_dtg != last_dtg:
                continuous_rotation_count = 0
            else:
                continuous_rotation_count += 1
            last_dtg = curr_dtg

            if early_stop_rotation >= 0 and continuous_rotation_count > early_stop_rotation:
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
            current_pose = _extract_agent_pose(env)
            if _poses_match(current_pose, last_pose):
                unchanged_pose_count += 1
            else:
                unchanged_pose_count = 0
            max_unchanged_pose_count = max(max_unchanged_pose_count, unchanged_pose_count)
            last_pose = current_pose
            if (
                int(stuck_pose_stop_steps) > 0
                and unchanged_pose_count >= int(stuck_pose_stop_steps)
            ):
                early_stop_reason = f"stuck_pose_{int(stuck_pose_stop_steps)}"
                print(
                    f"[info][ce] early stop unchanged pose "
                    f"key={key} step={step_id} count={unchanged_pose_count}"
                )
                done = True

        metrics = _extract_metrics(env)
        row = {
            "id": f"{scene_id}_{episode_id:04d}",
            "split": eval_split,
            "scene_id": scene_id,
            "episode_id": episode_id,
            "success": float(metrics.get("success", 0.0)),
            "spl": float(metrics.get("spl", 0.0)),
            "os": float(metrics.get("oracle_success", 0.0)),
            "ne": float(metrics.get("distance_to_goal", 0.0)),
            "path_length": float(metrics.get("path_length", 0.0)),
            "steps": int(step_id),
            "instruction": instruction,
            "early_stop_reason": early_stop_reason,
            "max_unchanged_pose_count": int(max_unchanged_pose_count),
        }

        append_progress_row(progress_path, row)
        write_episode_log(log_dir, row["id"], row)
        rows.append(row)
        write_ce_outputs(output_dir, rows, split_name=eval_split)
        done_episodes.add(key)
        processed += 1

    summary = summarize_ce(rows)
    summary["processed_episodes"] = processed
    summary["split"] = eval_split
    summary["chunk"] = {"split_num": split_num, "split_id": split_id}
    write_json(result_path, summary)
    return rows, summary
