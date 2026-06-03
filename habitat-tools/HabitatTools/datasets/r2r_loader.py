from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import numpy as np


def get_episode_instruction(episode) -> str:
    instruction = getattr(episode, "instruction", "")
    if hasattr(instruction, "instruction_text"):
        return str(instruction.instruction_text)
    if isinstance(instruction, dict):
        return str(instruction.get("instruction_text", ""))
    return str(instruction)


def get_scene_episode_key(episode) -> tuple[str, int]:
    raw_scene = str(getattr(episode, "scene_id", ""))
    scene_id = Path(raw_scene).stem if raw_scene.endswith((".glb", ".ply")) else raw_scene
    episode_id = int(getattr(episode, "episode_id", 0))
    return scene_id, episode_id


def load_episode_key_filter(path: str) -> set[tuple[str, int]] | None:
    if not path:
        return None
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    keys: set[tuple[str, int]] = set()
    if not isinstance(payload, list):
        raise ValueError("episode key filter must be a JSON list")

    for item in payload:
        if isinstance(item, dict):
            keys.add((str(item["scene_id"]), int(item["episode_id"])))
            continue
        if isinstance(item, str) and "|" in item:
            scene_id, episode_id = item.rsplit("|", 1)
            keys.add((scene_id, int(episode_id)))
            continue
        raise ValueError(f"unsupported episode key item: {item!r}")
    return keys


def extract_distance_to_goal(info: dict | None) -> float | None:
    if not isinstance(info, dict):
        return None
    value = info.get("distance_to_goal")
    try:
        return None if value is None else float(value)
    except (TypeError, ValueError):
        return None


def select_chunked_episodes(
    episodes: Iterable,
    start_idx: int,
    end_idx: int,
    max_episodes: int,
    split_num: int,
    split_id: int,
):
    all_eps = list(episodes)
    if end_idx < 0:
        end_idx = len(all_eps)
    scoped = all_eps[start_idx:end_idx]
    if max_episodes > 0:
        scoped = scoped[:max_episodes]

    if len(scoped) == 0:
        return []

    chunks = np.array_split(scoped, split_num)
    return list(chunks[split_id])
