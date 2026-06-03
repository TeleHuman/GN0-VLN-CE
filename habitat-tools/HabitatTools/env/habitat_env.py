from __future__ import annotations

import gzip
import json
from pathlib import Path
import numpy as np

from HabitatTools.config.resolver import resolve_dataset_path, resolve_scenes_dir
from HabitatTools.datasets import (
    get_scene_episode_key,
    load_episode_key_filter,
    select_chunked_episodes,
)
from HabitatTools.io.progress import load_done_episodes

# habitat-sim on some cluster images still references deprecated numpy aliases.
if not hasattr(np, "float"):
    np.float = float  # type: ignore[attr-defined]


def _is_legacy_config(config_path: Path) -> bool:
    text = config_path.read_text(encoding="utf-8")
    return "TASK_CONFIG:" in text


def _is_habitat_vln_dataset(dataset_path: Path) -> bool:
    if not dataset_path.exists() or dataset_path.suffix not in {".gz", ".json"}:
        return False
    try:
        if dataset_path.suffix == ".gz":
            with gzip.open(dataset_path, "rt", encoding="utf-8") as f:
                payload = json.load(f)
        else:
            payload = json.loads(dataset_path.read_text(encoding="utf-8"))
    except Exception:
        return False
    if not isinstance(payload, dict):
        return False
    if "instruction_vocab" not in payload:
        return False
    episodes = payload.get("episodes")
    return isinstance(episodes, list)


def _load_habitat_dataset(habitat_module, dataset_cfg):
    from habitat.datasets.registration import make_dataset

    dataset_type = getattr(dataset_cfg, "TYPE", None)
    if dataset_type is None:
        dataset_type = getattr(dataset_cfg, "type", None)
    if dataset_type is None:
        raise AttributeError("dataset config does not expose TYPE/type")
    return make_dataset(dataset_type, config=dataset_cfg)


def _select_bootstrap_episode(
    episodes,
    start_idx: int,
    end_idx: int,
    max_episodes: int,
    split_num: int,
    split_id: int,
    episode_keys_json: str,
    progress_path: str,
    resume: bool,
):
    scoped = select_chunked_episodes(
        episodes,
        start_idx=start_idx,
        end_idx=end_idx,
        max_episodes=max_episodes,
        split_num=split_num,
        split_id=split_id,
    )
    episode_filter = load_episode_key_filter(episode_keys_json)
    if episode_filter is not None:
        scoped = [ep for ep in scoped if get_scene_episode_key(ep) in episode_filter]

    if not scoped:
        return None, 0

    if resume and progress_path:
        done_episodes, _ = load_done_episodes(Path(progress_path))
        for episode in scoped:
            if get_scene_episode_key(episode) not in done_episodes:
                return episode, len(scoped)

    return scoped[0], len(scoped)


def _build_env_with_bootstrap_dataset(Env, config, full_dataset, bootstrap_episode):
    if bootstrap_episode is None:
        return Env(config=config, dataset=full_dataset)

    original_episodes = list(full_dataset.episodes)
    bootstrap_key = get_scene_episode_key(bootstrap_episode)
    first_key = get_scene_episode_key(original_episodes[0]) if original_episodes else None

    if first_key == bootstrap_key:
        env = Env(config=config, dataset=full_dataset)
    else:
        full_dataset.episodes = [bootstrap_episode]
        try:
            env = Env(config=config, dataset=full_dataset)
        finally:
            full_dataset.episodes = original_episodes

    env._dataset = full_dataset
    env.episodes = original_episodes
    env.number_of_episodes = len(original_episodes)
    if hasattr(env, "_task") and hasattr(env._task, "_dataset"):
        env._task._dataset = full_dataset
    return env


def build_habitat_env(
    habitat_config_path: str,
    eval_split: str,
    dataset_data_path: str,
    scenes_dir: str,
    repo_root: str,
    task: str,
    allow_sliding: bool | None = None,
    split_num: int = 1,
    split_id: int = 0,
    start_idx: int = 0,
    end_idx: int = -1,
    max_episodes: int = 0,
    episode_keys_json: str = "",
    progress_path: str = "",
    resume: bool = False,
):
    try:
        import habitat
        from habitat import Env
    except Exception as exc:
        raise RuntimeError(
            "Habitat import failed. Use a conda env with habitat-lab installed."
        ) from exc

    repo_root_p = Path(repo_root).resolve()
    config_path = Path(habitat_config_path)
    if not config_path.is_absolute():
        config_path = repo_root_p / config_path

    if not config_path.exists():
        raise FileNotFoundError(f"Habitat config not found: {config_path}")

    if task != "vlnce":
        raise ValueError(f"Unsupported task in submit branch: {task}")

    dataset_fallbacks = [
        "/mnt/data/InternRobotics/data/vln_ce/r2r/{split}/{split}.json.gz",
        "/mnt/data/NaVILA/evaluation/data/datasets/R2R_VLNCE_v1-3_preprocessed/{split}/{split}.json.gz",
    ]

    scene_fallbacks = [
        "/mnt/data/NaVILA/evaluation/data/scene_datasets",
        "/mnt/data/InternRobotics/data/scene_datasets",
        "data/scene_datasets",
    ]

    is_legacy = _is_legacy_config(config_path)

    if is_legacy:
        from habitat.config.default import get_config as get_legacy_habitat_config

        loaded = get_legacy_habitat_config(str(config_path))
        if hasattr(loaded, "TASK_CONFIG"):
            config = get_legacy_habitat_config()
            config.defrost()
            config.merge_from_other_cfg(loaded.TASK_CONFIG)
        else:
            config = loaded
            if hasattr(config, "defrost"):
                config.defrost()

        data_template = dataset_data_path or str(config.DATASET.DATA_PATH)
        scene_template = scenes_dir or str(config.DATASET.SCENES_DIR)

        resolved_data = resolve_dataset_path(
            data_template,
            eval_split,
            repo_root_p,
            fallback_paths=dataset_fallbacks,
            validator=_is_habitat_vln_dataset,
        )
        resolved_scenes = resolve_scenes_dir(
            scene_template,
            repo_root_p,
            fallback_dirs=scene_fallbacks,
        )

        config.DATASET.SPLIT = eval_split
        config.DATASET.DATA_PATH = str(resolved_data)
        config.DATASET.SCENES_DIR = str(resolved_scenes)
        if allow_sliding is not None:
            try:
                config.SIMULATOR.HABITAT_SIM_V0.ALLOW_SLIDING = bool(allow_sliding)
            except Exception:
                pass
        if hasattr(config, "freeze"):
            config.freeze()
        dataset = _load_habitat_dataset(habitat, config.DATASET)
        bootstrap_episode, scoped_count = _select_bootstrap_episode(
            dataset.episodes,
            start_idx=start_idx,
            end_idx=end_idx,
            max_episodes=max_episodes,
            split_num=split_num,
            split_id=split_id,
            episode_keys_json=episode_keys_json,
            progress_path=progress_path,
            resume=resume,
        )
        if bootstrap_episode is not None:
            scene_id, episode_id = get_scene_episode_key(bootstrap_episode)
            print(
                f"[info][habitat_env] bootstrap legacy task={task} split_id={split_id} "
                f"scene={scene_id} episode_id={episode_id} scoped_count={scoped_count}"
            )
        return _build_env_with_bootstrap_dataset(Env, config, dataset, bootstrap_episode)

    from habitat_baselines.config.default import get_config as get_habitat_config

    config = get_habitat_config(str(config_path))
    with habitat.config.read_write(config):
        config.habitat.dataset.split = eval_split

        data_template = dataset_data_path or str(config.habitat.dataset.data_path)
        scene_template = scenes_dir or str(config.habitat.dataset.scenes_dir)

        resolved_data = resolve_dataset_path(
            data_template,
            eval_split,
            repo_root_p,
            fallback_paths=dataset_fallbacks,
            validator=_is_habitat_vln_dataset,
        )
        resolved_scenes = resolve_scenes_dir(
            scene_template,
            repo_root_p,
            fallback_dirs=scene_fallbacks,
        )

        config.habitat.dataset.data_path = str(resolved_data)
        config.habitat.dataset.scenes_dir = str(resolved_scenes)
        if allow_sliding is not None:
            try:
                config.habitat.simulator.habitat_sim_v0.allow_sliding = bool(
                    allow_sliding
                )
            except Exception:
                pass

    dataset = _load_habitat_dataset(habitat, config.habitat.dataset)
    bootstrap_episode, scoped_count = _select_bootstrap_episode(
        dataset.episodes,
        start_idx=start_idx,
        end_idx=end_idx,
        max_episodes=max_episodes,
        split_num=split_num,
        split_id=split_id,
        episode_keys_json=episode_keys_json,
        progress_path=progress_path,
        resume=resume,
    )
    if bootstrap_episode is not None:
        scene_id, episode_id = get_scene_episode_key(bootstrap_episode)
        print(
            f"[info][habitat_env] bootstrap task={task} split_id={split_id} "
            f"scene={scene_id} episode_id={episode_id} scoped_count={scoped_count}"
        )

    return _build_env_with_bootstrap_dataset(Env, config, dataset, bootstrap_episode)
