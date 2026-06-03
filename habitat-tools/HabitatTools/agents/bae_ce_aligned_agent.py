from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any

from HabitatTools.agents.base import BaseNavAgent
from HabitatTools.agents.habitat_dagger_sim_adapter import HabitatDaggerSimAdapter


def _extract_goal_position(episode: Any) -> list[float] | None:
    goals = getattr(episode, "goals", None)
    if isinstance(goals, list) and goals:
        goal0 = goals[0]
        if isinstance(goal0, dict):
            pos = goal0.get("position")
            if pos is not None:
                return [float(pos[0]), float(pos[2]), 0.0]
        pos = getattr(goal0, "position", None)
        if pos is not None:
            return [float(pos[0]), float(pos[2]), 0.0]
    return None


class _BaseBAECEAlignedAgent(BaseNavAgent):
    def __init__(
        self,
        env,
        model_path: str,
        output_dir: str,
        occupancy_root: str,
        prompt_type: str = "V3HF",
        action_num: int = 1,
        load_dtype: str = "bf16",
        gt_strict_coverage: bool = False,
        astar_margins: tuple[int, ...] = (9, 8, 7, 6, 5, 4),
        current_resize_w: int = 480,
        current_resize_h: int = 360,
        history_grid_size: int = 4,
        history_tile_w: int = 160,
        history_tile_h: int = 120,
        agent_module: str = "bae_agent_dagger",
    ):
        # Reuse the BAE implementation while swapping in the Habitat-backed sim adapter.
        AgentClass = importlib.import_module(agent_module).BAEAgent

        self._env = env
        self._output_dir = Path(output_dir)
        self._output_dir.mkdir(parents=True, exist_ok=True)

        self._sim_adapter = HabitatDaggerSimAdapter(
            env=env,
            occupancy_root=occupancy_root,
            margins=tuple(int(m) for m in astar_margins),
            skip_supervision_when_in_obstacle=False,
            gt_strict_coverage=bool(gt_strict_coverage),
            strict_gt_dataset_goal_only=True,
        )
        self._agent = AgentClass(
            model_path=model_path,
            result_path=str(self._output_dir),
            prompt_type=str(prompt_type),
            action_num=int(action_num),
            dtype=str(load_dtype),
            current_resize_w=int(current_resize_w),
            current_resize_h=int(current_resize_h),
            history_grid_size=int(history_grid_size),
            history_tile_w=int(history_tile_w),
            history_tile_h=int(history_tile_h),
        )

    def reset(self, scene_id: str, episode_id: int, instruction: str) -> None:
        _ = instruction
        episode = getattr(self._env, "current_episode", None)
        if episode is None:
            raise RuntimeError("env.current_episode is None during agent.reset")

        self._sim_adapter.reset_episode(scene_id=str(scene_id), episode=episode)

        episode_ref = Path(f"/tmp/{scene_id}/{int(episode_id):04d}.json")
        self._agent.reset(episode_ref=episode_ref, sim=self._sim_adapter)

    def act(self, observations: dict, instruction: str, step_id: int) -> int:
        _ = step_id
        episode = getattr(self._env, "current_episode", None)
        goal_position = _extract_goal_position(episode) if episode is not None else None

        agent_obs = {
            "sim": self._sim_adapter,
            "goal_position": goal_position,
            "instruction": {"text": str(instruction)},
            "rgb": observations["rgb"],
        }
        info = observations.get("metrics", {}) if isinstance(observations, dict) else {}
        out = self._agent.act(agent_obs, info)

        if isinstance(out, dict) and "action" in out:
            return int(out["action"])
        return int(out)


class BAECEDaggerAlignedAgent(_BaseBAECEAlignedAgent):
    pass


class BAECEEvalAlignedAgent(_BaseBAECEAlignedAgent):
    def __init__(self, *args, **kwargs):
        kwargs["gt_strict_coverage"] = False
        kwargs["agent_module"] = "bae_agent_eval"
        super().__init__(*args, **kwargs)
