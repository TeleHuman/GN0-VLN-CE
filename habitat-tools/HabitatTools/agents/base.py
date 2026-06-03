from __future__ import annotations

from abc import ABC, abstractmethod


class BaseNavAgent(ABC):
    @abstractmethod
    def reset(self, scene_id: str, episode_id: int, instruction: str) -> None:
        raise NotImplementedError

    @abstractmethod
    def act(self, observations: dict, instruction: str, step_id: int) -> int:
        raise NotImplementedError
