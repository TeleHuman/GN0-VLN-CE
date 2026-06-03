from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from HabitatTools.agents.base import BaseNavAgent
from HabitatTools.utils.actions import ACTION_STOP, choose_first_action
from HabitatTools.utils.images import build_history_mosaic, ensure_uint8_rgb


class BAEV3Agent(BaseNavAgent):
    """BAE adapter for Habitat CE with V3HF visual input.

    Input contract is fixed to two images:
    1) 4x4 history mosaic (previous frames only)
    2) current RGB
    """

    def __init__(
        self,
        model_path: str,
        output_dir: str,
        load_dtype: str = "bf16",
        max_new_tokens: int = 512,
        action_num: int = 1,
        history_len: int = 16,
        fallback_action: int = ACTION_STOP,
        ignore_stop_on_step0: bool = True,
        step0_override_action: int = 1,
    ):
        try:
            from bae import BAEInference
        except Exception as exc:
            raise RuntimeError(
                "Cannot import BAEInference. Ensure PYTHONPATH includes /mnt/data/GN0-VLN-CE"
            ) from exc

        self.inference = BAEInference(
            model_path=model_path,
            prompt_type="V3HF",
            dtype=load_dtype,
            device_map="auto",
            trust_remote_code=True,
            max_new_tokens=max_new_tokens,
        )

        self.output_dir = Path(output_dir)
        self.runtime_image_dir = self.output_dir / "runtime_images"
        self.runtime_image_dir.mkdir(parents=True, exist_ok=True)

        self.action_num = max(1, int(action_num))
        self.history_len = max(1, min(32, int(history_len)))
        self.fallback_action = int(fallback_action)
        self.ignore_stop_on_step0 = bool(ignore_stop_on_step0)
        self.step0_override_action = int(step0_override_action)

        self.scene_id = ""
        self.episode_id = 0
        self.episode_key = ""
        self.rgb_list: list[np.ndarray] = []

    def reset(self, scene_id: str, episode_id: int, instruction: str) -> None:
        self.scene_id = scene_id
        self.episode_id = int(episode_id)
        self.episode_key = f"{scene_id}_{episode_id:04d}"
        self.rgb_list = []

    def _save_image(self, image: np.ndarray, step_id: int, tag: str) -> str:
        img = ensure_uint8_rgb(image)
        image_path = self.runtime_image_dir / f"{self.episode_key}_step{step_id:04d}_{tag}.png"
        cv2.imwrite(str(image_path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        return str(image_path)

    def _build_model_inputs(self, rgb: np.ndarray, step_id: int) -> list[str]:
        history_frames = self.rgb_list[-self.history_len :]
        history_mosaic = build_history_mosaic(history_frames, grid_size=4, tile_w=160, tile_h=120)
        rgb_path = self._save_image(rgb, step_id, "rgb")
        hist_path = self._save_image(history_mosaic, step_id, "hist")
        return [hist_path, rgb_path]

    def act(self, observations: dict, instruction: str, step_id: int) -> int:
        rgb = ensure_uint8_rgb(observations["rgb"])
        image_paths = self._build_model_inputs(rgb, step_id)

        action_seq = None
        try:
            actions, _, _, _ = self.inference.predict(
                image_paths=image_paths,
                instruction=instruction,
                cur_x=None,
                cur_y=None,
                occ_w=None,
                occ_h=None,
                occ_meter_per_px=0.05,
                occ_rot_deg=0,
                prev_actions=None,
            )
            if actions:
                action_seq = list(actions[: self.action_num])
        except Exception as exc:
            print(f"[Warn] BAE inference failed at step {step_id}: {exc}")

        action = choose_first_action(action_seq, fallback=self.fallback_action)
        if self.ignore_stop_on_step0 and int(step_id) == 0 and int(action) == ACTION_STOP:
            action = int(self.step0_override_action)

        self.rgb_list.append(rgb.copy())
        if len(self.rgb_list) > self.history_len:
            self.rgb_list = self.rgb_list[-self.history_len :]

        return int(action)
