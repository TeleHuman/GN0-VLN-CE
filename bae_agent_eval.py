"""Dedicated CE eval agent.

This is the formal eval runtime used by the submit branch. It keeps the
evaluation-time action/runtime behavior while staying lightweight:
- prompt/debug images are written only to a temporary per-episode cache
- eval results still dump `meta.json` and `sample.json` for inspection
"""

import json
import math
import re
import shutil
import tempfile
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R

from bae import BAEInference
from bae.constants import (
    ACTION_MOVE_FORWARD,
    ACTION_STOP,
    ACTION_TURN_LEFT,
    ACTION_TURN_RIGHT,
    NUM_ACTIONS,
    PIXEL_TO_METER,
)


class BAEAgent:
    def __init__(
        self,
        model_path,
        result_path,
        prompt_type,
        action_num=1,
        dtype="bf16",
        current_resize_w=480,
        current_resize_h=360,
        history_grid_size=4,
        history_tile_w=160,
        history_tile_h=120,
    ):
        print("Initialize BAE")

        self.result_path = result_path
        self.prompt_type = str(prompt_type).upper()
        if self.prompt_type != "V3HF":
            raise ValueError(
                f"Unsupported prompt type: {prompt_type!r}. Only 'V3HF' is supported."
            )
        self.action_num = max(1, int(action_num))
        self.current_resize_w = int(current_resize_w)
        self.current_resize_h = int(current_resize_h)
        self.history_grid_size = int(history_grid_size)
        self.history_tile_w = int(history_tile_w)
        self.history_tile_h = int(history_tile_h)
        self.history_target_size = (
            self.history_grid_size * self.history_tile_w,
            self.history_grid_size * self.history_tile_h,
        )

        self.inference = BAEInference(
            model_path=model_path,
            prompt_type=prompt_type,
            dtype=dtype,
            device_map="auto",
            trust_remote_code=True,
            max_new_tokens=512,
        )

        self.rgb_list = []
        self.episode_id = None
        self.terminated_by_invalid = False
        self.llm_call_count = 0
        self.image_path = None
        self.sample_path = None
        self.meta_path = None
        self.history_action_list = None
        self.position_dump_counts = None
        self._last_action_trace = None
        self._last_progress_pixel = None
        self._stuck_frame_count = 0
        self._non_forward_exec_count = 0
        self._prompt_image_root = None

        print("BAE Initialization Complete")

    def reset(self, episode_ref, sim=None):
        _ = sim
        self._cleanup_prompt_image_root()
        self.rgb_list = []
        self.terminated_by_invalid = False
        self.llm_call_count = 0
        self.history_action_list = []
        self.position_dump_counts = {}
        self._last_action_trace = None
        self._last_progress_pixel = None
        self._stuck_frame_count = 0
        self._non_forward_exec_count = 0
        self.episode_id = f"{episode_ref.parent.name}_{episode_ref.stem}"
        episode_root = Path(self.result_path) / self.episode_id
        episode_root.mkdir(parents=True, exist_ok=True)
        self.sample_path = episode_root / "sample.json"
        self.meta_path = episode_root / "meta.json"

        meta_payload = dict(getattr(sim, "meta", {}) or {})
        meta_payload["prompt_image_persisted"] = False
        with open(self.meta_path, "w", encoding="utf-8") as f:
            json.dump(meta_payload, f, ensure_ascii=False, indent=2)

        self._prompt_image_root = Path(
            tempfile.mkdtemp(prefix=f"{self.episode_id}_", dir="/tmp")
        )
        self.image_path = self._prompt_image_root
        print("BAE Reset Complete for Episode:", self.episode_id)

    def act(self, observations, info):
        sim = observations.get("sim")
        goal_position = observations.get("goal_position")
        instruction = observations.get("instruction")["text"]
        curr_dtg = info.get("distance_to_goal")

        rgb = observations.get("rgb")
        occ_map = sim.get_occ_map()
        occ_h, occ_w = occ_map.shape[:2]

        if self.terminated_by_invalid:
            return self._return_action(ACTION_STOP)

        self._last_action_trace = None
        self.rgb_list.append(rgb)
        self._prepare_prompt_image_cache()

        model_rgb = self._resize_current_rgb(rgb)
        paths = {
            "rgb": self._save_prompt_image(model_rgb, "rgb"),
            "occ": self._save_prompt_image(occ_map, "occ"),
            "bev": sim.bev_path,
        }

        hist = self.build_history_mosaic(
            self.rgb_list[:-1],
            self.history_action_list,
            grid_size=self.history_grid_size,
            target_size=self.history_target_size,
        )
        paths["hist"] = self._save_prompt_image(hist, "hist")

        cur_x, cur_y = -1, -1
        pixel_pos = sim.get_current_pixel_position()
        if pixel_pos is not None and len(pixel_pos) >= 2:
            cur_x, cur_y = map(int, pixel_pos)
        self._update_stuck_progress((cur_x, cur_y))
        if self._should_abort_due_to_stuck():
            print(
                "Stuck abort triggered: "
                f"stuck_frames={self._stuck_frame_count}, "
                "mark episode invalid and stop."
            )
            self.terminated_by_invalid = True
            return self._return_action(ACTION_STOP)

        image_paths = [paths["hist"], paths["rgb"]]

        prev_actions_xml = self._build_prev_actions_xml(self.history_action_list)

        try:
            habitat_state = (
                sim.get_habitat_agent_state()
                if hasattr(sim, "get_habitat_agent_state")
                else sim.get_agent_state()
            )
            curr_pos = np.array(habitat_state.position, dtype=np.float32)
            curr_rot = R.from_quat(habitat_state.rotation)
            world_fwd = curr_rot.apply(np.array([1.0, 0.0, 0.0], dtype=np.float32))
            curr_yaw_deg = float(math.degrees(math.atan2(world_fwd[2], world_fwd[0])))
            goal_position_xyz = None
            get_goal_position_3d = getattr(sim, "get_goal_position_3d", None)
            if callable(get_goal_position_3d):
                try:
                    goal_xyz = get_goal_position_3d()
                    if goal_xyz is not None and len(goal_xyz) >= 3:
                        goal_position_xyz = [
                            float(goal_xyz[0]),
                            float(goal_xyz[1]),
                            float(goal_xyz[2]),
                        ]
                except Exception:
                    goal_position_xyz = None
            success_distance = self._get_goal_stop_distance(sim)
            within_success_distance = False
            try:
                within_success_distance = (
                    curr_dtg is not None and float(curr_dtg) < float(success_distance)
                )
            except Exception:
                within_success_distance = False
            agent_pose = [
                float(curr_pos[0]),
                float(curr_pos[1]),
                float(curr_pos[2]),
                curr_yaw_deg,
            ]

            actions, pixels, raw_text, prompt_text, token_probs = self.inference.predict(
                image_paths=image_paths,
                instruction=instruction,
                cur_x=cur_x,
                cur_y=cur_y,
                occ_w=occ_w,
                occ_h=occ_h,
                occ_meter_per_px=PIXEL_TO_METER,
                occ_rot_deg=0,
                prev_actions=prev_actions_xml,
                return_token_probs=True,
            )

            valid_reasons = []
            act_valid, act_reason = self._rollout_actions_valid(
                sim, actions, viz_prefix="action"
            )
            action_hit_step_idx = self._parse_rollout_hit_step(act_reason)
            if not act_valid:
                valid_reasons.append(act_reason)

            exit_by_action, exit_action_reason, _ = self._goal_zone_analysis_on_actions(
                sim, actions, goal_position
            )
            if exit_by_action:
                valid_reasons.append(exit_action_reason)

            pix_reason = "disabled_for_v3hf"

            exit_by_pixel, exit_pixel_reason, _ = self._goal_zone_analysis_on_pixels(
                sim, pixels, goal_position
            )
            if exit_by_pixel:
                valid_reasons.append(exit_pixel_reason)

            stop_in_actions = (
                actions is not None
                and len(actions) == NUM_ACTIONS
                and int(actions[0]) == ACTION_STOP
            )
            stop_outside_goal = bool(
                stop_in_actions and self._distance_outside_goal(curr_dtg, sim)
            )
            if stop_outside_goal:
                valid_reasons.append("stop_outside_goal")

            self.llm_call_count += 1

            action_hard_fail = (not act_valid) and (not self._is_action_wall_reason(act_reason))
            if actions and len(actions) == NUM_ACTIONS and not action_hard_fail:
                exec_action = int(actions[0])

                if stop_in_actions:
                    if stop_outside_goal:
                        print("Model STOP outside goal detected. Keep STOP without override.")
                    return self._return_action(ACTION_STOP)

                if action_hit_step_idx is not None:
                    print(
                        "Warning: Rollout wall hit at step "
                        f"{action_hit_step_idx}, continuing."
                    )
                if any(self._is_pixel_wall_reason(r) for r in valid_reasons):
                    print("Warning: Pixel path hits wall, continuing.")

                self._log_eval_step(
                    instruction=instruction,
                    prompt_text=prompt_text,
                    image_paths=image_paths,
                    occ_image_path=paths["occ"],
                    cur_x=cur_x,
                    cur_y=cur_y,
                    occ_h=occ_h,
                    occ_w=occ_w,
                    goal_position=goal_position,
                    goal_position_xyz=goal_position_xyz,
                    success_distance=success_distance,
                    within_success_distance=within_success_distance,
                    agent_pose=agent_pose,
                    raw_text=raw_text,
                    actions=actions,
                    pixels=pixels,
                    token_probs=token_probs,
                    action_trace=self._last_action_trace,
                    valid=(act_valid and not exit_by_action and not exit_by_pixel and not stop_outside_goal),
                    act_valid=act_valid,
                    act_reason=act_reason,
                    action_hit_step_idx=action_hit_step_idx,
                    stop_in_actions=stop_in_actions,
                    curr_dtg=curr_dtg,
                    exit_by_action=exit_by_action,
                    exit_action_reason=exit_action_reason,
                    pix_valid=None,
                    pix_reason=pix_reason,
                    pixel_hit_step_idx=None,
                    exit_by_pixel=exit_by_pixel,
                    exit_pixel_reason=exit_pixel_reason,
                    valid_reasons=valid_reasons,
                )

                print(f"BAE predicted actions: {actions}")
                if pixels:
                    print(f"BAE predicted pixels: {len(pixels)} waypoints")
                print(f"Raw output: {raw_text}")
                return self._return_action(exec_action)

            print(
                f"Invalid output encountered. Reasons: {valid_reasons}. "
                "Fallback to default <FWD>."
            )
            self._log_eval_step(
                instruction=instruction,
                prompt_text=prompt_text,
                image_paths=image_paths,
                occ_image_path=paths["occ"],
                cur_x=cur_x,
                cur_y=cur_y,
                occ_h=occ_h,
                occ_w=occ_w,
                goal_position=goal_position,
                goal_position_xyz=goal_position_xyz,
                success_distance=success_distance,
                within_success_distance=within_success_distance,
                agent_pose=agent_pose,
                raw_text=raw_text,
                actions=actions,
                pixels=pixels,
                token_probs=token_probs,
                action_trace=self._last_action_trace,
                valid=False,
                act_valid=act_valid,
                act_reason=act_reason,
                action_hit_step_idx=action_hit_step_idx,
                stop_in_actions=stop_in_actions,
                curr_dtg=curr_dtg,
                exit_by_action=exit_by_action,
                exit_action_reason=exit_action_reason,
                pix_valid=None,
                pix_reason=pix_reason,
                pixel_hit_step_idx=None,
                exit_by_pixel=exit_by_pixel,
                exit_pixel_reason=exit_pixel_reason,
                valid_reasons=valid_reasons,
            )
            if not bool(getattr(sim.config, "COLLIDABLE", True)) and actions:
                exec_action = int(actions[0])
                print(f"Raw output: {raw_text}")
                return self._return_action(exec_action)
            return self._return_action(ACTION_MOVE_FORWARD)

        except Exception as e:
            print(f"Error during inference: {e}")
            import traceback

            traceback.print_exc()
            self.llm_call_count += 1
            self._log_eval_step(
                instruction=instruction,
                prompt_text=None,
                image_paths=image_paths,
                occ_image_path=paths["occ"],
                cur_x=cur_x,
                cur_y=cur_y,
                occ_h=occ_h,
                occ_w=occ_w,
                goal_position=goal_position,
                goal_position_xyz=None,
                success_distance=self._get_goal_stop_distance(sim),
                within_success_distance=False,
                agent_pose=None,
                raw_text=str(e),
                actions=None,
                pixels=None,
                token_probs=None,
                action_trace=None,
                valid=False,
                act_valid=False,
                act_reason="inference_exception",
                action_hit_step_idx=None,
                stop_in_actions=False,
                curr_dtg=curr_dtg,
                exit_by_action=False,
                exit_action_reason="skip",
                pix_valid=None,
                pix_reason="disabled_for_v3hf",
                pixel_hit_step_idx=None,
                exit_by_pixel=False,
                exit_pixel_reason="skip",
                valid_reasons=["inference_exception"],
            )
            print("Inference exception fallback to default <FWD>.")
            return self._return_action(ACTION_MOVE_FORWARD)

    def _return_action(self, action: int):
        out_action = int(action)
        if out_action == ACTION_MOVE_FORWARD:
            self._non_forward_exec_count = 0
        else:
            self._non_forward_exec_count += 1
            if self._non_forward_exec_count >= 10:
                print(
                    "Force one FWD after 10 consecutive non-FWD executed actions. "
                    f"override {out_action} -> {ACTION_MOVE_FORWARD}"
                )
                out_action = ACTION_MOVE_FORWARD
                self._non_forward_exec_count = 0

        self.history_action_list.append(out_action)
        return {"action": out_action}

    def _update_stuck_progress(self, pixel_pos):
        if pixel_pos is None:
            self._last_progress_pixel = None
            self._stuck_frame_count = 0
            return
        try:
            px = (int(pixel_pos[0]), int(pixel_pos[1]))
        except Exception:
            self._last_progress_pixel = None
            self._stuck_frame_count = 0
            return
        if px[0] < 0 or px[1] < 0:
            self._last_progress_pixel = None
            self._stuck_frame_count = 0
            return
        if self._last_progress_pixel == px:
            self._stuck_frame_count += 1
        else:
            self._last_progress_pixel = px
            self._stuck_frame_count = 0

    def _cleanup_prompt_image_root(self):
        if self._prompt_image_root is None:
            return
        try:
            shutil.rmtree(self._prompt_image_root, ignore_errors=True)
        except Exception:
            pass
        self._prompt_image_root = None

    def _prepare_prompt_image_cache(self):
        if self.image_path is None:
            return
        try:
            shutil.rmtree(self.image_path, ignore_errors=True)
        except Exception:
            pass
        Path(self.image_path).mkdir(parents=True, exist_ok=True)

    def _save_prompt_image(self, img: np.ndarray, prefix: str) -> str:
        save_dir = Path(self.image_path) / prefix
        save_dir.mkdir(parents=True, exist_ok=True)
        filepath = save_dir / f"{len(self.rgb_list) - 1}.png"
        cv2.imwrite(str(filepath), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        return str(filepath)

    @staticmethod
    def _get_goal_stop_distance(_sim=None) -> float:
        return 1.0

    def _distance_outside_goal(self, curr_dtg, sim=None) -> bool:
        try:
            return float(curr_dtg) > self._get_goal_stop_distance(sim)
        except Exception:
            return False

    def _should_abort_due_to_stuck(self) -> bool:
        return False

    def build_history_mosaic(
        self,
        rgb_list,
        history_action_list,
        grid_size=4,
        target_size=(640, 480),
        only_moved=False,
    ):
        tw, th = target_size
        num_tiles = grid_size**2
        tile_w, tile_h = tw // grid_size, th // grid_size

        if not only_moved:
            selected_frames = rgb_list.copy()
        else:
            selected_frames = []
            for idx, frame in enumerate(rgb_list):
                if idx == 0:
                    selected_frames.append(frame)
                    continue
                prev_action_idx = idx - 1
                if prev_action_idx >= len(history_action_list):
                    continue
                if int(history_action_list[prev_action_idx]) == ACTION_MOVE_FORWARD:
                    selected_frames.append(frame)

        recent_frames = selected_frames[-num_tiles:][::-1]
        padding = [np.zeros((tile_h, tile_w, 3), dtype=np.uint8)] * (
            num_tiles - len(recent_frames)
        )
        tiles = [cv2.resize(f, (tile_w, tile_h)) for f in recent_frames] + padding
        grid = np.array(tiles).reshape(grid_size, grid_size, tile_h, tile_w, 3)
        return grid.swapaxes(1, 2).reshape(th, tw, 3)

    def _resize_current_rgb(self, rgb: np.ndarray) -> np.ndarray:
        if rgb is None:
            return rgb
        h, w = rgb.shape[:2]
        target = (self.current_resize_w, self.current_resize_h)
        if (w, h) == target:
            return rgb
        return cv2.resize(rgb, target, interpolation=cv2.INTER_AREA)

    def save_image(self, img: np.ndarray, prefix: str) -> str:
        return self._save_prompt_image(img, prefix)

    def _build_prev_actions_xml(self, history_actions: list[int]) -> str:
        id2tok = {
            ACTION_STOP: "<STOP>",
            ACTION_MOVE_FORWARD: "<FWD>",
            ACTION_TURN_LEFT: "<LEFT>",
            ACTION_TURN_RIGHT: "<RIGHT>",
        }
        recent = list(reversed(history_actions))[:5]
        toks = [id2tok.get(int(a), "<None>") for a in recent]
        while len(toks) < 5:
            toks.append("<None>")
        return "<action>" + ",".join(toks) + "</action>"

    def _rollout_actions_valid(self, sim, actions, viz_prefix="action"):
        self._last_action_trace = None

        def _maybe_save(img):
            if viz_prefix:
                self.save_image(img, str(viz_prefix))

        if not actions or len(actions) != NUM_ACTIONS:
            action_img = sim.get_occ_map_with_actions(actions, traj_color=(255, 165, 0))
            _maybe_save(action_img)
            return False, "invalid_actions_len"

        habitat_trace_fn = getattr(sim, "rollout_actions_habitat_trace", None)
        if callable(habitat_trace_fn):
            try:
                ok, reason, trace = habitat_trace_fn(actions)
                self._last_action_trace = trace if isinstance(trace, list) else []
                try:
                    action_img = sim.get_occ_map_with_actions(
                        actions,
                        traj_color=(255, 165, 0),
                        trace=self._last_action_trace,
                    )
                except TypeError:
                    action_img = sim.get_occ_map_with_actions(actions, traj_color=(255, 165, 0))
                _maybe_save(action_img)
                return bool(ok), str(reason)
            except Exception:
                self._last_action_trace = []

        habitat_rollout_fn = getattr(sim, "rollout_actions_valid_habitat", None)
        if callable(habitat_rollout_fn):
            try:
                action_img = sim.get_occ_map_with_actions(actions, traj_color=(255, 165, 0))
                _maybe_save(action_img)
                ok, reason = habitat_rollout_fn(actions)
                return bool(ok), str(reason)
            except Exception as e:
                return False, f"habitat_rollout_error:{e}"

        action_img = sim.get_occ_map_with_actions(actions, traj_color=(255, 165, 0))
        _maybe_save(action_img)
        grid = self._get_collision_grid(sim)
        if grid is None:
            return False, "missing_passable_grid"

        step_size = float(sim.config.FORWARD_STEP_SIZE)
        turn_rad = np.deg2rad(float(sim.config.TURN_ANGLE))
        map_size = (int(sim.map_width), int(sim.map_height))
        rot_left = R.from_euler("z", turn_rad, degrees=False)
        rot_right = R.from_euler("z", -turn_rad, degrees=False)
        local_fwd_vec = np.array([1.0, 0.0, 0.0], dtype=np.float32)

        agent_state = sim.get_agent_state()
        curr_pos = np.array(agent_state.position, dtype=np.float32)
        curr_rot = R.from_quat(agent_state.rotation)

        for step_idx, action in enumerate(map(int, actions)):
            if action == ACTION_STOP:
                continue
            if action == ACTION_TURN_LEFT:
                curr_rot = curr_rot * rot_left
                continue
            if action == ACTION_TURN_RIGHT:
                curr_rot = curr_rot * rot_right
                continue
            if action != ACTION_MOVE_FORWARD:
                return False, f"unknown_action_{action}"

            world_fwd = curr_rot.apply(local_fwd_vec)
            next_pos = curr_pos + (world_fwd * step_size)
            px_start = sim.transform_from_world_to_pixel(curr_pos)
            px_end = sim.transform_from_world_to_pixel(next_pos)
            if not self._segment_is_clear(grid, *map_size, px_start, px_end, sim):
                return False, f"action_rollout_hit_wall_step_{step_idx}"
            curr_pos = next_pos

        return True, "ok"

    def _get_collision_grid(self, sim):
        margins = getattr(sim, "margins", [])
        safe_grids = getattr(sim, "safe_passable_grids", {})
        grid = safe_grids.get(margins[-1]) if margins and safe_grids else None
        if grid is None:
            grid = getattr(sim, "passable_grid", None)
        return grid

    def _segment_is_clear(self, grid, map_w, map_h, start_px, end_px, sim):
        for x, y in sim._bresenham_line(start_px, end_px):
            if not (0 <= x < map_w and 0 <= y < map_h):
                return False
            if not grid[y][x]:
                return False
        return True

    @staticmethod
    def _parse_rollout_hit_step(reason):
        m = re.match(r"^(?:action|habitat)_rollout_hit_wall_step_(\d+)$", str(reason))
        return int(m.group(1)) if m else None

    @staticmethod
    def _parse_pixel_hit_step(reason):
        m = re.match(r"^pixel_branch_hit_wall_seg_(\d+)$", str(reason))
        return int(m.group(1)) if m else None

    def _goal_zone_analysis_on_actions(self, sim, actions, goal_position):
        if goal_position is None or actions is None or len(actions) != NUM_ACTIONS:
            return False, "skip", {"first_enter_idx": None}

        goal_radius = max(min(3, sim.path_length * 0.2), 1)
        step_size = float(sim.config.FORWARD_STEP_SIZE)
        turn_rad = np.deg2rad(float(sim.config.TURN_ANGLE))
        rot_left = R.from_euler("z", turn_rad, degrees=False)
        rot_right = R.from_euler("z", -turn_rad, degrees=False)
        local_fwd_vec = np.array([1.0, 0.0, 0.0], dtype=np.float32)

        agent_state = sim.get_agent_state()
        curr_pos = np.array(agent_state.position, dtype=np.float32)
        curr_rot = R.from_quat(agent_state.rotation)
        goal_xy = np.array(goal_position[:2], dtype=np.float32)

        curr_dist = float(np.linalg.norm(curr_pos[:2] - goal_xy))
        is_inside = curr_dist <= goal_radius
        first_enter_idx = -1 if is_inside else None

        for step_idx, action in enumerate(map(int, actions)):
            if action == ACTION_TURN_LEFT:
                curr_rot = curr_rot * rot_left
            elif action == ACTION_TURN_RIGHT:
                curr_rot = curr_rot * rot_right
            elif action == ACTION_MOVE_FORWARD:
                curr_pos = curr_pos + (curr_rot.apply(local_fwd_vec) * step_size)

            curr_dist = float(np.linalg.norm(curr_pos[:2] - goal_xy))
            is_now_inside = curr_dist <= goal_radius
            if not is_inside and is_now_inside:
                is_inside = True
                first_enter_idx = step_idx
            elif is_inside and not is_now_inside:
                return True, f"goal_zone_exit_by_action_step_{step_idx}", {
                    "first_enter_idx": first_enter_idx
                }

        return False, "ok", {"first_enter_idx": first_enter_idx}

    def _pixel_path_valid(self, sim, pixel_waypoints_norm):
        if not pixel_waypoints_norm:
            return False, "missing_pixel_waypoints"

        grid = self._get_collision_grid(sim)
        if grid is None:
            return False, "missing_passable_grid"
        map_h, map_w = grid.shape[:2] if hasattr(grid, "shape") else (len(grid), len(grid[0]))
        if map_w <= 0 or map_h <= 0:
            return False, "bad_occ_shape"

        waypoints_px = []
        for pt in pixel_waypoints_norm:
            if not (isinstance(pt, (list, tuple)) and len(pt) == 2):
                return False, "bad_pixel_waypoint_format"
            waypoints_px.append(self._denorm_to_occ_pixel(int(pt[0]), int(pt[1]), map_w, map_h))

        if waypoints_px:
            pixel_img = sim.get_occ_map_with_pixels(waypoints_px, path_color=(255, 0, 255))
            self.save_image(pixel_img, "pixel")

        for idx, (p_start, p_end) in enumerate(zip(waypoints_px, waypoints_px[1:])):
            if not self._segment_is_clear(grid, map_w, map_h, p_start, p_end, sim):
                return False, f"pixel_branch_hit_wall_seg_{idx}"
        return True, "ok"

    def _denorm_to_occ_pixel(self, x_norm, y_norm, occ_w, occ_h):
        px = int(round((int(x_norm) / 1000.0) * float(occ_w)))
        py = int(round((int(y_norm) / 1000.0) * float(occ_h)))
        return max(0, min(px, occ_w - 1)), max(0, min(py, occ_h - 1))

    def _goal_zone_analysis_on_pixels(self, sim, pixel_waypoints_norm, goal_position):
        if goal_position is None or not pixel_waypoints_norm:
            return False, "skip", {"first_enter_idx": None}

        goal_radius = max(min(3, sim.path_length * 0.2), 1)
        map_w, map_h = int(sim.map_width), int(sim.map_height)
        goal_xy = np.array(goal_position[:2], dtype=np.float32)

        is_inside = False
        first_enter_idx = None
        for idx, pt_norm in enumerate(pixel_waypoints_norm):
            if not (isinstance(pt_norm, (list, tuple)) and len(pt_norm) == 2):
                return True, "goal_zone_exit_bad_pixel_format", {
                    "first_enter_idx": first_enter_idx
                }

            px_pt = self._denorm_to_occ_pixel(int(pt_norm[0]), int(pt_norm[1]), map_w, map_h)
            world_pt_tuple = sim.transform_from_pixel_to_world(px_pt)
            world_pos = np.array(world_pt_tuple[:2], dtype=np.float32)
            dist = float(np.linalg.norm(world_pos - goal_xy))
            is_now_inside = dist <= goal_radius

            if not is_inside and is_now_inside:
                is_inside = True
                first_enter_idx = idx
            elif is_inside and not is_now_inside:
                return True, f"goal_zone_exit_by_pixel_step_{idx}", {
                    "first_enter_idx": first_enter_idx
                }

        return False, "ok", {"first_enter_idx": first_enter_idx}

    @staticmethod
    def _is_action_wall_reason(reason):
        return isinstance(reason, str) and (
            reason.startswith("action_rollout_hit_wall_step_")
            or reason.startswith("habitat_rollout_hit_wall_step_")
        )

    @staticmethod
    def _is_pixel_wall_reason(reason):
        return isinstance(reason, str) and reason.startswith("pixel_branch_hit_wall_seg_")

    def _log_eval_step(
        self,
        instruction,
        prompt_text,
        image_paths,
        occ_image_path,
        cur_x,
        cur_y,
        occ_h,
        occ_w,
        goal_position,
        goal_position_xyz,
        success_distance,
        within_success_distance,
        agent_pose,
        raw_text,
        actions,
        pixels,
        token_probs,
        action_trace,
        valid,
        act_valid,
        act_reason,
        action_hit_step_idx,
        stop_in_actions,
        curr_dtg,
        exit_by_action,
        exit_action_reason,
        pix_valid,
        pix_reason,
        pixel_hit_step_idx,
        exit_by_pixel,
        exit_pixel_reason,
        valid_reasons,
    ):
        payload = {
            "episode_id": self.episode_id,
            "llm_call_idx": self.llm_call_count,
            "step_idx": len(self.rgb_list) - 1,
            "prompt_type": self.prompt_type,
            "in_obstacle": False,
            "input": {
                "instruction": instruction,
                "prompt_text": prompt_text,
                "image_paths": image_paths,
                "occ_image_path": occ_image_path,
                "prompt_image_persisted": False,
                "cur_pixel": [cur_x, cur_y],
                "occ_shape": [occ_h, occ_w],
                "goal_position": (
                    goal_position_xyz if goal_position_xyz is not None else goal_position
                ),
                "goal_position_format": (
                    "xyz" if goal_position_xyz is not None else "xz0_legacy"
                ),
                "goal_position_legacy_xz0": goal_position,
                "agent_pose": agent_pose,
                "agent_pose_format": "xyz_heading_deg",
            },
            "output": {
                "raw_text": raw_text,
                "actions": actions,
                "pixels": pixels,
                "token_probs": token_probs,
                "action_trace": action_trace,
            },
            "validation": {
                "is_true": bool(valid),
                "action_rollout_valid": bool(act_valid),
                "action_reason": act_reason,
                "action_hit_step_idx": action_hit_step_idx,
                "stop_in_actions": bool(stop_in_actions),
                "distance_to_goal": curr_dtg,
                "distance_to_goal_metric": "geodesic",
                "success_distance": float(success_distance),
                "within_success_distance": bool(within_success_distance),
                "success_if_stop_now": bool(within_success_distance),
                "goal_zone_exit_action": bool(exit_by_action),
                "goal_zone_exit_action_reason": exit_action_reason,
                "pixel_path_valid": pix_valid,
                "pixel_reason": pix_reason,
                "pixel_hit_step_idx": pixel_hit_step_idx,
                "goal_zone_exit_pixel": bool(exit_by_pixel),
                "goal_zone_exit_pixel_reason": exit_pixel_reason,
                "reasons": valid_reasons,
            },
        }
        self._append_sample_log(payload)

    def _append_sample_log(self, payload: dict):
        if not self.sample_path:
            return

        cur_pixel = payload.get("input", {}).get("cur_pixel")
        if isinstance(cur_pixel, (list, tuple)) and len(cur_pixel) == 2:
            try:
                pos_key = (int(cur_pixel[0]), int(cur_pixel[1]))
                seen = self.position_dump_counts.get(pos_key, 0)
                self.position_dump_counts[pos_key] = seen + 1
            except Exception:
                pass

        try:
            with open(self.sample_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(payload, ensure_ascii=False) + "\n")
        except Exception as e:
            print(f"Failed to write sample log: {e}")


__all__ = ["BAEAgent"]
