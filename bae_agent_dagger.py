import json
import re
import cv2
import numpy as np
import math
from collections import deque

from pathlib import Path
from scipy.spatial.transform import Rotation as R
from bae import BAEInference
from bae.constants import (
    ACTION_STOP,
    ACTION_MOVE_FORWARD,
    ACTION_TURN_LEFT,
    ACTION_TURN_RIGHT,
    NUM_ACTIONS,
    PIXEL_TO_METER,
)
from bae.mpc_tools import (
    smooth_polyline,
    Polyline2D,
    plan_actions_beam_mpc,
    wrap_pi,
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
        self.pending_pixel_list = None
        self.terminated_by_invalid = False
        self.llm_call_count = 0
        self.image_path = None
        self.sample_path = None
        self.history_action_list = None
        self.position_dump_counts = None
        self.goal_zone_exit_triggered = False
        self.goal_zone_exit_post_budget = None
        self.strict_action_queue = None
        self.late_gt_action_queue = None
        self._late_gt_takeover_active = False
        self._late_gt_takeover_step_threshold = 300
        self._last_action_trace = None
        self._last_progress_pixel = None
        self._stuck_frame_count = 0

        print("BAE Initialization Complete")

    def reset(self, episode_ref, sim=None):
        self.rgb_list = []
        self.pending_pixel_list = []
        self.terminated_by_invalid = False
        self.llm_call_count = 0
        self.history_action_list = []
        self.position_dump_counts = {}
        self.goal_zone_exit_triggered = False
        self.goal_zone_exit_post_budget = None
        self.strict_action_queue = []
        self.late_gt_action_queue = []
        self._late_gt_takeover_active = False
        self._last_action_trace = None
        self._last_progress_pixel = None
        self._stuck_frame_count = 0
        self.episode_id = f"{episode_ref.parent.name}_{episode_ref.stem}"

        self.image_path = Path(self.result_path) / self.episode_id / "image"
        self.sample_path = Path(self.result_path) / self.episode_id / "sample.json"
        self.meta_path = Path(self.result_path) / self.episode_id / "meta.json"
        self.image_path.mkdir(parents=True, exist_ok=True)

        meta_payload = dict(getattr(sim, "meta", {}) or {})
        occ_meta = meta_payload.get("occ")
        if not isinstance(occ_meta, dict):
            occ_meta = {}
        occ_meta.setdefault("occ_image_root", str(self.image_path / "occ"))
        meta_payload["occ"] = occ_meta

        with open(self.meta_path, "w", encoding="utf-8") as f:
            json.dump(meta_payload, f, ensure_ascii=False, indent=2)

        print("BAE Reset Complete for Episode:", self.episode_id)

    def act(self, observations, info):
        sim = observations.get("sim")
        goal_position = observations.get("goal_position")
        instruction = observations.get("instruction")["text"]
        curr_dtg = info.get("distance_to_goal")

        rgb = observations.get("rgb")
        occ_map = sim.get_occ_map()
        occ_traj = sim.get_occ_map_with_trajectory()
        bev_traj = sim.get_bev_map_with_trajectory()

        occ_h, occ_w = occ_map.shape[:2]
        strict_gt_coverage = self._strict_gt_enabled(sim)
        skip_supervision = (
            (not strict_gt_coverage)
            and not bool(getattr(sim.config, "COLLIDABLE", True))
            and bool(
                getattr(sim.config, "SKIP_SUPERVISION_WHEN_IN_OBSTACLE", True)
            )
            and self._is_agent_in_obstacle(sim)
        )

        if self.terminated_by_invalid:
            return self._return_action(ACTION_STOP)

        if self._late_gt_takeover_active and self.late_gt_action_queue:
            queued_action = int(self.late_gt_action_queue.pop(0))
            if queued_action == ACTION_STOP:
                self.late_gt_action_queue = []
            print(
                "Late GT takeover active: "
                f"step={len(self.history_action_list)} action={queued_action}"
            )
            return self._return_action(queued_action)

        if (
            self.goal_zone_exit_post_budget is not None
            and self.goal_zone_exit_post_budget <= 0
        ):
            return self._return_action(ACTION_STOP)
        self._last_action_trace = None

        self.rgb_list.append(rgb)

        model_rgb = self._resize_current_rgb(rgb)

        paths = {
            "rgb": self.save_image(model_rgb, "rgb"),
            "occ": self.save_image(occ_map, "occ"),
            "occ_traj": self.save_image(occ_traj, "occ_traj"),
            "bev_traj": self.save_image(bev_traj, "bev_traj"),
            "bev": sim.bev_path,
        }

        hist = self.build_history_mosaic(
            self.rgb_list[:-1],
            self.history_action_list,
            grid_size=self.history_grid_size,
            target_size=self.history_target_size,
        )
        paths["hist"] = self.save_image(hist, "hist")

        cur_x, cur_y = -1, -1
        pixel_pos = sim.get_current_pixel_position()
        if pixel_pos is not None and len(pixel_pos) >= 2:
            cur_x, cur_y = map(int, pixel_pos)
        self._update_stuck_progress((cur_x, cur_y))

        image_paths = [paths["hist"], paths["rgb"]]

        prev_actions_xml = self._build_prev_actions_xml(self.history_action_list)

        # Always dump oracle planning visualizations for debugging,
        # regardless of whether later correction uses Strategy A or B.
        if skip_supervision:
            oracle_cache = None
        else:
            oracle_cache = self._save_oracle_debug_images(sim, goal_position)

        # Call inference
        try:
            habitat_state = sim.get_habitat_agent_state() if hasattr(sim, "get_habitat_agent_state") else sim.get_agent_state()
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
            # Persist the native Habitat 3D position plus heading.
            agent_pose = [
                float(curr_pos[0]),
                float(curr_pos[1]),
                float(curr_pos[2]),
                curr_yaw_deg,
            ]

            actions, pixels, raw_text, prompt_text, token_probs = (
                self.inference.predict(
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
            )

            if skip_supervision:
                self.llm_call_count += 1
                self._log_skip_supervision_step(
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
                )

                if actions and len(actions) == NUM_ACTIONS:
                    exec_action = int(actions[0])
                    return self._return_action(exec_action)

                return self._return_action(ACTION_STOP)

            valid = True
            valid_reasons = []

            # vlnce action validity
            act_valid, act_reason = self._rollout_actions_valid(
                sim, actions, viz_prefix="action"
            )
            model_action_trace = list(self._last_action_trace or [])
            if not act_valid:
                valid = False
                valid_reasons.append(act_reason)
            action_hit_step_idx = self._parse_rollout_hit_step(act_reason)

            # Goal zone analysis on actions
            exit_by_action, exit_action_reason, action_goal_info = (
                self._goal_zone_analysis_on_actions(sim, actions, goal_position)
            )
            if exit_by_action:
                valid = False
                valid_reasons.append(exit_action_reason)

            # Pixel path validity
            pix_valid = None
            pix_reason = "disabled_for_v3hf"
            pixel_hit_step_idx = None

            # Goal zone analysis on pixels
            exit_by_pixel, exit_pixel_reason, pixel_goal_info = (
                self._goal_zone_analysis_on_pixels(sim, pixels, goal_position)
            )
            if exit_by_pixel:
                valid = False
                valid_reasons.append(exit_pixel_reason)

            if exit_by_action or exit_by_pixel:
                if not self.goal_zone_exit_triggered:
                    self.goal_zone_exit_triggered = True
                    self.goal_zone_exit_post_budget = 20

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

            gt_true_pair = None
            gt_build_reason = None

            context_payload = {
                "instruction": instruction,
                "prompt_text": prompt_text,
                "image_paths": image_paths,
                "occ_image_path": paths["occ"],
                "cur_pixel": [cur_x, cur_y],
                "occ_shape": [occ_h, occ_w],
                "action_goal_info": action_goal_info,
                "pixel_goal_info": pixel_goal_info,
            }

            is_overshoot = any(r.startswith("goal_zone_exit_") for r in valid_reasons)

            # Strategy A: Re-plan path using A* + MPC
            gt_output, meta, reason = self._handle_standard_correction(
                sim, goal_position, oracle_cache
            )
            if is_overshoot:
                # Strategy B: Stop exactly at entry point
                gt_output, meta, reason = self._handle_goal_zone_exit(
                    sim, actions, pixels, context_payload
                )

            if strict_gt_coverage:
                gt_output, meta, reason = self._enforce_strict_gt_output(
                    sim=sim,
                    goal_position=goal_position,
                    gt_output=gt_output,
                    meta=meta,
                    reason=reason,
                )

            # Assemble final GT pair if correction was successful
            if gt_output:
                if isinstance(reason, dict):
                    gt_build_reason = reason
                else:
                    gt_build_reason = {"detail": reason}
                gt_true_pair = {
                    "input": self._construct_gt_input(context_payload),
                    "output": gt_output,
                    **meta,
                }

            self.llm_call_count += 1
            self._log_inference_step(
                instruction,
                prompt_text,
                image_paths,
                paths["occ"],
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
                model_action_trace,
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
                gt_true_pair,
                gt_build_reason,
            )

            teacher_exec_action = None
            if strict_gt_coverage and isinstance(gt_true_pair, dict):
                gt_output = gt_true_pair.get("output")
                gt_actions = self._extract_actions_from_gt_output(gt_output)
                if gt_actions and len(gt_actions) == NUM_ACTIONS:
                    if not self.strict_action_queue:
                        self.strict_action_queue = list(gt_actions)
                    elif len(self.strict_action_queue) <= 1:
                        self.strict_action_queue = list(gt_actions)
                    elif (
                        int(self.strict_action_queue[0]) == ACTION_STOP
                        and int(gt_actions[0]) != ACTION_STOP
                    ):
                        self.strict_action_queue = list(gt_actions)

                    teacher_exec_action = int(self.strict_action_queue.pop(0))
                    if teacher_exec_action == ACTION_STOP:
                        self.strict_action_queue = []

            if strict_gt_coverage and teacher_exec_action is not None:
                return self._return_action(teacher_exec_action)

            if self._should_force_gt_takeover(curr_dtg=curr_dtg, sim=sim):
                if isinstance(gt_true_pair, dict):
                    gt_output = gt_true_pair.get("output")
                    gt_actions = self._extract_actions_from_gt_output(gt_output)
                    takeover_action = self._arm_gt_takeover(
                        gt_actions=gt_actions,
                        reason=(
                            "Late GT takeover triggered: "
                            f"step={len(self.history_action_list)} "
                            f"dtg={curr_dtg} "
                            f"success_radius={self._get_eval_success_distance(sim)}"
                        ),
                    )
                    if takeover_action is not None:
                        return self._return_action(takeover_action)
                print(
                    "Late GT takeover armed, but GT actions unavailable for this step."
                )

            stuck_override_action = None
            if isinstance(gt_true_pair, dict):
                gt_output = gt_true_pair.get("output")
                gt_actions = self._extract_actions_from_gt_output(gt_output)
                if gt_actions and len(gt_actions) == NUM_ACTIONS:
                    if self._stuck_frame_count > 15:
                        takeover_action = self._arm_gt_takeover(
                            gt_actions=gt_actions,
                            reason=(
                                "Stuck GT takeover triggered: "
                                f"stuck_frames={self._stuck_frame_count}"
                            ),
                        )
                        if takeover_action is not None:
                            self._stuck_frame_count = 0
                            return self._return_action(takeover_action)
                    stuck_override_action = self._maybe_get_stuck_override_action(
                        gt_actions=gt_actions,
                        curr_dtg=curr_dtg,
                        sim=sim,
                    )
                    if stuck_override_action is not None:
                        print(
                            "Stuck recovery triggered: "
                            f"stuck_frames={self._stuck_frame_count}, "
                            f"override_with_gt={stuck_override_action}"
                        )
                        self._stuck_frame_count = 0
                        return self._return_action(stuck_override_action)

            action_hard_fail = False
            if not act_valid and not self._is_action_wall_reason(act_reason):
                action_hard_fail = True

            if actions and len(actions) == NUM_ACTIONS and not action_hard_fail:
                exec_action = int(actions[0])

                if stop_in_actions:
                    if stop_outside_goal:
                        print(
                            "Model STOP outside goal detected. "
                            "Replace first action with <FWD>."
                        )
                        return self._return_action(ACTION_MOVE_FORWARD)
                    return self._return_action(ACTION_STOP)

                if action_hit_step_idx is not None:
                    print(
                        "Warning: Rollout wall hit at step "
                        f"{action_hit_step_idx}, continuing."
                    )
                if any(self._is_pixel_wall_reason(r) for r in valid_reasons):
                    print("Warning: Pixel path hits wall, continuing.")

                print(f"BAE predicted actions: {actions}")

                if pixels:
                    print(f"BAE predicted pixels: {len(pixels)} waypoints")

                print(f"Raw output: {raw_text}")
                return self._return_action(exec_action)
            else:
                print(
                    f"Invalid output encountered. Reasons: {valid_reasons}"
                )
                if not bool(getattr(sim.config, "COLLIDABLE", True)):
                    exec_action = int(actions[0])
                    print(f"Raw output: {raw_text}")
                    return self._return_action(exec_action)

                if isinstance(gt_true_pair, dict):
                    gt_output = gt_true_pair.get("output")
                    gt_actions = self._extract_actions_from_gt_output(gt_output)
                    if gt_actions and len(gt_actions) == NUM_ACTIONS:
                        gt_first_action = int(gt_actions[0])
                        print(
                            "Invalid output fallback to GT first action: "
                            f"action={gt_first_action}"
                        )
                        return self._return_action(gt_first_action)

                print("GT fallback unavailable; terminating episode with STOP.")
                self.terminated_by_invalid = True
                return self._return_action(ACTION_STOP)

        except Exception as e:
            print(f"Error during inference: {e}")
            import traceback

            traceback.print_exc()
            self.llm_call_count += 1
            return self._return_action(ACTION_STOP)

    def _return_action(self, action: int):
        """Centralized action return with post-exit budget enforcement."""
        out_action = int(action)

        if (
            self.goal_zone_exit_post_budget is not None
            and self.goal_zone_exit_post_budget <= 0
        ):
            out_action = ACTION_STOP

        if (
            self.goal_zone_exit_post_budget is not None
            and self.goal_zone_exit_post_budget > 0
        ):
            self.goal_zone_exit_post_budget -= 1

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

    @staticmethod
    def _get_goal_stop_distance(_sim=None) -> float:
        return 1.0

    @staticmethod
    def _get_eval_success_distance(sim=None) -> float:
        get_success_distance = getattr(sim, "get_success_distance", None)
        if callable(get_success_distance):
            try:
                return float(get_success_distance())
            except Exception:
                pass
        return 3.0

    def _distance_outside_goal(self, curr_dtg, sim=None) -> bool:
        try:
            return float(curr_dtg) > self._get_goal_stop_distance(sim)
        except Exception:
            return False

    def _is_eval_success(self, curr_dtg, sim=None) -> bool:
        try:
            return float(curr_dtg) < self._get_eval_success_distance(sim)
        except Exception:
            return False

    def _should_force_gt_takeover(self, curr_dtg, sim=None) -> bool:
        if len(self.history_action_list) < int(self._late_gt_takeover_step_threshold):
            return False
        return not self._is_eval_success(curr_dtg, sim)

    def _arm_gt_takeover(self, gt_actions, reason="GT takeover triggered"):
        if not gt_actions or len(gt_actions) != NUM_ACTIONS:
            return None
        self._late_gt_takeover_active = True
        self.late_gt_action_queue = list(gt_actions)
        takeover_action = int(self.late_gt_action_queue.pop(0))
        if takeover_action == ACTION_STOP:
            self.late_gt_action_queue = []
        print(f"{reason} action={takeover_action}")
        return takeover_action

    def _maybe_get_stuck_override_action(self, gt_actions, curr_dtg, sim=None):
        if self._stuck_frame_count < 5:
            return None
        if not gt_actions:
            return None

        gt_action = int(gt_actions[0])
        if gt_action == ACTION_STOP and self._distance_outside_goal(curr_dtg, sim):
            return None
        return gt_action

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
        mosaic = grid.swapaxes(1, 2).reshape(th, tw, 3)

        return mosaic

    def _resize_current_rgb(self, rgb: np.ndarray) -> np.ndarray:
        if rgb is None:
            return rgb
        h, w = rgb.shape[:2]
        target = (self.current_resize_w, self.current_resize_h)
        if (w, h) == target:
            return rgb
        return cv2.resize(rgb, target, interpolation=cv2.INTER_AREA)

    def save_image(self, img: np.ndarray, prefix: str) -> str:
        """Save numpy image to temp file and return path."""
        save_dir = Path(self.image_path) / prefix
        save_dir.mkdir(parents=True, exist_ok=True)

        file_name = f"{len(self.rgb_list) - 1}.png"
        filepath = save_dir / file_name

        cv2.imwrite(str(filepath), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

        return str(filepath)

    def _build_prev_actions_xml(self, history_actions: list[int]) -> str:
        """
        Build Action History XML for prompt.
        Uses last 5 executed actions (NEW -> OLD). Pads with <None> if不足5个。
        """
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
                if isinstance(trace, list):
                    self._last_action_trace = trace
                else:
                    self._last_action_trace = []
                try:
                    action_img = sim.get_occ_map_with_actions(
                        actions,
                        traj_color=(255, 165, 0),
                        trace=self._last_action_trace,
                    )
                except TypeError:
                    action_img = sim.get_occ_map_with_actions(
                        actions, traj_color=(255, 165, 0)
                    )
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

            elif action == ACTION_TURN_LEFT:
                curr_rot = curr_rot * rot_left

            elif action == ACTION_TURN_RIGHT:
                curr_rot = curr_rot * rot_right

            elif action == ACTION_MOVE_FORWARD:
                world_fwd = curr_rot.apply(local_fwd_vec)
                next_pos = curr_pos + (world_fwd * step_size)

                px_start = sim.transform_from_world_to_pixel(curr_pos)
                px_end = sim.transform_from_world_to_pixel(next_pos)

                if not self._segment_is_clear(grid, *map_size, px_start, px_end, sim):
                    return False, f"action_rollout_hit_wall_step_{step_idx}"

                curr_pos = next_pos

            else:
                return False, f"unknown_action_{action}"

        return True, "ok"

    def _get_collision_grid(self, sim):
        margins = getattr(sim, "margins", [])
        safe_grids = getattr(sim, "safe_passable_grids", {})

        grid = None
        if margins and safe_grids:
            grid = safe_grids.get(margins[-1])

        if grid is None:
            grid = getattr(sim, "passable_grid", None)

        return grid

    def _is_agent_in_obstacle(self, sim) -> bool:
        margins = getattr(sim, "margins", [])
        safe_grids = getattr(sim, "safe_passable_grids", {})
        if not margins or not safe_grids:
            return False

        grid = safe_grids.get(margins[-1])
        if grid is None:
            return False

        try:
            agent_state = sim.get_agent_state()
            curr_pos = np.array(agent_state.position, dtype=np.float32)
            px, py = sim.transform_from_world_to_pixel(curr_pos)

            if not (0 <= px < int(sim.map_width) and 0 <= py < int(sim.map_height)):
                return False

            return not bool(grid[py][px])
        except Exception:
            return False

    def _segment_is_clear(self, grid, map_w, map_h, start_px, end_px, sim):
        for x, y in sim._bresenham_line(start_px, end_px):
            if not (0 <= x < map_w and 0 <= y < map_h):
                return False
            if not grid[y][x]:
                return False
        return True

    @staticmethod
    def _parse_rollout_hit_step(reason):
        m = re.match(
            r"^(?:action|habitat)_rollout_hit_wall_step_(\d+)$", str(reason)
        )
        if not m:
            return None
        try:
            return int(m.group(1))
        except Exception:
            return None

    @staticmethod
    def _parse_pixel_hit_step(reason):
        m = re.match(r"^pixel_branch_hit_wall_seg_(\d+)$", str(reason))
        if not m:
            return None
        try:
            return int(m.group(1))
        except Exception:
            return None

    def _goal_zone_analysis_on_actions(self, sim, actions, goal_position):
        """
        Analyzes if the agent enters and then leaves the goal zone.
        Returns: (is_error, message, info_dict)
        """
        if goal_position is None or actions is None or len(actions) != NUM_ACTIONS:
            return False, "skip", {"first_enter_idx": None}

        GOAL_RADIUS = max(min(3, sim.path_length * 0.2), 1)
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
        is_inside = curr_dist <= GOAL_RADIUS

        first_enter_idx = -1 if is_inside else None

        for step_idx, action in enumerate(map(int, actions)):
            if action == ACTION_TURN_LEFT:
                curr_rot = curr_rot * rot_left
            elif action == ACTION_TURN_RIGHT:
                curr_rot = curr_rot * rot_right
            elif action == ACTION_MOVE_FORWARD:
                world_fwd = curr_rot.apply(local_fwd_vec)
                curr_pos = curr_pos + (world_fwd * step_size)

            curr_dist = float(np.linalg.norm(curr_pos[:2] - goal_xy))
            is_now_inside = curr_dist <= GOAL_RADIUS

            #  Outside -> Inside
            if not is_inside and is_now_inside:
                is_inside = True
                first_enter_idx = step_idx

            #  Inside -> Outside
            elif is_inside and not is_now_inside:
                return (
                    True,
                    f"goal_zone_exit_by_action_step_{step_idx}",
                    {"first_enter_idx": first_enter_idx},
                )

        return False, "ok", {"first_enter_idx": first_enter_idx}

    def _pixel_path_valid(self, sim, pixel_waypoints_norm):
        if not pixel_waypoints_norm:
            return False, "missing_pixel_waypoints"

        grid = self._get_collision_grid(sim)
        if grid is None:
            return False, "missing_passable_grid"

        if hasattr(grid, "shape"):
            map_h, map_w = grid.shape[:2]
        else:
            map_h = len(grid)
            map_w = len(grid[0]) if map_h > 0 else 0

        if map_w <= 0 or map_h <= 0:
            return False, "bad_occ_shape"

        waypoints_px = []
        for pt in pixel_waypoints_norm:
            if not (isinstance(pt, (list, tuple)) and len(pt) == 2):
                return False, "bad_pixel_waypoint_format"
            raw_x, raw_y = int(pt[0]), int(pt[1])
            px_pt = self._denorm_to_occ_pixel(raw_x, raw_y, map_w, map_h)
            waypoints_px.append(px_pt)

        if waypoints_px:
            pixel_img = sim.get_occ_map_with_pixels(
                waypoints_px, path_color=(255, 0, 255)
            )
            self.save_image(pixel_img, "pixel")

        for idx, (p_start, p_end) in enumerate(zip(waypoints_px, waypoints_px[1:])):
            is_clear = self._segment_is_clear(grid, map_w, map_h, p_start, p_end, sim)

            if not is_clear:
                return False, f"pixel_branch_hit_wall_seg_{idx}"

        return True, "ok"

    def _denorm_to_occ_pixel(self, x_norm, y_norm, occ_w, occ_h):
        px = int(round((int(x_norm) / 1000.0) * float(occ_w)))
        py = int(round((int(y_norm) / 1000.0) * float(occ_h)))
        px = max(0, min(px, occ_w - 1))
        py = max(0, min(py, occ_h - 1))
        return px, py

    def _goal_zone_analysis_on_pixels(self, sim, pixel_waypoints_norm, goal_position):
        """
        Analyzes if the pixel path enters and then leaves the goal zone.
        Pipeline: Norm Coord -> Occ Pixel -> World Coord -> Distance Check.
        """
        if goal_position is None or not pixel_waypoints_norm:
            return False, "skip", {"first_enter_idx": None}

        GOAL_RADIUS = max(min(3, sim.path_length * 0.2), 1)
        map_w, map_h = int(sim.map_width), int(sim.map_height)

        goal_xy = np.array(goal_position[:2], dtype=np.float32)

        is_inside = False
        first_enter_idx = None

        for idx, pt_norm in enumerate(pixel_waypoints_norm):
            if not (isinstance(pt_norm, (list, tuple)) and len(pt_norm) == 2):
                return (
                    True,
                    "goal_zone_exit_bad_pixel_format",
                    {"first_enter_idx": first_enter_idx},
                )

            # Step A: Norm -> Occ Pixel
            norm_x, norm_y = int(pt_norm[0]), int(pt_norm[1])
            px_pt = self._denorm_to_occ_pixel(norm_x, norm_y, map_w, map_h)

            # Step B: Occ Pixel -> World
            world_pt_tuple = sim.transform_from_pixel_to_world(px_pt)
            world_pos = np.array(world_pt_tuple[:2], dtype=np.float32)

            dist = float(np.linalg.norm(world_pos - goal_xy))
            is_now_inside = dist <= GOAL_RADIUS

            if not is_inside and is_now_inside:
                is_inside = True
                first_enter_idx = idx

            elif is_inside and not is_now_inside:
                return (
                    True,
                    f"goal_zone_exit_by_pixel_step_{idx}",
                    {"first_enter_idx": first_enter_idx},
                )

        return False, "ok", {"first_enter_idx": first_enter_idx}

    def _log_inference_step(
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
        gt_true_pair,
        gt_build_reason,
    ):
        """Constructs and writes the comprehensive sample log."""
        gt_fallback_level = None
        gt_rollout_valid = None
        gt_rollout_reason = None
        planning_start_snapped = None
        planning_goal_snapped = None
        if isinstance(gt_build_reason, dict):
            gt_fallback_level = gt_build_reason.get("gt_fallback_level")
            gt_rollout_valid = gt_build_reason.get("gt_rollout_valid")
            gt_rollout_reason = gt_build_reason.get("gt_rollout_reason")
            planning_start_snapped = gt_build_reason.get("planning_start_snapped")
            planning_goal_snapped = gt_build_reason.get("planning_goal_snapped")

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
            "gt_true_pair": gt_true_pair,
            "gt_build_reason": gt_build_reason,
            "gt_fallback_level": gt_fallback_level,
            "gt_rollout_valid": gt_rollout_valid,
            "gt_rollout_reason": gt_rollout_reason,
            "planning_start_snapped": planning_start_snapped,
            "planning_goal_snapped": planning_goal_snapped,
        }
        self._append_sample_log(payload)

    def _log_skip_supervision_step(
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
    ):
        """Writes a lightweight dump when obstacle supervision is skipped."""
        payload = {
            "episode_id": self.episode_id,
            "llm_call_idx": self.llm_call_count,
            "step_idx": len(self.rgb_list) - 1,
            "prompt_type": self.prompt_type,
            "in_obstacle": True,
            "input": {
                "instruction": instruction,
                "prompt_text": prompt_text,
                "image_paths": image_paths,
                "occ_image_path": occ_image_path,
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
                "distance_to_goal_metric": "geodesic",
                "success_distance": float(success_distance),
                "within_success_distance": bool(within_success_distance),
                "success_if_stop_now": bool(within_success_distance),
            },
            "gt_fallback_level": None,
            "gt_rollout_valid": None,
            "gt_rollout_reason": None,
            "planning_start_snapped": None,
            "planning_goal_snapped": None,
        }
        self._append_sample_log(payload)

    def _append_sample_log(self, payload: dict):
        if not self.sample_path:
            return

        # Limit repeated dumps at the same position within one episode.
        cur_pixel = payload.get("input", {}).get("cur_pixel")
        if isinstance(cur_pixel, (list, tuple)) and len(cur_pixel) == 2:
            try:
                pos_key = (int(cur_pixel[0]), int(cur_pixel[1]))
                seen = self.position_dump_counts.get(pos_key, 0)
                # if seen >= 2:
                #     return
                self.position_dump_counts[pos_key] = seen + 1
            except Exception:
                pass

        try:
            with open(self.sample_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(payload, ensure_ascii=False) + "\n")
        except Exception as e:
            print(f"Failed to write sample log: {e}")

    def _handle_goal_zone_exit(self, sim, actions, pixels, context):
        """
        Correction Strategy A: The agent walked THROUGH the goal.
        Fix: Truncate actions/pixels to stop exactly when it entered the zone.
        """
        # 1. Truncate Actions
        enter_idx_act = context["action_goal_info"].get("first_enter_idx")
        gt_actions = self._truncate_and_pad_actions(actions, enter_idx_act)
        gt_output = {"vlnce": self._action_xml_from_ids(gt_actions)}

        meta = {"is_true": True, "source": "oracle_stop_at_goal"}
        reason = {"mode": "stop_at_goal"}

        return gt_output, meta, reason

    def _handle_standard_correction(self, sim, goal_position, oracle_cache=None):
        """
        Correction Strategy B: The agent hit a wall, is lost, or stopped too early.
        Fix: Generate a new path from current location using A* and MPC.
        """
        # 1. Plan Actions
        gt_actions = None
        act_reason = "cache_miss"
        if oracle_cache is not None:
            gt_actions = oracle_cache.get("gt_actions")
            act_reason = oracle_cache.get("actions_reason", act_reason)

        if gt_actions is None:
            gt_actions, act_reason = self._build_gt_actions6(sim, goal_position)

        if gt_actions is None:
            return None, {}, None

        gt_output = {"vlnce": self._action_xml_from_ids(gt_actions)}
        reason = {"actions": act_reason}

        meta = {"is_true": True, "source": "oracle_astar_fallback"}

        return gt_output, meta, reason

    def _construct_gt_input(self, context):
        """Extracts standard input fields for the GT pair."""
        return {
            "instruction": context["instruction"],
            "prompt_text": context["prompt_text"],
            "image_paths": context["image_paths"],
            "occ_image_path": context.get("occ_image_path"),
            "cur_pixel": context["cur_pixel"],
            "occ_shape": context["occ_shape"],
        }

    def _strict_gt_enabled(self, sim) -> bool:
        cfg = getattr(sim, "config", None)
        return bool(getattr(cfg, "GT_STRICT_COVERAGE", False))

    def _strict_gt_dataset_goal_only(self, sim) -> bool:
        cfg = getattr(sim, "config", None)
        # Default to True to keep GT target anchored to dataset goal in strict mode.
        return bool(getattr(cfg, "STRICT_GT_DATASET_GOAL_ONLY", True))

    def _strict_gt_prefer_habitat_greedy(self, sim) -> bool:
        cfg = getattr(sim, "config", None)
        return bool(getattr(cfg, "STRICT_GT_PREFER_HABITAT_GREEDY", True))

    def _extract_actions_from_gt_output(self, gt_output):
        if not isinstance(gt_output, dict):
            return None
        xml = gt_output.get("vlnce")
        if not isinstance(xml, str):
            return None
        m = re.search(r"<action>(.*?)</action>", xml)
        if not m:
            return None
        token_str = m.group(1).strip()
        if not token_str:
            return None
        toks = [t.strip() for t in token_str.split(",")]
        tok2id = {
            "<STOP>": ACTION_STOP,
            "<FWD>": ACTION_MOVE_FORWARD,
            "<LEFT>": ACTION_TURN_LEFT,
            "<RIGHT>": ACTION_TURN_RIGHT,
        }
        actions = []
        for tok in toks:
            if tok not in tok2id:
                return None
            actions.append(tok2id[tok])
        if len(actions) != NUM_ACTIONS:
            return None
        return actions

    def _collect_planning_meta(self, sim):
        out = {
            "planning_start_snapped": False,
            "planning_goal_snapped": False,
        }
        dbg = getattr(sim, "last_astar_debug", None)
        if isinstance(dbg, dict):
            out["planning_start_snapped"] = bool(dbg.get("start_snapped", False))
            out["planning_goal_snapped"] = bool(dbg.get("goal_snapped", False))
            if dbg.get("used_margin") is not None:
                out["planning_margin"] = int(dbg["used_margin"])
            if dbg.get("fallback") is not None:
                out["planning_fallback"] = str(dbg["fallback"])
        return out

    def _snap_pixel_to_passable(self, sim, pixel, grid, max_radius=32):
        px, py = int(pixel[0]), int(pixel[1])
        h = len(grid)
        w = len(grid[0]) if h else 0
        if not (0 <= px < w and 0 <= py < h):
            return None
        if bool(grid[py][px]):
            return (px, py)

        snap_fn = getattr(sim, "_snap_to_nearest_passable", None)
        if callable(snap_fn):
            try:
                snapped = snap_fn((px, py), grid, max_radius=max_radius)
                if snapped is not None:
                    return (int(snapped[0]), int(snapped[1]))
            except Exception:
                pass

        for r in range(1, int(max_radius) + 1):
            x0 = max(px - r, 0)
            x1 = min(px + r, w - 1)
            y0 = max(py - r, 0)
            y1 = min(py + r, h - 1)
            best = None
            best_d2 = None
            for x in range(x0, x1 + 1):
                for y in (y0, y1):
                    if not grid[y][x]:
                        continue
                    d2 = (x - px) * (x - px) + (y - py) * (y - py)
                    if best_d2 is None or d2 < best_d2:
                        best_d2 = d2
                        best = (x, y)
            for y in range(y0 + 1, y1):
                for x in (x0, x1):
                    if not grid[y][x]:
                        continue
                    d2 = (x - px) * (x - px) + (y - py) * (y - py)
                    if best_d2 is None or d2 < best_d2:
                        best_d2 = d2
                        best = (x, y)
            if best is not None:
                return best
        return None

    def _build_proxy_goal_position(self, sim, goal_position):
        if goal_position is None or len(goal_position) < 2:
            return None
        grid = self._get_collision_grid(sim)
        if grid is None:
            return None

        h = len(grid)
        w = len(grid[0]) if h else 0
        if h <= 0 or w <= 0:
            return None

        state = sim.get_agent_state()
        start_world = np.array([state.position[0], state.position[1], 0.0], dtype=np.float32)
        goal_world = np.array([goal_position[0], goal_position[1], 0.0], dtype=np.float32)
        start_px = sim.transform_from_world_to_pixel(start_world)
        goal_px = sim.transform_from_world_to_pixel(goal_world)

        start_adj = self._snap_pixel_to_passable(sim, start_px, grid, max_radius=32)
        if start_adj is None:
            return None

        q = deque([start_adj])
        visited = {start_adj}
        best = start_adj
        best_d2 = (start_adj[0] - goal_px[0]) ** 2 + (start_adj[1] - goal_px[1]) ** 2
        neighbors = (
            (1, 0),
            (-1, 0),
            (0, 1),
            (0, -1),
            (1, 1),
            (1, -1),
            (-1, 1),
            (-1, -1),
        )

        while q:
            cx, cy = q.popleft()
            d2 = (cx - goal_px[0]) ** 2 + (cy - goal_px[1]) ** 2
            if d2 < best_d2:
                best_d2 = d2
                best = (cx, cy)

            for dx, dy in neighbors:
                nx, ny = cx + dx, cy + dy
                if not (0 <= nx < w and 0 <= ny < h):
                    continue
                if (nx, ny) in visited:
                    continue
                if not grid[ny][nx]:
                    continue
                if dx != 0 and dy != 0:
                    if not grid[cy][nx] or not grid[ny][cx]:
                        continue
                visited.add((nx, ny))
                q.append((nx, ny))

        proxy_xy = sim.transform_from_pixel_to_world(best)
        return [float(proxy_xy[0]), float(proxy_xy[1]), 0.0]

    def _build_strict_gt_actions(self, sim, goal_position):
        candidates = []
        dataset_goal_only = self._strict_gt_dataset_goal_only(sim)
        if goal_position is not None and len(goal_position) >= 2:
            candidates.append(("oracle_goal_astar_mpc", 0, goal_position))
            if not dataset_goal_only:
                proxy_goal = self._build_proxy_goal_position(sim, goal_position)
                if proxy_goal is not None:
                    goal_arr = np.array(goal_position[:2], dtype=np.float32)
                    proxy_arr = np.array(proxy_goal[:2], dtype=np.float32)
                    if float(np.linalg.norm(proxy_arr - goal_arr)) > 1e-6:
                        candidates.append(("oracle_proxy_goal_astar_mpc", 1, proxy_goal))

        candidates.append(("oracle_stop_fallback", 2, None))
        last_reason = "strict_not_run"
        last_plan_meta = self._collect_planning_meta(sim)

        for source, level, candidate_goal in candidates:
            if candidate_goal is None:
                actions = [ACTION_STOP] * NUM_ACTIONS
                actions_reason = "fallback_stop"
            else:
                actions, actions_reason = self._build_gt_actions6(sim, candidate_goal)

            if actions is None or len(actions) != NUM_ACTIONS:
                last_reason = f"invalid_actions:{actions_reason}"
                last_plan_meta = self._collect_planning_meta(sim)
                continue

            rollout_valid, rollout_reason = self._rollout_actions_valid(
                sim, actions, viz_prefix=None
            )
            last_reason = rollout_reason
            last_plan_meta = self._collect_planning_meta(sim)
            if rollout_valid:
                return actions, {
                    "gt_source": source,
                    "gt_fallback_level": int(level),
                    "actions_reason": str(actions_reason),
                    "gt_rollout_valid": True,
                    "gt_rollout_reason": str(rollout_reason),
                    **last_plan_meta,
                }

        actions = [ACTION_STOP] * NUM_ACTIONS
        rollout_valid, rollout_reason = self._rollout_actions_valid(
            sim, actions, viz_prefix=None
        )
        return actions, {
            "gt_source": "oracle_stop_last_resort",
            "gt_fallback_level": 3,
            "actions_reason": str(last_reason),
            "gt_rollout_valid": bool(rollout_valid),
            "gt_rollout_reason": str(rollout_reason),
            **last_plan_meta,
        }

    def _enforce_strict_gt_output(self, sim, goal_position, gt_output, meta, reason):
        strict_meta = dict(meta) if isinstance(meta, dict) else {}
        strict_meta.setdefault("is_true", True)
        strict_meta.setdefault("source", "oracle_strict")

        strict_reason = dict(reason) if isinstance(reason, dict) else {}
        current_plan_meta = self._collect_planning_meta(sim)
        for k, v in current_plan_meta.items():
            strict_reason.setdefault(k, v)

        final_output = dict(gt_output) if isinstance(gt_output, dict) else {}
        actions = self._extract_actions_from_gt_output(final_output)
        rollout_valid = False
        rollout_reason = "missing_gt_actions"
        if actions is not None:
            rollout_valid, rollout_reason = self._rollout_actions_valid(
                sim, actions, viz_prefix=None
            )

        if not rollout_valid:
            actions, fallback_info = self._build_strict_gt_actions(sim, goal_position)
            final_output["vlnce"] = self._action_xml_from_ids(actions)
            strict_reason.update(fallback_info)
            rollout_valid = bool(fallback_info.get("gt_rollout_valid", False))
            rollout_reason = str(fallback_info.get("gt_rollout_reason", "unknown"))
        else:
            final_output["vlnce"] = self._action_xml_from_ids(actions)
            strict_reason.setdefault("gt_source", "oracle_existing_gt")
            strict_reason.setdefault("gt_fallback_level", 0)
            strict_reason["gt_rollout_valid"] = True
            strict_reason["gt_rollout_reason"] = str(rollout_reason)

        if not rollout_valid:
            stop_actions = [ACTION_STOP] * NUM_ACTIONS
            final_output["vlnce"] = self._action_xml_from_ids(stop_actions)
            rv, rr = self._rollout_actions_valid(sim, stop_actions, viz_prefix=None)
            strict_reason["gt_source"] = "oracle_stop_last_resort"
            strict_reason["gt_fallback_level"] = max(
                int(strict_reason.get("gt_fallback_level", 0)), 3
            )
            strict_reason["gt_rollout_valid"] = bool(rv)
            strict_reason["gt_rollout_reason"] = str(rr)
            rollout_valid = bool(rv)

        strict_reason.setdefault("mode", "strict_gt")
        strict_reason["gt_rollout_valid"] = bool(
            strict_reason.get("gt_rollout_valid", rollout_valid)
        )
        strict_reason["gt_rollout_reason"] = str(
            strict_reason.get("gt_rollout_reason", rollout_reason)
        )
        strict_reason["gt_fallback_level"] = int(
            strict_reason.get("gt_fallback_level", 0)
        )
        strict_reason["planning_start_snapped"] = bool(
            strict_reason.get("planning_start_snapped", False)
        )
        strict_reason["planning_goal_snapped"] = bool(
            strict_reason.get("planning_goal_snapped", False)
        )

        return final_output, strict_meta, strict_reason

    def _truncate_and_pad_actions(self, actions, enter_idx):
        """Helper: Keeps actions up to entry index, fills rest with STOP."""
        if (
            not actions
            or len(actions) != NUM_ACTIONS
            or enter_idx is None
            or enter_idx < 0
        ):
            return [ACTION_STOP] * NUM_ACTIONS

        keep_count = enter_idx + 1
        pad_count = NUM_ACTIONS - keep_count
        return list(actions[:keep_count]) + [ACTION_STOP] * pad_count

    def _truncate_and_pad_pixels(self, pixels, enter_idx):
        """Helper: Keeps pixels up to entry index, pads last point to length 13."""
        if not pixels or enter_idx is None or not (0 <= enter_idx < len(pixels)):
            return None

        try:
            # 1. Slice valid path
            valid_path = [list(map(int, p)) for p in pixels[: enter_idx + 1]]

            if not valid_path:
                return None

            # 2. Pad to target length (13)
            TARGET_LEN = 13
            last_pt = valid_path[-1]
            while len(valid_path) < TARGET_LEN:
                valid_path.append(last_pt)

            # 3. Ensure exact length
            final_path = valid_path[:TARGET_LEN]

            return self._pixel_tokstr_from_norm_points(final_path)

        except Exception as e:
            print(f"Error truncating pixels: {e}")
            return None

    @staticmethod
    def _action_xml_from_ids(action_ids):
        id2tok = {
            ACTION_STOP: "<STOP>",
            ACTION_MOVE_FORWARD: "<FWD>",
            ACTION_TURN_LEFT: "<LEFT>",
            ACTION_TURN_RIGHT: "<RIGHT>",
        }
        toks = [id2tok.get(int(a), "<STOP>") for a in action_ids]
        return "<action>" + ",".join(toks) + "</action>"

    def _pixel_tokstr_from_norm_points(self, waypoints_norm):
        parts = []
        for x, y in waypoints_norm:
            xi = max(0, min(int(x), 999))
            yi = max(0, min(int(y), 999))
            parts.append(f"[{self._tok(xi)},{self._tok(yi)}]")
        return "[" + ",".join(parts) + "]"

    def _build_gt_pixel_tokstr_13(self, sim, goal_position):
        """
        Generates the ground truth pixel string (1 current + 12 future waypoints).
        Strategy: Plan full path -> Slice 60 steps -> Pad to 60 -> Sample 12 -> Tokenize.
        """
        # 1. Validation
        if goal_position is None or len(goal_position) < 2:
            return None, "missing_goal_position"

        # 2. Path Planning (A*)
        agent_state = sim.get_agent_state()
        pos_xy = np.array(agent_state.position[:2], dtype=np.float32)
        goal_xy = np.array(goal_position[:2], dtype=np.float32)

        try:
            dense_path = self._plan_dense_path_pixels(sim, pos_xy, goal_xy)
        except Exception as e:
            return None, f"gt_astar_failed:{e}"

        if len(dense_path) == 0:
            return None, "empty_dense_path"

        # 3. Windowing & Sampling Configuration
        LOOKAHEAD_STEPS = 60
        NUM_SAMPLES = 12

        current_px = dense_path[0]
        future_window = dense_path[1 : 1 + LOOKAHEAD_STEPS]

        if not future_window:
            future_window = [current_px]

        # 4. Padding (Extension Strategy)

        needed_pad = LOOKAHEAD_STEPS - len(future_window)
        if needed_pad > 0:
            future_window.extend([future_window[-1]] * needed_pad)

        # 5. Downsampling
        indices = np.linspace(0, LOOKAHEAD_STEPS - 1, NUM_SAMPLES)
        sampled_pts = [future_window[int(np.round(i))] for i in indices]

        # 6. Final Assembly (1 + 12 = 13 points)
        final_waypoints = [current_px] + sampled_pts

        # Visualization should show the full A* trajectory (dense path),
        # while tokenization still uses the 13-point waypoint format.
        dense_pixels = [tuple(map(int, px)) for px in dense_path]
        astar_img = sim.get_occ_map_with_pixels(
            dense_pixels[1:], path_color=(138, 43, 226)
        )
        self.save_image(astar_img, "astar")

        # 7. Tokenization
        occ_w, occ_h = int(sim.map_width), int(sim.map_height)
        return self._pixel_tokstr_from_pixels(final_waypoints, occ_w, occ_h), "ok"

    def _pixel_tokstr_from_pixels(self, waypoints_px, occ_w, occ_h):
        tokens = []
        for x, y in waypoints_px:
            xn, yn = self._norm_xy_from_pixel(x, y, occ_w, occ_h)
            tokens.append(f"[{self._tok(xn)},{self._tok(yn)}]")
        return f"[{','.join(tokens)}]"

    @staticmethod
    def _tok(n):
        return f"<{int(n)}>"

    def _build_gt_actions6(self, sim, goal_position):
        """
        Orchestrates the Oracle action generation pipeline:
        1. Validation & State Extraction
        2. A* Path Planning (Pixel Space)
        3. Path Smoothing (World Space)
        4. MPC Trajectory Generation
        5. Kinematic Simulation -> Discrete Actions
        """
        # 1. Validation & Pre-checks
        if goal_position is None or len(goal_position) < 2:
            return None, "missing_goal_position"

        grid = self._get_collision_grid(sim)
        if grid is None:
            return None, "missing_passable_grid"

        # 2. State Extraction
        agent_state = sim.get_agent_state()
        pos_xy = np.array(agent_state.position[:2], dtype=np.float32)
        goal_xy = np.array(goal_position[:2], dtype=np.float32)

        # Trivial success case
        goal_dist_3d_fn = getattr(sim, "distance_to_goal_3d", None)
        dist_to_goal = None
        if callable(goal_dist_3d_fn):
            try:
                dist_to_goal = goal_dist_3d_fn()
            except Exception:
                dist_to_goal = None
        if dist_to_goal is None:
            dist_to_goal = float(np.linalg.norm(goal_xy - pos_xy))
        goal_stop_distance = self._get_goal_stop_distance(sim)
        if float(dist_to_goal) <= goal_stop_distance:
            return [ACTION_STOP] * NUM_ACTIONS, "ok_within_1m_geodesic"

        # Calculate current Yaw
        rot = R.from_quat(agent_state.rotation)
        fwd_vec = rot.apply(np.array([1.0, 0.0, 0.0], dtype=np.float32))
        curr_yaw = math.atan2(fwd_vec[1], fwd_vec[0])

        # Habitat-aligned path: use native greedy geodesic follower first
        # (path -> action conversion provided by Habitat/Habitat-Sim).
        habitat_plan_fn = getattr(sim, "plan_greedy_actions", None)
        if callable(habitat_plan_fn):
            try:
                habitat_actions, habitat_reason = habitat_plan_fn(
                    goal_xy=goal_xy,
                    max_steps=NUM_ACTIONS,
                )
            except Exception as e:
                habitat_actions, habitat_reason = None, f"greedy_failed:{e}"

            if habitat_actions is not None and len(habitat_actions) == NUM_ACTIONS:
                final_actions = list(map(int, habitat_actions))
                actions_reason = f"ok_habitat_{habitat_reason}"

                if (not self._strict_gt_prefer_habitat_greedy(sim)) and self._is_turn_oscillation(final_actions):
                    fallback_actions, fallback_reason = self._build_gt_actions6_path_rule(
                        sim=sim,
                        start_xy=pos_xy,
                        curr_yaw=curr_yaw,
                        goal_xy=goal_xy,
                    )
                    if fallback_actions is not None and len(fallback_actions) == NUM_ACTIONS:
                        final_actions = fallback_actions
                        actions_reason = f"ok_habitat_{habitat_reason}+{fallback_reason}"

                rollout_valid, rollout_reason = self._rollout_actions_valid(
                    sim, final_actions, viz_prefix=None
                )
                if rollout_valid:
                    mpc_img = sim.get_occ_map_with_actions(final_actions)
                    self.save_image(mpc_img, "mpc")
                    return final_actions, actions_reason

                fallback_actions, fallback_reason = self._build_gt_actions6_path_rule(
                    sim=sim,
                    start_xy=pos_xy,
                    curr_yaw=curr_yaw,
                    goal_xy=goal_xy,
                )
                if fallback_actions is not None and len(fallback_actions) == NUM_ACTIONS:
                    rv2, rr2 = self._rollout_actions_valid(
                        sim, fallback_actions, viz_prefix=None
                    )
                    if rv2:
                        mpc_img = sim.get_occ_map_with_actions(fallback_actions)
                        self.save_image(mpc_img, "mpc")
                        return fallback_actions, f"ok_habitat_fallback_{fallback_reason}"

                return [ACTION_STOP] * NUM_ACTIONS, f"habitat_invalid:{rollout_reason}"

        # 3. Path Planning (A* -> Dense Pixels)
        try:
            dense_path_px = self._plan_dense_path_pixels(sim, pos_xy, goal_xy)
        except Exception as e:
            return None, f"gt_astar_failed:{e}"

        if not dense_path_px or len(dense_path_px) <= 1:
            return [ACTION_STOP] * NUM_ACTIONS, "ok_no_path_or_already_there"

        # 4. Path Processing (Pixel -> World -> Smooth)
        # Convert pixels to world coordinates
        world_pts = [sim.transform_from_pixel_to_world(px)[:2] for px in dense_path_px]
        pts_array = np.array(world_pts, dtype=np.float64)

        if pts_array.shape[0] < 2:
            return [ACTION_STOP] * NUM_ACTIONS, "ok_path_too_short"

        # Smooth the path for the controller
        pts_smoothed = smooth_polyline(pts_array, mode="vel", win=9, kind="tri")
        poly = Polyline2D(pts_smoothed)

        # 5. MPC Control
        mpc_actions = self._run_mpc_solver(
            sim, poly, float(pos_xy[0]), float(pos_xy[1]), float(curr_yaw)
        )

        if not mpc_actions:
            return [ACTION_STOP] * NUM_ACTIONS, "ok_mpc_empty"

        # 6. Kinematic Simulation (MPC -> Gym Actions)
        final_actions = self._simulate_mpc_kinematics(
            sim, mpc_actions, pos_xy, curr_yaw, goal_xy
        )
        actions_reason = "ok_mpc"

        if self._is_turn_oscillation(final_actions):
            fallback_actions, fallback_reason = self._build_gt_actions6_path_rule(
                sim=sim,
                start_xy=pos_xy,
                curr_yaw=curr_yaw,
                goal_xy=goal_xy,
            )
            if fallback_actions is not None and len(fallback_actions) == NUM_ACTIONS:
                final_actions = fallback_actions
                actions_reason = f"ok_{fallback_reason}"

        # Save MPC rollout visualization whenever oracle actions are built.
        mpc_img = sim.get_occ_map_with_actions(final_actions)
        self.save_image(mpc_img, "mpc")

        return final_actions, actions_reason

    def _save_oracle_debug_images(self, sim, goal_position):
        """Best-effort dump of oracle MPC/A* visualizations for every step."""
        cache = {
            "gt_actions": None,
            "actions_reason": "missing_goal_position",
            "gt_pixel_str": None,
            "pixel_reason": "missing_goal_position",
        }

        if goal_position is None or len(goal_position) < 2:
            return cache

        try:
            gt_actions, actions_reason = self._build_gt_actions6(sim, goal_position)
            cache["gt_actions"] = gt_actions
            cache["actions_reason"] = actions_reason
        except Exception as e:
            cache["actions_reason"] = f"debug_save_failed:{e}"
            print(f"Warning: failed to save MPC debug image: {e}")

        try:
            gt_pixel_str, pixel_reason = self._build_gt_pixel_tokstr_13(
                sim, goal_position
            )
            cache["gt_pixel_str"] = gt_pixel_str
            cache["pixel_reason"] = pixel_reason
        except Exception as e:
            cache["pixel_reason"] = f"debug_save_failed:{e}"
            print(f"Warning: failed to save A* debug image: {e}")

        return cache

    def _run_mpc_solver(self, sim, poly, x0, y0, th0):
        """
        Helper: Encapsulates the massive configuration for the MPC solver.
        """
        return plan_actions_beam_mpc(
            poly=poly,
            x0=x0,
            y0=y0,
            th0=th0,
            lookahead_m=5.0,
            horizon=32,
            beam=300,
            step_m=float(sim.config.FORWARD_STEP_SIZE),
            turn_deg=float(sim.config.TURN_ANGLE),
            goal_stop_m=0.15,
            max_steps=6,
            relocalize=False,
            relocalize_thresh=0.35,
            w_step=1.0,
            w_turn=0.5,
            w_perp=300.0,
            d0=0.03,
            w_head=0.10,
            w_head_tangent=0.10,
            w_switch=10.0,
            w_terminal=120.0,
            w_progress=2.0,
            w_back=20.0,
            w_goal_heur=1.5,
            w_spin=10.0,
            turn_slack=1,
            commit=2,
            stall_steps=20,
            stall_ds_eps=1e-3,
            endgame_dist=0.0,
            endgame_turn_tol_deg=7.5,
            w_stop_good=-80.0,
            w_stop_bad=300.0,
        )

    def _simulate_mpc_kinematics(self, sim, mpc_actions, start_xy, start_yaw, goal_xy):
        """
        Helper: Simulates the execution of MPC actions to generate
        the final discrete action sequence, handling state updates and stop conditions.
        """
        out_actions = []
        x, y, th = float(start_xy[0]), float(start_xy[1]), float(start_yaw)

        step_m = float(sim.config.FORWARD_STEP_SIZE)
        turn_rad = np.deg2rad(float(sim.config.TURN_ANGLE))

        # MPC Action ID -> Gym Action ID Mapping
        # 1: FWD, 2: LEFT, 3: RIGHT
        mpc_to_gym = {
            1: ACTION_MOVE_FORWARD,
            2: ACTION_TURN_LEFT,
            3: ACTION_TURN_RIGHT,
        }

        for raw_a in mpc_actions:
            if len(out_actions) >= NUM_ACTIONS:
                break

            # Check goal condition BEFORE action
            if math.hypot(x - goal_xy[0], y - goal_xy[1]) <= 1.0:
                break

            # Map and Record Action
            gym_action = mpc_to_gym.get(int(raw_a), ACTION_STOP)
            out_actions.append(gym_action)

            # Kinematic Update (Simulate the move)
            if gym_action == ACTION_TURN_LEFT:
                th = wrap_pi(th + turn_rad)
            elif gym_action == ACTION_TURN_RIGHT:
                th = wrap_pi(th - turn_rad)
            elif gym_action == ACTION_MOVE_FORWARD:
                x += step_m * math.cos(th)
                y += step_m * math.sin(th)

            # Check goal condition AFTER action
            if math.hypot(x - goal_xy[0], y - goal_xy[1]) <= 1.0:
                break

        # Pad with STOP if sequence is shorter than NUM_ACTIONS
        if len(out_actions) < NUM_ACTIONS:
            out_actions.extend([ACTION_STOP] * (NUM_ACTIONS - len(out_actions)))

        return out_actions[:NUM_ACTIONS]

    def _is_turn_oscillation(self, actions):
        if not actions or len(actions) != NUM_ACTIONS:
            return False
        prefix = list(map(int, actions[:4]))
        if ACTION_MOVE_FORWARD in prefix:
            return False
        if not all(
            a in {ACTION_TURN_LEFT, ACTION_TURN_RIGHT, ACTION_STOP} for a in prefix
        ):
            return False
        turn_count = sum(
            1 for a in prefix if a in {ACTION_TURN_LEFT, ACTION_TURN_RIGHT}
        )
        if turn_count >= 3:
            return True
        turn_only = [a for a in prefix if a in {ACTION_TURN_LEFT, ACTION_TURN_RIGHT}]
        if len(turn_only) < 2:
            return False
        if (
            turn_only[0] == ACTION_TURN_LEFT and turn_only[1] == ACTION_TURN_RIGHT
        ) or (
            turn_only[0] == ACTION_TURN_RIGHT and turn_only[1] == ACTION_TURN_LEFT
        ):
            return True
        return (
            ACTION_TURN_LEFT in turn_only
            and ACTION_TURN_RIGHT in turn_only
            and len(turn_only) >= 3
        )

    def _build_gt_actions6_path_rule(self, sim, start_xy, curr_yaw, goal_xy):
        try:
            dense_path_px = self._plan_dense_path_pixels(sim, start_xy, goal_xy)
        except Exception as e:
            return None, f"rule_astar_failed:{e}"

        if not dense_path_px:
            return [ACTION_STOP] * NUM_ACTIONS, "rule_no_path"

        path_xy = [
            np.array(sim.transform_from_pixel_to_world(px)[:2], dtype=np.float32)
            for px in dense_path_px
        ]
        if len(path_xy) < 2:
            return [ACTION_STOP] * NUM_ACTIONS, "rule_short_path"

        x, y = float(start_xy[0]), float(start_xy[1])
        th = float(curr_yaw)
        step_m = float(sim.config.FORWARD_STEP_SIZE)
        turn_rad = np.deg2rad(float(sim.config.TURN_ANGLE))
        grid = self._get_collision_grid(sim)
        map_w = int(sim.map_width)
        map_h = int(sim.map_height)
        cursor = 0
        out = []

        for _ in range(NUM_ACTIONS):
            if math.hypot(x - goal_xy[0], y - goal_xy[1]) <= 1.0:
                out.append(ACTION_STOP)
                continue

            curr_xy = np.array([x, y], dtype=np.float32)
            while cursor + 1 < len(path_xy):
                if float(np.linalg.norm(path_xy[cursor] - curr_xy)) <= 0.35:
                    cursor += 1
                else:
                    break

            target_idx = min(cursor + 4, len(path_xy) - 1)
            tx, ty = float(path_xy[target_idx][0]), float(path_xy[target_idx][1])
            desired = math.atan2(ty - y, tx - x)
            diff = wrap_pi(desired - th)

            if abs(diff) > (turn_rad * 0.6):
                if diff > 0:
                    out.append(ACTION_TURN_LEFT)
                    th = wrap_pi(th + turn_rad)
                else:
                    out.append(ACTION_TURN_RIGHT)
                    th = wrap_pi(th - turn_rad)
                continue

            nx = x + step_m * math.cos(th)
            ny = y + step_m * math.sin(th)
            if grid is not None:
                px_start = sim.transform_from_world_to_pixel(
                    np.array([x, y, 0.0], dtype=np.float32)
                )
                px_end = sim.transform_from_world_to_pixel(
                    np.array([nx, ny, 0.0], dtype=np.float32)
                )
                if not self._segment_is_clear(
                    grid, map_w, map_h, px_start, px_end, sim
                ):
                    if diff >= 0:
                        out.append(ACTION_TURN_LEFT)
                        th = wrap_pi(th + turn_rad)
                    else:
                        out.append(ACTION_TURN_RIGHT)
                        th = wrap_pi(th - turn_rad)
                    continue

            out.append(ACTION_MOVE_FORWARD)
            x, y = nx, ny

        if ACTION_MOVE_FORWARD not in out:
            fx = float(start_xy[0]) + step_m * math.cos(float(curr_yaw))
            fy = float(start_xy[1]) + step_m * math.sin(float(curr_yaw))
            if grid is not None:
                px_start = sim.transform_from_world_to_pixel(
                    np.array([float(start_xy[0]), float(start_xy[1]), 0.0], dtype=np.float32)
                )
                px_end = sim.transform_from_world_to_pixel(
                    np.array([fx, fy, 0.0], dtype=np.float32)
                )
                if self._segment_is_clear(grid, map_w, map_h, px_start, px_end, sim):
                    out[0] = ACTION_MOVE_FORWARD

        if len(out) < NUM_ACTIONS:
            out.extend([ACTION_STOP] * (NUM_ACTIONS - len(out)))
        return out[:NUM_ACTIONS], "rule_path_follow"

    def _plan_dense_path_pixels(self, sim, start_world_xy, goal_world_xy):
        start_world = np.array(
            [start_world_xy[0], start_world_xy[1], 0.0], dtype=np.float32
        )
        goal_world = np.array(
            [goal_world_xy[0], goal_world_xy[1], 0.0], dtype=np.float32
        )
        start_px = sim.transform_from_world_to_pixel(start_world)
        goal_px = sim.transform_from_world_to_pixel(goal_world)
        raw_path = sim._astar_with_fallback_margin(start_px, goal_px)
        smooth_path = sim._smooth_path(raw_path)
        dense_path = sim._densify_path(smooth_path)
        return dense_path

    @staticmethod
    def _round_half_up_float(x):
        if x >= 0:
            return int(np.floor(x + 0.5))
        return -int(np.floor(-x + 0.5))

    def _norm_xy_from_pixel(self, x, y, occ_w, occ_h):
        xx = max(0, min(int(x), int(occ_w) - 1))
        yy = max(0, min(int(y), int(occ_h) - 1))
        xn = self._round_half_up_float((xx / float(occ_w)) * 1000.0)
        yn = self._round_half_up_float((yy / float(occ_h)) * 1000.0)
        xn = max(0, min(xn, 999))
        yn = max(0, min(yn, 999))
        return xn, yn

    @staticmethod
    def _is_action_wall_reason(reason):
        return bool(
            re.match(r"^(?:action|habitat)_rollout_hit_wall_step_(\d+)$", str(reason))
        )

    @staticmethod
    def _is_pixel_wall_reason(reason):
        return str(reason).startswith("pixel_branch_hit_wall_seg_")

    def _build_gt_stop_pair(self, sim):
        gt_output = {"vlnce": self._action_xml_from_ids([ACTION_STOP] * NUM_ACTIONS)}
        return gt_output
