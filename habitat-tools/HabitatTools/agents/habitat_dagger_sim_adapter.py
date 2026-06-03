from __future__ import annotations

import heapq
import json
import math
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Optional, Sequence

import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R

try:
    from habitat.sims.habitat_simulator.actions import HabitatSimActions
except Exception:  # pragma: no cover
    HabitatSimActions = None  # type: ignore

try:
    from habitat.utils.visualizations import maps as habitat_maps
except Exception:  # pragma: no cover
    habitat_maps = None  # type: ignore


def _reconstruct_passable_grid(
    rows: list[list[int]], threshold: int
) -> list[list[bool]]:
    return [[value >= threshold for value in row] for row in rows]


def _inflate_obstacles(grid: list[list[bool]], margin: int) -> list[list[bool]]:
    if margin <= 0:
        return [row[:] for row in grid]

    height = len(grid)
    width = len(grid[0]) if height else 0
    inflated = [row[:] for row in grid]
    blocked_cells = [
        (x, y) for y in range(height) for x in range(width) if not grid[y][x]
    ]
    radius_sq = margin * margin

    for bx, by in blocked_cells:
        y0 = max(by - margin, 0)
        y1 = min(by + margin, height - 1)
        for y in range(y0, y1 + 1):
            dy = y - by
            x0 = max(bx - margin, 0)
            x1 = min(bx + margin, width - 1)
            for x in range(x0, x1 + 1):
                dx = x - bx
                if dx * dx + dy * dy <= radius_sq:
                    inflated[y][x] = False

    return inflated


@dataclass
class _AgentState:
    position: np.ndarray
    rotation: np.ndarray


class HabitatDaggerSimAdapter:
    """Adapter exposing BAE planner sim APIs on top of Habitat env/sim."""

    def __init__(
        self,
        env: Any,
        occupancy_root: str,
        threshold: int = 200,
        margins: Sequence[int] = (7, 6, 5, 4),
        skip_supervision_when_in_obstacle: bool = False,
        gt_strict_coverage: bool = False,
        strict_gt_dataset_goal_only: bool = True,
        use_habitat_topdown_map: bool = True,
        topdown_map_resolution: int = 1024,
        topdown_floor_switch_threshold: float = 0.75,
    ) -> None:
        self.env = env
        self.sim = getattr(env, "sim", None)
        self.occupancy_root = Path(occupancy_root)
        self.threshold = int(threshold)
        self.margins = [int(m) for m in margins]
        self.use_habitat_topdown_map = bool(use_habitat_topdown_map)
        self.topdown_map_resolution = int(topdown_map_resolution)
        self.topdown_floor_switch_threshold = float(topdown_floor_switch_threshold)

        step_size = 0.25
        turn_angle = 15.0
        greedy_goal_radius = 1.5
        env_cfg = getattr(env, "_config", None) or getattr(env, "config", None)
        sim_cfg = getattr(env_cfg, "SIMULATOR", None)
        if sim_cfg is not None:
            step_size = float(getattr(sim_cfg, "FORWARD_STEP_SIZE", step_size))
            turn_angle = float(getattr(sim_cfg, "TURN_ANGLE", turn_angle))
        self.config = SimpleNamespace(
            SKIP_SUPERVISION_WHEN_IN_OBSTACLE=bool(skip_supervision_when_in_obstacle),
            GT_STRICT_COVERAGE=bool(gt_strict_coverage),
            STRICT_GT_DATASET_GOAL_ONLY=bool(strict_gt_dataset_goal_only),
            STRICT_GT_PREFER_HABITAT_GREEDY=True,
            GREEDY_GOAL_RADIUS=greedy_goal_radius,
            FORWARD_STEP_SIZE=step_size,
            TURN_ANGLE=turn_angle,
        )

        self.map_width = 0
        self.map_height = 0
        self.occupancy_rows: list[list[int]] = []
        self.passable_grid: Optional[list[list[bool]]] = None
        self.safe_passable_grids: dict[int, list[list[bool]]] = {}
        self.meta: dict[str, Any] = {}
        self.path_length: float = 0.0

        self.scene_id = ""
        self.scene_dir: Optional[Path] = None
        self.goal_position_3d: Optional[np.ndarray] = None
        self.occ_map_original: Optional[np.ndarray] = None
        self.occ_map_with_trajectory: Optional[np.ndarray] = None
        self.bev_map_original: Optional[np.ndarray] = None
        self.bev_map_with_trajectory: Optional[np.ndarray] = None
        self.bev_path: Optional[str] = None

        self.start_pixel: Optional[tuple[int, int]] = None
        self.trajectory_pixels: list[tuple[int, int]] = []
        self.trajectory_world_xz: list[tuple[float, float]] = []
        self.last_astar_debug: dict[str, Any] = {}
        self._greedy_follower = None
        self._greedy_scene = None
        self._use_habitat_map_coords = False
        self.agent_state = _AgentState(
            position=np.zeros(3, dtype=np.float32),
            rotation=np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
        )
        self.habitat_agent_state = _AgentState(
            position=np.zeros(3, dtype=np.float32),
            rotation=np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
        )

    def _scene_occ_paths(self) -> tuple[Optional[Path], Optional[Path]]:
        if self.scene_dir is None:
            return None, None
        occ_json = self.scene_dir / "occupancy.json"
        occ_png = self.scene_dir / "occupancy.png"
        return (
            occ_json if occ_json.exists() else None,
            occ_png if occ_png.exists() else None,
        )

    def _build_coord_transform_meta(self) -> dict[str, Any]:
        span_x = float(self.meta["upper"][0] - self.meta["lower"][0])
        span_y = float(self.meta["upper"][1] - self.meta["lower"][1])
        scale_x = span_x / max(int(self.map_width), 1)
        scale_y = span_y / max(int(self.map_height), 1)

        payload: dict[str, Any] = {
            "plane": "x-z",
            "mode": (
                "habitat_to_grid"
                if self._use_habitat_map_coords
                else "interiorgs_formula"
            ),
            "use_habitat_map_coords": bool(self._use_habitat_map_coords),
            "map_width": int(self.map_width),
            "map_height": int(self.map_height),
            "scale_x": float(scale_x),
            "scale_y": float(scale_y),
            "world_to_pixel_formula": (
                "if habitat_to_grid: "
                "row,col = habitat_maps.to_grid(realworld_x=world_z, realworld_y=world_x); "
                "pixel=(col,row); "
                "else: px=(-world_x-lower_x)/scale_x-0.5; py=(upper_y+world_z)/scale_y-0.5"
            ),
            "pixel_to_world_formula": (
                "if habitat_to_grid: "
                "world_z,world_x = habitat_maps.from_grid(grid_x=py, grid_y=px); "
                "world=(world_x,world_z); "
                "else: flipped_x=lower_x+(px+0.5)*scale_x; "
                "flipped_y=upper_y-(py+0.5)*scale_y; world=(-flipped_x,-flipped_y)"
            ),
        }

        if self._use_habitat_map_coords and self.sim is not None:
            pathfinder = getattr(self.sim, "pathfinder", None)
            if pathfinder is not None:
                try:
                    lower_bound, upper_bound = pathfinder.get_bounds()
                    payload["pathfinder_bounds"] = {
                        "lower": [float(x) for x in lower_bound],
                        "upper": [float(x) for x in upper_bound],
                    }
                except Exception:
                    pass
        return payload

    def _decorate_meta(self, occ_source: str) -> None:
        occ_json, occ_png = self._scene_occ_paths()
        self.meta["occ"] = {
            "occ_source": str(occ_source),
            "occupancy_root": str(self.occupancy_root),
            "scene_dir": str(self.scene_dir) if self.scene_dir is not None else "",
            "occupancy_json_path": str(occ_json) if occ_json is not None else "",
            "occupancy_png_path": str(occ_png) if occ_png is not None else "",
            # Filled by bae_agent_dagger.reset with per-episode output path.
            "occ_image_root": "",
        }
        self.meta["astar_margins_effective"] = [int(m) for m in self.margins]
        self.meta["coord_transform"] = self._build_coord_transform_meta()

    @staticmethod
    def _extract_goal_position_3d(episode: Any) -> Optional[np.ndarray]:
        goals = getattr(episode, "goals", None)
        if not isinstance(goals, list) or not goals:
            return None
        goal0 = goals[0]
        if isinstance(goal0, dict):
            pos = goal0.get("position")
        else:
            pos = getattr(goal0, "position", None)
        if pos is None or len(pos) < 3:
            return None
        try:
            return np.array([float(pos[0]), float(pos[1]), float(pos[2])], dtype=np.float32)
        except (TypeError, ValueError):
            return None

    def reset_episode(self, scene_id: str, episode: Any) -> None:
        self.scene_id = str(scene_id)
        self.scene_dir = self.occupancy_root / self.scene_id
        self.goal_position_3d = self._extract_goal_position_3d(episode)
        self._use_habitat_map_coords = False

        # Prefer Habitat-native topdown map and coordinate transforms so the
        # OCC grid is guaranteed to align with the active simulator frame.
        if self.use_habitat_topdown_map and self._load_habitat_topdown_map(episode):
            self.trajectory_world_xz = []
            self.trajectory_pixels = []
            self.start_pixel = None
            self._greedy_follower = None
            self._greedy_scene = None
            self._sync_state_and_trajectory()
            return

        if not self.scene_dir.exists():
            raise FileNotFoundError(
                f"occupancy scene directory not found: {self.scene_dir}"
            )

        occ_json = self.scene_dir / "occupancy.json"
        occ_png = self.scene_dir / "occupancy.png"
        if not occ_json.exists() or not occ_png.exists():
            raise FileNotFoundError(
                f"missing occupancy assets in {self.scene_dir} "
                f"(need occupancy.json and occupancy.png)"
            )

        payload = json.loads(occ_json.read_text(encoding="utf-8"))
        scale = float(payload.get("scale"))
        min_x, _, min_z = map(float, payload.get("min"))
        lower = list(map(float, payload.get("lower")))
        upper = list(map(float, payload.get("upper")))
        max_y = float(payload.get("max", [0.0, 0.0, 0.0])[1])

        occ_gray = cv2.imread(str(occ_png), cv2.IMREAD_GRAYSCALE)
        if occ_gray is None:
            raise RuntimeError(f"failed to read occupancy png: {occ_png}")

        self.map_height, self.map_width = int(occ_gray.shape[0]), int(occ_gray.shape[1])
        self.occupancy_rows = occ_gray.tolist()
        self.passable_grid = _reconstruct_passable_grid(self.occupancy_rows, self.threshold)
        self.safe_passable_grids = {
            int(m): _inflate_obstacles(self.passable_grid, int(m)) for m in self.margins
        }

        self.meta = {
            "width": self.map_width,
            "height": self.map_height,
            "scale": scale,
            "left": min_x,
            "right": min_x + self.map_width * scale,
            "top": max_y,
            "bottom": max_y - self.map_height * scale,
            "lower_z": float(min_z),
            "lower": lower,
            "upper": upper,
            "source": "occupancy_asset_png",
        }
        self._decorate_meta("occupancy_asset_png")

        self.occ_map_original = cv2.cvtColor(occ_gray, cv2.COLOR_GRAY2RGB)
        self.occ_map_with_trajectory = self.occ_map_original.copy()

        bev_png = self.scene_dir / "bev_map.png"
        if bev_png.exists():
            bev = cv2.imread(str(bev_png))
            if bev is not None:
                bev = cv2.cvtColor(bev, cv2.COLOR_BGR2RGB)
                self.bev_map_original = bev
            else:
                self.bev_map_original = self.occ_map_original.copy()
            self.bev_path = str(bev_png)
        else:
            self.bev_map_original = self.occ_map_original.copy()
            self.bev_path = str(occ_png)
        self.bev_map_with_trajectory = self.bev_map_original.copy()

        info = getattr(episode, "info", None)
        geodesic = None
        if isinstance(info, dict):
            geodesic = info.get("geodesic_distance")
        if geodesic is None and hasattr(info, "get"):
            geodesic = info.get("geodesic_distance")
        try:
            self.path_length = float(geodesic) if geodesic is not None else 0.0
        except (TypeError, ValueError):
            self.path_length = 0.0

        self.trajectory_world_xz = []
        self.trajectory_pixels = []
        self.start_pixel = None
        self._greedy_follower = None
        self._greedy_scene = None
        self._sync_state_and_trajectory()

    def _apply_habitat_topdown_map(
        self, top_down: np.ndarray, pathfinder: Any, height: float, episode: Any = None
    ) -> bool:
        if top_down is None or top_down.size == 0:
            return False
        top_down = np.ascontiguousarray(top_down)
        # get_topdown_map returns 0 (occupied) / 1 (unoccupied) / 2 (border if enabled).
        # We disable borders above; treat non-zero as passable for robustness.
        occ_gray = np.where(top_down > 0, 255, 0).astype(np.uint8)

        self.map_height, self.map_width = int(occ_gray.shape[0]), int(occ_gray.shape[1])
        self.occupancy_rows = occ_gray.tolist()
        self.passable_grid = _reconstruct_passable_grid(self.occupancy_rows, self.threshold)
        self.safe_passable_grids = {
            int(m): _inflate_obstacles(self.passable_grid, int(m)) for m in self.margins
        }

        lower_bound, upper_bound = pathfinder.get_bounds()
        x_min, x_max = float(lower_bound[0]), float(upper_bound[0])
        z_min, z_max = float(lower_bound[2]), float(upper_bound[2])
        y_min, y_max = float(lower_bound[1]), float(upper_bound[1])

        # Keep metadata fields compatible with legacy logging consumers.
        self.meta = {
            "width": self.map_width,
            "height": self.map_height,
            "scale": float(
                max(z_max - z_min, x_max - x_min) / max(self.map_height, self.map_width, 1)
            ),
            "left": x_min,
            "right": x_max,
            "top": z_max,
            "bottom": z_min,
            "lower_z": y_min,
            "lower": [x_max, z_min, y_min],
            "upper": [x_min, z_max, y_max],
            "source": "habitat_pathfinder_topdown_runtime",
            "selected_height": float(height),
        }
        self._use_habitat_map_coords = True
        self._decorate_meta("habitat_pathfinder_topdown_runtime")

        self.occ_map_original = cv2.cvtColor(occ_gray, cv2.COLOR_GRAY2RGB)
        self.occ_map_with_trajectory = self.occ_map_original.copy()
        self.bev_map_original = self.occ_map_original.copy()
        self.bev_map_with_trajectory = self.bev_map_original.copy()
        self.bev_path = ""

        info = getattr(episode, "info", None) if episode is not None else None
        geodesic = None
        if isinstance(info, dict):
            geodesic = info.get("geodesic_distance")
        if geodesic is None and hasattr(info, "get"):
            geodesic = info.get("geodesic_distance")
        try:
            self.path_length = float(geodesic) if geodesic is not None else self.path_length
        except (TypeError, ValueError):
            pass

        return True

    def _refresh_habitat_topdown_map(self, height: float, episode: Any = None) -> bool:
        if self.sim is None or habitat_maps is None:
            return False
        pathfinder = getattr(self.sim, "pathfinder", None)
        if pathfinder is None:
            return False
        try:
            top_down = habitat_maps.get_topdown_map(
                pathfinder=pathfinder,
                height=height,
                map_resolution=max(int(self.topdown_map_resolution), 64),
                draw_border=False,
            )
        except Exception:
            return False
        return self._apply_habitat_topdown_map(top_down, pathfinder, height, episode=episode)

    def _load_habitat_topdown_map(self, episode: Any) -> bool:
        if self.sim is None or habitat_maps is None:
            return False
        start_position = getattr(episode, "start_position", None)
        if isinstance(start_position, (list, tuple)) and len(start_position) >= 2:
            height = float(start_position[1])
        else:
            try:
                height = float(self.sim.get_agent_state().position[1])
            except Exception:
                return False
        return self._refresh_habitat_topdown_map(height=height, episode=episode)

    def _reproject_trajectory_pixels(self) -> None:
        self.trajectory_pixels = []
        self.start_pixel = None
        for idx, (wx, wz) in enumerate(self.trajectory_world_xz):
            px = self.transform_from_world_to_pixel(
                np.array([float(wx), float(wz), 0.0], dtype=np.float32)
            )
            if idx == 0:
                self.start_pixel = px
            if not self.trajectory_pixels or self.trajectory_pixels[-1] != px:
                self.trajectory_pixels.append(px)

    def _maybe_refresh_topdown_map_by_height(self, current_height: float) -> None:
        if not self._use_habitat_map_coords or not self.use_habitat_topdown_map:
            return
        prev_h = self.meta.get("selected_height")
        if prev_h is None:
            return
        try:
            prev_h = float(prev_h)
        except (TypeError, ValueError):
            return
        if abs(float(current_height) - prev_h) < self.topdown_floor_switch_threshold:
            return
        if self._refresh_habitat_topdown_map(height=float(current_height), episode=None):
            self._reproject_trajectory_pixels()
            print(
                f"[HabitatDaggerSimAdapter] floor switch: topdown height "
                f"{prev_h:.3f} -> {float(current_height):.3f}"
            )

    def _quat_to_xyzw(self, quat: Any) -> np.ndarray:
        if isinstance(quat, np.ndarray):
            arr = quat.astype(np.float32).reshape(-1)
            if arr.shape[0] == 4:
                return arr

        if hasattr(quat, "x") and hasattr(quat, "y") and hasattr(quat, "z") and hasattr(quat, "w"):
            return np.array([quat.x, quat.y, quat.z, quat.w], dtype=np.float32)

        if hasattr(quat, "imag") and hasattr(quat, "real"):
            imag = np.asarray(quat.imag, dtype=np.float32).reshape(-1)
            if imag.shape[0] == 3:
                return np.array([imag[0], imag[1], imag[2], float(quat.real)], dtype=np.float32)

        arr = np.asarray(quat, dtype=np.float32).reshape(-1)
        if arr.shape[0] == 4:
            return arr
        return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)

    def _sync_state_and_trajectory(self) -> None:
        if self.sim is None:
            return
        s = self.sim.get_agent_state()
        if self.use_habitat_topdown_map and self._use_habitat_map_coords:
            self._maybe_refresh_topdown_map_by_height(float(s.position[1]))
        pos_3d = np.array(s.position, dtype=np.float32)
        rot_3d = self._quat_to_xyzw(s.rotation)
        self.habitat_agent_state = _AgentState(position=pos_3d.copy(), rotation=rot_3d.copy())
        rot_obj = R.from_quat(rot_3d)
        # Habitat planar navigation is in X-Z. Convert to a pseudo X-Y plane
        # so legacy DAgger logic (which assumes X-Y) can be reused unchanged.
        planar_pos = np.array([pos_3d[0], pos_3d[2], 0.0], dtype=np.float32)
        fwd_world = rot_obj.apply(np.array([1.0, 0.0, 0.0], dtype=np.float32))
        yaw_xz = float(math.atan2(float(fwd_world[2]), float(fwd_world[0])))
        planar_rot = R.from_euler("z", yaw_xz, degrees=False).as_quat().astype(np.float32)
        self.agent_state = _AgentState(position=planar_pos, rotation=planar_rot)
        curr_world_xz = (float(pos_3d[0]), float(pos_3d[2]))
        if not self.trajectory_world_xz:
            self.trajectory_world_xz = [curr_world_xz]
        else:
            prev_wx, prev_wz = self.trajectory_world_xz[-1]
            if abs(curr_world_xz[0] - prev_wx) > 1e-4 or abs(curr_world_xz[1] - prev_wz) > 1e-4:
                self.trajectory_world_xz.append(curr_world_xz)

        px = self.transform_from_world_to_pixel(
            np.array([planar_pos[0], planar_pos[1], 0.0], dtype=np.float32)
        )
        if self.start_pixel is None:
            self.start_pixel = px
            self.trajectory_pixels = [px]
            return
        if not self.trajectory_pixels:
            self.trajectory_pixels = [self.start_pixel]
        if self.trajectory_pixels[-1] != px:
            self.trajectory_pixels.append(px)

    def get_agent_state(self) -> _AgentState:
        self._sync_state_and_trajectory()
        return self.agent_state

    def get_habitat_agent_state(self) -> _AgentState:
        self._sync_state_and_trajectory()
        return self.habitat_agent_state

    def get_current_pixel_position(self) -> Optional[tuple[int, int]]:
        self._sync_state_and_trajectory()
        if not self.trajectory_pixels:
            return None
        return self.trajectory_pixels[-1]

    def _draw_trajectory(self, canvas: np.ndarray) -> np.ndarray:
        if self.start_pixel is not None:
            cv2.circle(canvas, self.start_pixel, 3, (0, 255, 0), -1)
        if len(self.trajectory_pixels) >= 2:
            for a, b in zip(self.trajectory_pixels, self.trajectory_pixels[1:]):
                cv2.line(canvas, a, b, (0, 0, 0), 2)
        if self.trajectory_pixels:
            cv2.circle(canvas, self.trajectory_pixels[-1], 3, (255, 0, 0), -1)
        return canvas

    def get_occ_map(self) -> Optional[np.ndarray]:
        self._sync_state_and_trajectory()
        if self.occ_map_original is None:
            return None
        return self.occ_map_original.copy()

    def get_occ_map_with_trajectory(self) -> Optional[np.ndarray]:
        self._sync_state_and_trajectory()
        if self.occ_map_original is None:
            return None
        return self._draw_trajectory(self.occ_map_original.copy())

    def get_bev_map_with_trajectory(self) -> Optional[np.ndarray]:
        self._sync_state_and_trajectory()
        if self.bev_map_original is None:
            return None
        return self._draw_trajectory(self.bev_map_original.copy())

    def get_occ_map_with_actions(
        self,
        actions: list[int],
        traj_color: tuple[int, int, int] = (0, 0, 255),
        pos_color: tuple[int, int, int] = (255, 0, 0),
        trace: Optional[list[dict[str, Any]]] = None,
    ) -> Optional[np.ndarray]:
        base = self.get_occ_map_with_trajectory()
        if base is None:
            return None

        if trace is None:
            _, _, trace = self.rollout_actions_habitat_trace(actions)

        cur = self.get_current_pixel_position()
        if cur is not None:
            cv2.circle(base, cur, 10, pos_color, -1)

        arrow_len = float(self.config.FORWARD_STEP_SIZE) * 1.2
        for idx, step in enumerate(trace or []):
            try:
                aid = int(step.get("action", -1))
                start_px = tuple(map(int, step.get("start_pixel", [0, 0])))
                end_px = tuple(map(int, step.get("end_pixel", [0, 0])))
                collided = bool(step.get("collided", False))
                yaw = float(step.get("end_yaw", 0.0))
                end_pos = step.get("end_pos", [0.0, 0.0])
                end_x = float(end_pos[0])
                end_z = float(end_pos[1])
            except Exception:
                continue

            if aid == 1:
                line_color = (255, 80, 80) if collided else traj_color
                cv2.line(base, start_px, end_px, line_color, 3)
            elif aid in (2, 3):
                tip_world = np.array(
                    [
                        end_x + arrow_len * math.cos(yaw),
                        end_z + arrow_len * math.sin(yaw),
                        0.0,
                    ],
                    dtype=np.float32,
                )
                tip_px = self.transform_from_world_to_pixel(tip_world)
                cv2.arrowedLine(
                    base,
                    end_px,
                    tip_px,
                    (255, 255, 0),
                    2,
                    tipLength=0.35,
                )
            cv2.putText(
                base,
                str(idx),
                (end_px[0] + 2, end_px[1] - 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.35,
                (0, 0, 0),
                1,
                cv2.LINE_AA,
            )
        return base

    def get_occ_map_with_pixels(
        self,
        pixels: list[tuple[int, int]],
        path_color: tuple[int, int, int] = (255, 255, 0),
        pos_color: tuple[int, int, int] = (255, 0, 0),
    ) -> Optional[np.ndarray]:
        base = self.get_occ_map_with_trajectory()
        if base is None:
            return None

        cur = self.get_current_pixel_position()
        if cur is not None:
            cv2.circle(base, cur, 10, pos_color, -1)

        prev = cur
        for px in pixels or []:
            pt = (int(px[0]), int(px[1]))
            if prev is not None:
                cv2.line(base, prev, pt, path_color, 3)
            prev = pt
        return base

    def transform_from_world_to_pixel(self, position: np.ndarray) -> tuple[int, int]:
        if self._use_habitat_map_coords and habitat_maps is not None and self.sim is not None:
            # Legacy planner passes pseudo-world (x, z) in (position[0], position[1]).
            world_x = float(position[0])
            world_z = float(position[1])
            row, col = habitat_maps.to_grid(
                realworld_x=world_z,
                realworld_y=world_x,
                grid_resolution=(int(self.map_height), int(self.map_width)),
                pathfinder=self.sim.pathfinder,
            )
            px_int = int(round(col))
            py_int = int(round(row))
            return (
                max(0, min(px_int, self.map_width - 1)),
                max(0, min(py_int, self.map_height - 1)),
            )

        wx, wy = float(position[0]), float(position[1])
        span_x = float(self.meta["upper"][0] - self.meta["lower"][0])
        span_y = float(self.meta["upper"][1] - self.meta["lower"][1])
        scale_x = span_x / max(self.map_width, 1)
        scale_y = span_y / max(self.map_height, 1)

        px = (-wx - float(self.meta["lower"][0])) / scale_x - 0.5
        py = (float(self.meta["upper"][1]) + wy) / scale_y - 0.5

        px_int = int(round(px))
        py_int = int(round(py))
        return (
            max(0, min(px_int, self.map_width - 1)),
            max(0, min(py_int, self.map_height - 1)),
        )

    def transform_from_pixel_to_world(self, pixel: tuple[int, int]) -> tuple[float, float]:
        if self._use_habitat_map_coords and habitat_maps is not None and self.sim is not None:
            px, py = int(pixel[0]), int(pixel[1])
            world_z, world_x = habitat_maps.from_grid(
                grid_x=py,
                grid_y=px,
                grid_resolution=(int(self.map_height), int(self.map_width)),
                pathfinder=self.sim.pathfinder,
            )
            return float(world_x), float(world_z)

        px, py = int(pixel[0]), int(pixel[1])
        span_x = float(self.meta["upper"][0] - self.meta["lower"][0])
        span_y = float(self.meta["upper"][1] - self.meta["lower"][1])
        scale_x = span_x / max(self.map_width, 1)
        scale_y = span_y / max(self.map_height, 1)

        flipped_x = float(self.meta["lower"][0]) + (px + 0.5) * scale_x
        flipped_y = float(self.meta["upper"][1]) - (py + 0.5) * scale_y
        return float(-flipped_x), float(-flipped_y)

    def _astar(
        self,
        start: tuple[int, int],
        goal: tuple[int, int],
        grid: Optional[list[list[bool]]] = None,
    ) -> Optional[list[tuple[int, int]]]:
        if grid is None:
            grid = self.passable_grid
        if grid is None:
            return None

        height = len(grid)
        width = len(grid[0]) if height else 0

        def heuristic(a: tuple[int, int], b: tuple[int, int]) -> float:
            return math.hypot(a[0] - b[0], a[1] - b[1])

        neighbor_offsets = [
            (1, 0),
            (-1, 0),
            (0, 1),
            (0, -1),
            (1, 1),
            (1, -1),
            (-1, 1),
            (-1, -1),
        ]

        open_heap: list[tuple[float, tuple[int, int]]] = []
        heapq.heappush(open_heap, (heuristic(start, goal), start))
        came_from: dict[tuple[int, int], tuple[int, int]] = {}
        g_score: dict[tuple[int, int], float] = {start: 0.0}

        while open_heap:
            _, current = heapq.heappop(open_heap)
            if current == goal:
                path = [current]
                while current in came_from:
                    current = came_from[current]
                    path.append(current)
                path.reverse()
                return path

            cx, cy = current
            for dx, dy in neighbor_offsets:
                nx, ny = cx + dx, cy + dy
                if not (0 <= nx < width and 0 <= ny < height):
                    continue
                if not grid[ny][nx]:
                    continue
                if dx != 0 and dy != 0:
                    if not grid[cy][nx] or not grid[ny][cx]:
                        continue
                step_cost = math.hypot(dx, dy)
                tentative_g = g_score[current] + step_cost
                neighbor = (nx, ny)
                if tentative_g >= g_score.get(neighbor, float("inf")):
                    continue
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g
                f_score = tentative_g + heuristic(neighbor, goal)
                heapq.heappush(open_heap, (f_score, neighbor))
        return None

    def _astar_with_fallback_margin(
        self, start: tuple[int, int], goal: tuple[int, int]
    ) -> list[tuple[int, int]]:
        if self.passable_grid is None:
            raise RuntimeError("passable_grid is not initialized")

        self.last_astar_debug = {
            "start_input": [int(start[0]), int(start[1])],
            "goal_input": [int(goal[0]), int(goal[1])],
            "used_margin": None,
            "start_adjusted": None,
            "goal_adjusted": None,
            "start_snapped": False,
            "goal_snapped": False,
            "path_found": False,
            "fallback": None,
        }

        margins = [int(m) for m in self.margins] if self.margins else [0]
        tried: list[int] = []
        fallback_start: Optional[tuple[int, int]] = None
        for margin in margins:
            grid = self.safe_passable_grids.get(margin)
            if grid is None:
                continue
            start_adj = self._snap_to_nearest_passable(start, grid, max_radius=32)
            goal_adj = self._snap_to_nearest_passable(goal, grid, max_radius=32)
            if start_adj is None or goal_adj is None:
                tried.append(margin)
                continue
            self.last_astar_debug["used_margin"] = int(margin)
            self.last_astar_debug["start_adjusted"] = [int(start_adj[0]), int(start_adj[1])]
            self.last_astar_debug["goal_adjusted"] = [int(goal_adj[0]), int(goal_adj[1])]
            self.last_astar_debug["start_snapped"] = tuple(start_adj) != tuple(start)
            self.last_astar_debug["goal_snapped"] = tuple(goal_adj) != tuple(goal)
            if fallback_start is None:
                fallback_start = start_adj
            tried.append(margin)
            path = self._astar(start_adj, goal_adj, grid)
            if path is not None:
                self.last_astar_debug["path_found"] = True
                self.last_astar_debug["fallback"] = "margin_path"
                return path
        # Keep DAgger supervision dense even when goal is unreachable on OCC:
        # return a degenerate path at current location instead of throwing.
        if fallback_start is not None:
            self.last_astar_debug["path_found"] = False
            self.last_astar_debug["fallback"] = "degenerate_start"
            self.last_astar_debug["start_adjusted"] = [
                int(fallback_start[0]),
                int(fallback_start[1]),
            ]
            self.last_astar_debug["start_snapped"] = tuple(fallback_start) != tuple(start)
            return [fallback_start]
        self.last_astar_debug["path_found"] = False
        self.last_astar_debug["fallback"] = "degenerate_input_start"
        return [start]

    def _snap_to_nearest_passable(
        self,
        point: tuple[int, int],
        grid: list[list[bool]],
        max_radius: int = 32,
    ) -> Optional[tuple[int, int]]:
        h = len(grid)
        w = len(grid[0]) if h else 0
        px, py = int(point[0]), int(point[1])
        if not (0 <= px < w and 0 <= py < h):
            return None
        if grid[py][px]:
            return (px, py)

        for r in range(1, max_radius + 1):
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

    def _bresenham_line(
        self, start: tuple[int, int], end: tuple[int, int]
    ) -> list[tuple[int, int]]:
        x0, y0 = start
        x1, y1 = end
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        sx = 1 if x1 >= x0 else -1
        sy = 1 if y1 >= y0 else -1
        x, y = x0, y0
        points = [(x, y)]

        if dx >= dy:
            err = dx // 2
            while x != x1:
                x += sx
                err -= dy
                if err < 0:
                    y += sy
                    err += dx
                points.append((x, y))
        else:
            err = dy // 2
            while y != y1:
                y += sy
                err -= dx
                if err < 0:
                    x += sx
                    err += dy
                points.append((x, y))
        return points

    def _has_line_of_sight(self, start: tuple[int, int], end: tuple[int, int]) -> bool:
        primary_margin = int(self.margins[0]) if self.margins else 0
        grid = self.safe_passable_grids.get(primary_margin)
        if grid is None:
            return False
        for x, y in self._bresenham_line(start, end):
            if not (0 <= x < self.map_width and 0 <= y < self.map_height):
                return False
            if not grid[y][x]:
                return False
        return True

    def _smooth_path(self, path: list[tuple[int, int]]) -> list[tuple[int, int]]:
        if not path or len(path) <= 2:
            return path
        smoothed = [path[0]]
        index = 0
        last_idx = len(path) - 1
        while index < last_idx:
            next_index = index + 1
            for candidate in range(last_idx, index, -1):
                if self._has_line_of_sight(path[index], path[candidate]):
                    next_index = candidate
                    break
            smoothed.append(path[next_index])
            index = next_index
        return smoothed

    def _densify_path(self, nodes: Sequence[tuple[int, int]]) -> list[tuple[int, int]]:
        if not nodes:
            return []
        expanded = [nodes[0]]
        for a, b in zip(nodes, nodes[1:]):
            segment = self._bresenham_line(a, b)
            expanded.extend(segment[1:])
        return expanded

    def _build_greedy_follower(self):
        if self.sim is None or not hasattr(self.sim, "make_greedy_follower"):
            return None
        scene = None
        try:
            scene = str(getattr(self.sim.habitat_config, "SCENE", ""))
        except Exception:
            scene = ""
        if self._greedy_follower is not None and self._greedy_scene == scene:
            return self._greedy_follower

        stop_key = 0
        forward_key = 1
        left_key = 2
        right_key = 3
        if HabitatSimActions is not None:
            stop_key = HabitatSimActions.STOP
            forward_key = HabitatSimActions.MOVE_FORWARD
            left_key = HabitatSimActions.TURN_LEFT
            right_key = HabitatSimActions.TURN_RIGHT

        self._greedy_follower = self.sim.make_greedy_follower(
            0,
            float(self.config.GREEDY_GOAL_RADIUS),
            stop_key=stop_key,
            forward_key=forward_key,
            left_key=left_key,
            right_key=right_key,
        )
        self._greedy_scene = scene
        return self._greedy_follower

    def plan_greedy_actions(self, goal_xy: Sequence[float], max_steps: int = 6):
        follower = self._build_greedy_follower()
        if follower is None:
            return None, "greedy_follower_unavailable"
        if goal_xy is None or len(goal_xy) < 2:
            return None, "missing_goal_xy"
        if self.sim is None:
            return None, "missing_habitat_sim"

        try:
            state = self.sim.get_agent_state()
            if self.goal_position_3d is not None:
                goal_3d = self.goal_position_3d.astype(np.float32)
            else:
                goal_3d = np.array(
                    [float(goal_xy[0]), float(state.position[1]), float(goal_xy[1])],
                    dtype=np.float32,
                )
            path_actions = follower.find_path(goal_3d)
        except Exception as e:
            return None, f"greedy_find_path_failed:{e}"

        if path_actions is None:
            actions = []
        else:
            actions = []
            for action in path_actions:
                try:
                    actions.append(int(action))
                except Exception:
                    continue
                if len(actions) >= int(max_steps):
                    break

        if len(actions) < int(max_steps):
            actions.extend([0] * (int(max_steps) - len(actions)))
        return actions[: int(max_steps)], "greedy_ok"

    def _yaw_from_habitat_state(self, state: Any) -> float:
        rot = self._quat_to_xyzw(state.rotation)
        rot_obj = R.from_quat(rot)
        fwd_world = rot_obj.apply(np.array([1.0, 0.0, 0.0], dtype=np.float32))
        return float(math.atan2(float(fwd_world[2]), float(fwd_world[0])))

    def rollout_actions_habitat_trace(self, actions: Sequence[int]):
        if self.sim is None:
            return False, "habitat_sim_unavailable", []
        if not actions or len(actions) != 6:
            return False, "invalid_actions_len", []

        try:
            init_state = self.sim.get_agent_state()
            init_pos = np.array(init_state.position, dtype=np.float32)
            init_rot = self._quat_to_xyzw(init_state.rotation)
        except Exception as e:
            return False, f"habitat_state_unavailable:{e}", []

        trace: list[dict[str, Any]] = []
        try:
            current_state = self.sim.get_agent_state()
            current_px = self.transform_from_world_to_pixel(
                np.array(
                    [float(current_state.position[0]), float(current_state.position[2]), 0.0],
                    dtype=np.float32,
                )
            )
            current_yaw = self._yaw_from_habitat_state(current_state)

            for step_idx, action in enumerate(map(int, actions)):
                if action not in {0, 1, 2, 3}:
                    return False, f"unknown_action_{action}", trace

                start_state = self.sim.get_agent_state()
                start_px = current_px
                start_yaw = current_yaw
                collided = False

                if action != 0:
                    try:
                        self.sim.step(action)
                    except Exception as e:
                        return False, f"habitat_step_error_step_{step_idx}:{e}", trace
                    collided = bool(getattr(self.sim, "previous_step_collided", False))

                end_state = self.sim.get_agent_state()
                end_px = self.transform_from_world_to_pixel(
                    np.array(
                        [float(end_state.position[0]), float(end_state.position[2]), 0.0],
                        dtype=np.float32,
                    )
                )
                end_yaw = self._yaw_from_habitat_state(end_state)
                end_pos = [float(end_state.position[0]), float(end_state.position[2])]

                trace.append(
                    {
                        "step_idx": int(step_idx),
                        "action": int(action),
                        "start_pixel": [int(start_px[0]), int(start_px[1])],
                        "end_pixel": [int(end_px[0]), int(end_px[1])],
                        "start_yaw": float(start_yaw),
                        "end_yaw": float(end_yaw),
                        "collided": bool(collided),
                        "start_pos": [
                            float(start_state.position[0]),
                            float(start_state.position[2]),
                        ],
                        "end_pos": end_pos,
                    }
                )

                current_px = end_px
                current_yaw = end_yaw

                if action == 1 and collided:
                    return False, f"habitat_rollout_hit_wall_step_{step_idx}", trace

            return True, "ok_habitat_rollout", trace
        finally:
            try:
                self.sim.set_agent_state(init_pos, init_rot, reset_sensors=False)
            except TypeError:
                self.sim.set_agent_state(init_pos, init_rot)
            except Exception:
                pass
            self._sync_state_and_trajectory()

    def distance_to_goal_3d(self) -> Optional[float]:
        if self.sim is None or self.goal_position_3d is None:
            return None
        try:
            state = self.sim.get_agent_state()
            curr = np.array(state.position, dtype=np.float32)
            geodesic_distance = getattr(self.sim, "geodesic_distance", None)
            if callable(geodesic_distance):
                try:
                    return float(
                        geodesic_distance(
                            curr.tolist(),
                            [self.goal_position_3d.tolist()],
                        )
                    )
                except Exception:
                    pass
            return float(np.linalg.norm(curr - self.goal_position_3d))
        except Exception:
            return None

    def get_goal_position_3d(self) -> Optional[np.ndarray]:
        if self.goal_position_3d is None:
            return None
        return np.array(self.goal_position_3d, dtype=np.float32)

    def get_success_distance(self) -> float:
        env_cfg = getattr(self.env, "_config", None) or getattr(self.env, "config", None)
        task_cfg = getattr(env_cfg, "TASK", None)

        for owner in (task_cfg, getattr(task_cfg, "SUCCESS", None)):
            if owner is None:
                continue
            value = getattr(owner, "SUCCESS_DISTANCE", None)
            try:
                if value is not None:
                    return float(value)
            except Exception:
                pass

        return 3.0

    def rollout_actions_valid_habitat(self, actions: Sequence[int]):
        ok, reason, _ = self.rollout_actions_habitat_trace(actions)
        return bool(ok), str(reason)
