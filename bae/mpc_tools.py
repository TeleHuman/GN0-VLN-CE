# mpc_planner.py
# -*- coding: utf-8 -*-
"""
Minimal importable MPC planner:
- Smooth a 2D world polyline
- Build Polyline2D geometry helper
- Receding-horizon beam-search MPC to produce discrete actions
- Output contains NO STOP (0)

Action IDs (matching your MPC script):
  STOP=0, FWD=1, LEFT=2, RIGHT=3
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import math
import numpy as np

__all__ = [
    "STOP", "FWD", "LEFT", "RIGHT",
    "wrap_pi", "deg2rad", "rad2deg",
    "smooth_polyline",
    "Polyline2D",
    "plan_actions_beam_mpc",
]

# ======= action ids =======
STOP = 0
FWD = 1
LEFT = 2
RIGHT = 3


def wrap_pi(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi


def deg2rad(d: float) -> float:
    return d * math.pi / 180.0


def rad2deg(r: float) -> float:
    return r * 180.0 / math.pi


# ========= smoothing =========

def _kernel(win: int, kind: str = "tri") -> np.ndarray:
    if win < 1 or win % 2 != 1:
        raise ValueError("win must be odd and >= 1")
    if win == 1:
        return np.array([1.0], dtype=np.float64)

    if kind == "box":
        k = np.ones(win, dtype=np.float64)
    elif kind == "tri":
        mid = win // 2
        k = np.concatenate([np.arange(1, mid + 2), np.arange(mid, 0, -1)]).astype(np.float64)
    elif kind == "gauss":
        sigma = win / 6.0
        x = np.arange(win) - win // 2
        k = np.exp(-0.5 * (x / sigma) ** 2)
    else:
        raise ValueError(f"unknown kernel kind: {kind}")

    k /= np.sum(k)
    return k


def _smooth_polyline_pos(pts: np.ndarray, win: int, kind: str) -> np.ndarray:
    if win <= 1 or len(pts) < win:
        return pts
    if win % 2 == 0:
        win += 1
    k = _kernel(win, kind)
    pad = win // 2
    out = pts.copy()

    for d in range(2):
        x = pts[:, d]
        xp = np.pad(x, (pad, pad), mode="reflect")
        y = np.convolve(xp, k, mode="valid")
        out[:, d] = y

    # keep ends
    out[0] = pts[0]
    out[-1] = pts[-1]
    return out


def _smooth_polyline_vel(pts: np.ndarray, win: int, kind: str) -> np.ndarray:
    if len(pts) < 3:
        return pts
    if win <= 1:
        return pts
    if win % 2 == 0:
        win += 1
    if (len(pts) - 1) < win:
        return pts

    k = _kernel(win, kind)
    pad = win // 2

    v = pts[1:] - pts[:-1]  # (N-1,2)
    vlen = np.linalg.norm(v, axis=1) + 1e-12

    vs = np.zeros_like(v)
    for d in range(2):
        x = v[:, d]
        xp = np.pad(x, (pad, pad), mode="reflect")
        y = np.convolve(xp, k, mode="valid")
        vs[:, d] = y

    vs_len = np.linalg.norm(vs, axis=1) + 1e-12
    vs = vs * (vlen[:, None] / vs_len[:, None])  # preserve each segment length

    out = np.zeros_like(pts)
    out[0] = pts[0]
    out[1:] = pts[0] + np.cumsum(vs, axis=0)

    # force endpoint alignment (distribute correction along the path)
    delta = pts[-1] - out[-1]
    t = np.linspace(0.0, 1.0, len(out))[:, None]
    out = out + t * delta

    out[0] = pts[0]
    out[-1] = pts[-1]
    return out


def smooth_polyline(pts: np.ndarray, mode: str = "vel", win: int = 9, kind: str = "tri") -> np.ndarray:
    """
    pts: (N,2)
    mode: "none" | "pos" | "vel"
    """
    pts = np.asarray(pts, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 2:
        raise ValueError("pts must be (N,2).")
    if mode == "none" or win <= 1:
        return pts
    if win % 2 == 0:
        win += 1
    if mode == "pos":
        return _smooth_polyline_pos(pts, win=win, kind=kind)
    if mode == "vel":
        return _smooth_polyline_vel(pts, win=win, kind=kind)
    raise ValueError(f"unknown smooth mode: {mode}")


# ========= geometry =========

class Polyline2D:
    """
    2D polyline with arc-length parameterization and projection.
    """
    def __init__(self, pts_xy: np.ndarray):
        pts_xy = np.asarray(pts_xy, dtype=np.float64)
        if pts_xy.ndim != 2 or pts_xy.shape[1] != 2 or len(pts_xy) < 2:
            raise ValueError("Polyline must be (N,2) with N>=2.")
        self.pts = pts_xy

        seg = self.pts[1:] - self.pts[:-1]
        seg_len = np.linalg.norm(seg, axis=1)
        seg_len = np.maximum(seg_len, 1e-12)

        self.seg_len = seg_len
        self.cum_s = np.concatenate([[0.0], np.cumsum(seg_len)])

    @property
    def total_length(self) -> float:
        return float(self.cum_s[-1])

    def point_at_s(self, s: float) -> np.ndarray:
        s = float(np.clip(s, 0.0, self.total_length))
        i = int(np.searchsorted(self.cum_s, s, side="right") - 1)
        i = max(0, min(i, len(self.seg_len) - 1))
        s0 = self.cum_s[i]
        t = (s - s0) / self.seg_len[i]
        return self.pts[i] + t * (self.pts[i + 1] - self.pts[i])

    def tangent_angle_at_s(self, s: float) -> float:
        s = float(np.clip(s, 0.0, self.total_length))
        i = int(np.searchsorted(self.cum_s, s, side="right") - 1)
        i = max(0, min(i, len(self.seg_len) - 1))
        v = self.pts[i + 1] - self.pts[i]
        return math.atan2(float(v[1]), float(v[0]))

    def project(self, p: np.ndarray) -> Tuple[np.ndarray, float, float]:
        """
        Project point p onto polyline segments.
        Returns: (proj_xy, s, dist)
        """
        p = np.asarray(p, dtype=np.float64).reshape(2)
        a = self.pts[:-1]
        b = self.pts[1:]
        ab = b - a
        ab2 = np.sum(ab * ab, axis=1)
        ap = p[None, :] - a
        t = np.sum(ap * ab, axis=1) / np.maximum(ab2, 1e-12)
        t = np.clip(t, 0.0, 1.0)
        proj = a + t[:, None] * ab
        d = np.linalg.norm(proj - p[None, :], axis=1)
        j = int(np.argmin(d))
        s = float(self.cum_s[j] + t[j] * self.seg_len[j])
        return proj[j], s, float(d[j])


# ========= MPC =========

def _simulate_step(x: float, y: float, th: float, a: int,
                   step_m: float, turn_rad: float) -> Tuple[float, float, float]:
    if a == LEFT:
        th = wrap_pi(th + turn_rad)
    elif a == RIGHT:
        th = wrap_pi(th - turn_rad)
    elif a == FWD:
        x = x + step_m * math.cos(th)
        y = y + step_m * math.sin(th)
    elif a == STOP:
        pass
    return x, y, th


@dataclass
class _Node:
    g: float
    f: float
    x: float
    y: float
    th: float
    s: float
    actions: List[int]


def _turn_streak(seq: List[int]) -> int:
    k = 0
    for a in reversed(seq):
        if a in (LEFT, RIGHT):
            k += 1
        else:
            break
    return k


def _heur_steps_and_turns(nx: float, ny: float, nth: float,
                          gx: float, gy: float,
                          step_m: float, turn_rad: float) -> Tuple[float, float, float]:
    dx = gx - nx
    dy = gy - ny
    dist = math.hypot(dx, dy)
    th_goal = math.atan2(dy, dx)
    dang = abs(wrap_pi(th_goal - nth))
    h = (dist / max(step_m, 1e-9)) + (dang / max(turn_rad, 1e-9))
    return dist, dang, h


def _strip_stop(actions: List[int]) -> List[int]:
    """
    Output rule: STOP(0) must NOT appear in final sequence.
    If STOP appears, terminate at first STOP and remove it.
    """
    if not actions:
        return []
    if STOP in actions:
        actions = actions[:actions.index(STOP)]
    return [int(a) for a in actions if int(a) != STOP]


def plan_actions_beam_mpc(
    poly: Polyline2D,
    x0: float, y0: float, th0: float,
    lookahead_m: float = 5.0,
    horizon: int = 32,
    beam: int = 300,
    step_m: float = 0.25,
    turn_deg: float = 15.0,
    goal_stop_m: float = 0.15,
    max_steps: int = 800,
    relocalize: bool = False,
    relocalize_thresh: float = 0.35,

    # stage weights (defaults = your script defaults)
    w_step: float = 1.0,
    w_turn: float = 0.5,
    w_perp: float = 300.0,
    d0: float = 0.03,
    w_head: float = 0.10,
    w_head_tangent: float = 0.10,
    w_switch: float = 10.0,
    w_terminal: float = 120.0,
    w_progress: float = 2.0,
    w_back: float = 20.0,

    # pruning / anti-dither
    w_goal_heur: float = 1.5,
    w_spin: float = 10.0,
    turn_slack: int = 1,
    commit: int = 2,
    stall_steps: int = 20,
    stall_ds_eps: float = 1e-3,

    # endgame
    endgame_dist: float = 0.0,
    endgame_turn_tol_deg: float = 7.5,

    # STOP modeling
    w_stop_good: float = -80.0,
    w_stop_bad: float = 300.0,
) -> List[int]:
    """
    Returns a list of actions in {FWD, LEFT, RIGHT} (STOP removed).
    STOP is considered only during planning.
    """
    turn_rad = deg2rad(turn_deg)
    endgame_turn_tol = deg2rad(endgame_turn_tol_deg)

    # initial projection for progress parameter s
    proj0, s0, _ = poly.project(np.array([x0, y0], dtype=np.float64))
    if relocalize:
        x0, y0 = float(proj0[0]), float(proj0[1])

    x, y, th, s = float(x0), float(y0), float(th0), float(s0)
    end = poly.point_at_s(poly.total_length)
    ex, ey = float(end[0]), float(end[1])

    def dist_to_end(xx: float, yy: float) -> float:
        return math.hypot(xx - ex, yy - ey)

    actions: List[int] = []
    prev_turn: Optional[int] = None
    stall_ctr = 0
    s_last_exec = s
    executed = 0
    terminated = False

    while executed < max_steps and not terminated:
        if endgame_dist > 0.0 and dist_to_end(x, y) <= float(endgame_dist):
            break

        eff_lookahead = min(float(lookahead_m), float(horizon) * float(step_m))

        _, s_now, _ = poly.project(np.array([x, y], dtype=np.float64))
        s = float(s_now)
        s_goal = min(s + eff_lookahead, poly.total_length)
        gpt = poly.point_at_s(s_goal)
        gx, gy = float(gpt[0]), float(gpt[1])

        # ----- beam search over horizon -----
        beam_nodes: List[_Node] = [_Node(g=0.0, f=0.0, x=x, y=y, th=th, s=s, actions=[])]
        for _depth in range(int(horizon)):
            cand: List[_Node] = []
            for node in beam_nodes:
                if node.actions and node.actions[-1] == STOP:
                    cand.append(node)
                    continue

                _, dang_node, _ = _heur_steps_and_turns(node.x, node.y, node.th, gx, gy, step_m, turn_rad)
                need_turns = int(math.ceil(dang_node / max(turn_rad, 1e-9)))
                turns_done = _turn_streak(node.actions)

                for a in (FWD, LEFT, RIGHT, STOP):
                    nx, ny, nth = _simulate_step(node.x, node.y, node.th, a, step_m, turn_rad)
                    _proj, ns, dperp = poly.project(np.array([nx, ny], dtype=np.float64))

                    th_goal = math.atan2(gy - ny, gx - nx)
                    dth_goal = wrap_pi(nth - th_goal)

                    th_ref = poly.tangent_angle_at_s(ns)
                    dth_ref = wrap_pi(nth - th_ref)

                    c = 0.0
                    if a == STOP:
                        d_end = math.hypot(nx - ex, ny - ey)
                        c += (w_stop_good if d_end <= goal_stop_m else w_stop_bad)
                    else:
                        c += w_step
                        if a in (LEFT, RIGHT):
                            c += w_turn

                        dd = max(0.0, float(dperp) - float(d0))
                        c += w_perp * (dd * dd)

                        c += w_head * (dth_goal * dth_goal)
                        if w_head_tangent > 0.0:
                            c += w_head_tangent * (dth_ref * dth_ref)

                        ds = float(ns) - float(node.s)
                        c -= w_progress * ds
                        if ds < 0.0:
                            c += w_back * (ds * ds)

                        if node.actions:
                            last = node.actions[-1]
                            if (last == LEFT and a == RIGHT) or (last == RIGHT and a == LEFT):
                                c += w_switch

                        if a in (LEFT, RIGHT):
                            turns_after = turns_done + 1
                            free_allow = turns_done + need_turns + int(max(0, turn_slack))
                            excess = max(0, turns_after - free_allow)
                            if excess > 0:
                                c += w_spin * float(excess)

                    g_new = node.g + c
                    _dist, _dang, h = _heur_steps_and_turns(nx, ny, nth, gx, gy, step_m, turn_rad)
                    f_new = g_new if a == STOP else (g_new + w_goal_heur * h)

                    cand.append(_Node(
                        g=float(g_new), f=float(f_new),
                        x=float(nx), y=float(ny), th=float(nth), s=float(ns),
                        actions=node.actions + [int(a)],
                    ))

            cand.sort(key=lambda n: n.f)
            beam_nodes = cand[: int(beam)]

        # ----- choose best with terminal cost -----
        best = None
        best_cost = float("inf")
        for node in beam_nodes:
            dx, dy = node.x - gx, node.y - gy
            term = w_terminal * (dx * dx + dy * dy)

            extra = 0.0
            if prev_turn is not None and node.actions:
                first = node.actions[0]
                if (prev_turn == LEFT and first == RIGHT) or (prev_turn == RIGHT and first == LEFT):
                    extra += w_switch

            total = node.g + term + extra
            if total < best_cost:
                best_cost = total
                best = node

        plan = best.actions if (best and best.actions) else [FWD]

        # stall escape
        if stall_ctr >= int(stall_steps):
            plan = [FWD]
            stall_ctr = 0

        # ----- execute first commit steps (receding horizon) -----
        kmax = max(1, int(commit))
        for k in range(min(kmax, len(plan))):
            a0 = int(plan[k])

            # if decided STOP => terminate (do NOT append STOP)
            if a0 == STOP:
                terminated = True
                break

            x, y, th = _simulate_step(x, y, th, a0, step_m, turn_rad)
            if a0 in (LEFT, RIGHT):
                prev_turn = a0

            if relocalize:
                proj_exec, s_exec, d_exec = poly.project(np.array([x, y], dtype=np.float64))
                s = float(s_exec)
                if float(d_exec) <= float(relocalize_thresh):
                    x, y = float(proj_exec[0]), float(proj_exec[1])
            else:
                _, s, _ = poly.project(np.array([x, y], dtype=np.float64))
                s = float(s)

            actions.append(a0)
            executed += 1

            ds_exec = s - s_last_exec
            stall_ctr = 0 if ds_exec > float(stall_ds_eps) else (stall_ctr + 1)
            s_last_exec = s

            if executed >= max_steps:
                break

            # (same behavior as your script) once we execute a FWD, end this MPC cycle
            if a0 == FWD:
                break

        if terminated:
            break

        # if within stop_dist => terminate (do NOT append STOP)
        if dist_to_end(x, y) <= float(goal_stop_m):
            terminated = True
            break

    # ----- optional endgame controller -----
    if float(endgame_dist) > 0.0 and (not terminated):
        while executed < max_steps and (not terminated):
            dx = ex - x
            dy = ey - y
            d = math.hypot(dx, dy)

            if d <= float(goal_stop_m):
                terminated = True
                break

            th_goal = math.atan2(dy, dx)
            dang = wrap_pi(th_goal - th)

            if abs(dang) > max(float(endgame_turn_tol), 0.5 * float(turn_rad)):
                a0 = LEFT if dang > 0.0 else RIGHT
            else:
                if d < 0.5 * float(step_m):
                    terminated = True
                    break
                a0 = FWD

            x, y, th = _simulate_step(x, y, th, a0, step_m, turn_rad)

            if relocalize:
                proj_exec, s_exec, d_exec = poly.project(np.array([x, y], dtype=np.float64))
                if float(d_exec) <= float(relocalize_thresh):
                    x, y = float(proj_exec[0]), float(proj_exec[1])

            actions.append(int(a0))
            executed += 1

    return _strip_stop(actions)
