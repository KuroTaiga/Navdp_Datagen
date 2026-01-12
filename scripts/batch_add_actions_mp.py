#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Multiprocess batch: only process JSONs under:
  <root>/<scene>/label_paths/**/*.json

What it does
- For each JSON, read:
    data["path"]["raster_world"] : polyline points [{x,y}, ...] or [[x,y], ...]
    data["start"]["world"]       : {x,y}
- Smooth the polyline -> data["path"]["raster_world_smooth"]
- Plan VLN-CE discrete actions with receding-horizon beam-search MPC:
    STOP=0, FWD=1 (0.25m), LEFT=2 (turn_deg), RIGHT=3 (turn_deg)
- MPC includes STOP as a candidate action decision:
    - If within stop_dist to end: STOP gets reward (negative cost)
    - Else: STOP gets heavy penalty
- Write actions to data[--field] (default "actions")

IMPORTANT OUTPUT RULE
- STOP(0) may be considered during MPC decision-making,
  BUT the final written action sequence MUST NOT contain 0 at all:
    - if STOP appears in decided plan -> terminate and DO NOT append 0
    - if the planner would "append STOP at end" -> DO NOT do that
    - if existing actions contain STOP -> strip it out during recompute

Resume / skip behavior (compatible with your existing JSONs)
- On each JSON, BEFORE recomputing anything, the worker checks:
    1) data["path"]["raster_world_smooth_meta"] exists, and its fields (except init_yaw_rad)
       match the metadata that would be written with current CLI args
    2) data[--field] exists and is non-empty AND contains NO STOP(0)
    3) data["path"]["raster_world_smooth"] exists and has >=2 points
  If all true => SKIP.
- Use --force to disable this skip and recompute everything.

NOTE ABOUT RELOCALIZE
- Default: relocalize is OFF (pure kinematics state update + projection used only for costs).
- Enable relocalize explicitly with --relocalize if you want snapping during decision/rollout.

Examples:
  # dry run
  python batch_add_rasterworld_actions_mpc_mp.py /path/to/task_outputs_10w --dry_run

  # write outputs + backups if missing
  python batch_add_rasterworld_actions_mpc_mp.py /path/to/task_outputs_10w --backup_if_missing

  # enable viz png next to each json
  python batch_add_rasterworld_actions_mpc_mp.py /path/to/task_outputs_10w --viz

  # enable relocalize explicitly (NOT default)
  python batch_add_rasterworld_actions_mpc_mp.py /path/to/task_outputs_10w --relocalize
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import math
import os
import shutil
import sys
import time
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np


# ======= VLN-CE / Habitat-style action ids =======
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


# ========= Resume-by-metadata (compatible with your current JSON) =========

def _num_equal(a: Any, b: Any, tol: float = 1e-9) -> bool:
    if isinstance(a, (int, float)) and isinstance(b, (int, float)) and (not isinstance(a, bool)) and (not isinstance(b, bool)):
        return abs(float(a) - float(b)) <= tol
    return a == b


def _template_match(existing: Any, templ: Any, tol: float = 1e-9) -> bool:
    """
    True if `existing` contains at least the structure/values of `templ`.
    Extra keys in existing are allowed.
    Numeric values compared with tolerance.
    """
    if isinstance(templ, dict):
        if not isinstance(existing, dict):
            return False
        for k, v in templ.items():
            if k not in existing:
                return False
            if not _template_match(existing[k], v, tol=tol):
                return False
        return True

    if isinstance(templ, list):
        if not isinstance(existing, list):
            return False
        if len(existing) != len(templ):
            return False
        for a, b in zip(existing, templ):
            if not _template_match(a, b, tol=tol):
                return False
        return True

    return _num_equal(existing, templ, tol=tol)


def build_expected_meta_template(args: argparse.Namespace) -> Dict[str, Any]:
    """
    Template of raster_world_smooth_meta fields that must match existing JSON
    to consider the file already done. We intentionally EXCLUDE per-file
    init_yaw_rad from the template (but we require it exists in outputs).
    """
    return {
        "mode": str(args.smooth),
        "win": int(args.smooth_win),
        "kind": str(args.smooth_kind),
        "note": "smoothed version of path.raster_world used for planning",
        "relocalize": bool(args.relocalize),
        "init_yaw": "start_to_first_smooth",
        "init_yaw_eps": float(args.init_yaw_eps),
        "quantize_init_yaw": bool(args.quantize_init_yaw),
        "planner": {
            "horizon": int(args.horizon),
            "beam": int(args.beam),
            "lookahead": float(args.lookahead),
            "step": float(args.step),
            "turn_deg": float(args.turn_deg),
            "stop_dist": float(args.stop_dist),
            "stop_shaping": {
                "w_stop_good": float(args.w_stop_good),
                "w_stop_bad": float(args.w_stop_bad),
            },
        },
        "endgame": {
            "endgame_dist": float(args.endgame_dist),
            "endgame_turn_tol_deg": float(args.endgame_turn_tol_deg),
        },
    }


def has_valid_existing_outputs(data: Dict[str, Any], field: str) -> bool:
    acts = data.get(field, None)
    if not (isinstance(acts, list) and len(acts) > 0):
        return False
    # output should contain NO STOP at all
    if any(int(a) == STOP for a in acts):
        return False

    smooth = data.get("path", {}).get("raster_world_smooth", None)
    if not (isinstance(smooth, list) and len(smooth) >= 2):
        return False

    meta = data.get("path", {}).get("raster_world_smooth_meta", None)
    if not (isinstance(meta, dict) and isinstance(meta.get("init_yaw_rad", None), (int, float))):
        return False

    return True


# ========= Path smoothing =========

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


def smooth_polyline_pos(pts: np.ndarray, win: int = 9, kind: str = "tri", keep_ends: bool = True) -> np.ndarray:
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

    if keep_ends:
        out[0] = pts[0]
        out[-1] = pts[-1]
    return out


def smooth_polyline_vel(pts: np.ndarray, win: int = 9, kind: str = "tri", keep_ends: bool = True) -> np.ndarray:
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
    vs = vs * (vlen[:, None] / vs_len[:, None])  # preserve each step length

    out = np.zeros_like(pts)
    out[0] = pts[0]
    out[1:] = pts[0] + np.cumsum(vs, axis=0)

    # force endpoint alignment
    delta = pts[-1] - out[-1]
    t = np.linspace(0.0, 1.0, len(out))[:, None]
    out = out + t * delta

    if keep_ends:
        out[0] = pts[0]
        out[-1] = pts[-1]
    return out


def smooth_polyline(pts: np.ndarray, mode: str = "vel", win: int = 9, kind: str = "tri") -> np.ndarray:
    if mode == "none" or win <= 1:
        return pts
    if win % 2 == 0:
        win += 1
    if mode == "pos":
        return smooth_polyline_pos(pts, win=win, kind=kind, keep_ends=True)
    if mode == "vel":
        return smooth_polyline_vel(pts, win=win, kind=kind, keep_ends=True)
    raise ValueError(f"unknown smooth mode: {mode}")


# ========= Core geometry =========

class Polyline2D:
    def __init__(self, pts_xy: np.ndarray):
        if pts_xy.ndim != 2 or pts_xy.shape[1] != 2 or len(pts_xy) < 2:
            raise ValueError("Polyline must be (N,2) with N>=2.")
        self.pts = pts_xy.astype(np.float64)

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
        p = p.astype(np.float64)
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


def simulate_step(x: float, y: float, th: float, a: int,
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


def initial_yaw_start_to_first_smooth(
    pts_for_yaw: np.ndarray,                # pass smoothed pts
    start_xy: Tuple[float, float],
    fallback_poly: Polyline2D,
    turn_deg: float,
    eps: float = 1e-3,
    quantize: bool = False,
) -> float:
    sx, sy = float(start_xy[0]), float(start_xy[1])

    # pick the first point on *smoothed* path that is not too close to start
    for i in range(len(pts_for_yaw)):
        dx = float(pts_for_yaw[i, 0]) - sx
        dy = float(pts_for_yaw[i, 1]) - sy
        if (dx * dx + dy * dy) > (eps * eps):
            th = math.atan2(dy, dx)
            if quantize:
                q = deg2rad(turn_deg)
                th = q * round(th / q)
            return wrap_pi(th)

    # fallback: tangent at projection on smoothed polyline
    _, s0, _ = fallback_poly.project(np.array([sx, sy], dtype=np.float64))
    th = fallback_poly.tangent_angle_at_s(s0)
    if quantize:
        q = deg2rad(turn_deg)
        th = q * round(th / q)
    return wrap_pi(th)


# ========= Planner =========

@dataclass
class Node:
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
    actions = [int(a) for a in actions if int(a) != STOP]
    return actions


def plan_actions_beam_mpc(
    poly: Polyline2D,
    x0: float, y0: float, th0: float,
    lookahead_m: float,
    horizon: int,
    beam: int,
    step_m: float,
    turn_deg: float,
    goal_stop_m: float,
    max_steps: int,
    relocalize: bool,
    relocalize_thresh: float,

    # stage weights
    w_step: float = 1.0,
    w_turn: float = 0.20,
    w_perp: float = 80.0,
    d0: float = 0.15,
    w_head: float = 0.50,
    w_head_tangent: float = 0.10,
    w_switch: float = 0.30,
    w_terminal: float = 60.0,
    w_progress: float = 2.0,
    w_back: float = 20.0,

    # pruning / anti-dither
    w_goal_heur: float = 3.0,
    w_spin: float = 2.0,
    turn_slack: int = 1,
    commit: int = 2,
    stall_steps: int = 20,
    stall_ds_eps: float = 1e-3,

    # endgame
    endgame_dist: float = 0.0,
    endgame_turn_tol_deg: float = 7.5,

    # STOP modeling
    w_stop_good: float = -80.0,  # within goal_stop_m => reward (negative cost)
    w_stop_bad: float = 300.0,   # outside goal_stop_m => heavy penalty
) -> List[int]:
    turn_rad = deg2rad(turn_deg)
    endgame_turn_tol = deg2rad(endgame_turn_tol_deg)

    # projection used for s0; optional snapping to start projection if relocalize enabled
    proj0, s0, _ = poly.project(np.array([x0, y0], dtype=np.float64))
    if relocalize:
        x0, y0 = float(proj0[0]), float(proj0[1])

    x, y, th, s = x0, y0, th0, s0
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

        eff_lookahead = min(lookahead_m, horizon * step_m)

        _, s_now, _ = poly.project(np.array([x, y], dtype=np.float64))
        s = s_now
        s_goal = min(s + eff_lookahead, poly.total_length)
        gpt = poly.point_at_s(s_goal)
        gx, gy = float(gpt[0]), float(gpt[1])

        beam_nodes: List[Node] = [Node(g=0.0, f=0.0, x=x, y=y, th=th, s=s, actions=[])]
        for _depth in range(horizon):
            cand: List[Node] = []
            for node in beam_nodes:
                if node.actions and node.actions[-1] == STOP:
                    cand.append(node)
                    continue

                _, dang_node, _ = _heur_steps_and_turns(node.x, node.y, node.th, gx, gy, step_m, turn_rad)
                need_turns_from_node = int(math.ceil(dang_node / max(turn_rad, 1e-9)))
                turns_done = _turn_streak(node.actions)

                for a in (FWD, LEFT, RIGHT, STOP):
                    nx, ny, nth = simulate_step(node.x, node.y, node.th, a, step_m, turn_rad)
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

                        dd = max(0.0, dperp - d0)
                        c += w_perp * (dd * dd)

                        c += w_head * (dth_goal * dth_goal)
                        if w_head_tangent > 0.0:
                            c += w_head_tangent * (dth_ref * dth_ref)

                        ds = ns - node.s
                        c -= w_progress * ds
                        if ds < 0.0:
                            c += w_back * (ds * ds)

                        if node.actions:
                            last = node.actions[-1]
                            if (last == LEFT and a == RIGHT) or (last == RIGHT and a == LEFT):
                                c += w_switch

                        if a in (LEFT, RIGHT):
                            turns_after = turns_done + 1
                            free_allow = turns_done + need_turns_from_node + int(max(0, turn_slack))
                            excess = max(0, turns_after - free_allow)
                            if excess > 0:
                                c += w_spin * float(excess)

                    g_new = node.g + c
                    _dist, _dang, h = _heur_steps_and_turns(nx, ny, nth, gx, gy, step_m, turn_rad)

                    f_new = g_new if a == STOP else (g_new + w_goal_heur * h)

                    cand.append(Node(
                        g=g_new, f=f_new,
                        x=nx, y=ny, th=nth, s=ns,
                        actions=node.actions + [a],
                    ))

            cand.sort(key=lambda n: n.f)
            beam_nodes = cand[:beam]

        best: Optional[Node] = None
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

        if stall_ctr >= stall_steps:
            plan = [FWD]
            stall_ctr = 0

        kmax = max(1, int(commit))
        for k in range(min(kmax, len(plan))):
            a0 = plan[k]

            if (FWD not in plan) and (STOP not in plan) and (a0 in (LEFT, RIGHT)):
                a0 = FWD

            # if decision is STOP: terminate WITHOUT appending STOP
            if a0 == STOP:
                terminated = True
                break

            x, y, th = simulate_step(x, y, th, a0, step_m, turn_rad)
            if a0 in (LEFT, RIGHT):
                prev_turn = a0

            # relocalize affects state update only when enabled; default OFF
            if relocalize:
                proj_exec, s_exec, d_exec = poly.project(np.array([x, y], dtype=np.float64))
                s = s_exec
                if d_exec <= relocalize_thresh:
                    x, y = float(proj_exec[0]), float(proj_exec[1])
            else:
                _, s, _ = poly.project(np.array([x, y], dtype=np.float64))

            actions.append(a0)
            executed += 1

            ds_exec = s - s_last_exec
            if ds_exec > stall_ds_eps:
                stall_ctr = 0
            else:
                stall_ctr += 1
            s_last_exec = s

            if executed >= max_steps:
                break
            if a0 == FWD:
                break

        if terminated:
            break

        # if within stop_dist, terminate WITHOUT appending STOP
        if dist_to_end(x, y) <= goal_stop_m:
            terminated = True
            break

    # optional endgame controller (still no STOP appended)
    if endgame_dist > 0.0 and not terminated:
        while executed < max_steps and not terminated:
            dx = ex - x
            dy = ey - y
            d = math.hypot(dx, dy)

            if d <= goal_stop_m:
                terminated = True
                break

            th_goal = math.atan2(dy, dx)
            dang = wrap_pi(th_goal - th)

            if abs(dang) > max(endgame_turn_tol, 0.5 * turn_rad):
                a0 = LEFT if dang > 0.0 else RIGHT
            else:
                if d < 0.5 * step_m:
                    terminated = True
                    break
                a0 = FWD

            x, y, th = simulate_step(x, y, th, a0, step_m, turn_rad)

            if relocalize:
                proj_exec, s_exec, d_exec = poly.project(np.array([x, y], dtype=np.float64))
                s = s_exec
                if d_exec <= relocalize_thresh:
                    x, y = float(proj_exec[0]), float(proj_exec[1])
            else:
                _, s, _ = poly.project(np.array([x, y], dtype=np.float64))

            actions.append(a0)
            executed += 1

    return _strip_stop(actions)


# ========= IO =========

def parse_raster_world(data: Dict[str, Any]) -> np.ndarray:
    rw = data.get("path", {}).get("raster_world", None)
    if rw is None or not isinstance(rw, list) or len(rw) < 2:
        raise ValueError("Missing/invalid path.raster_world (need >=2 points).")

    pts: List[Tuple[float, float]] = []
    for it in rw:
        if isinstance(it, dict) and "x" in it and "y" in it:
            pts.append((float(it["x"]), float(it["y"])))
        elif isinstance(it, (list, tuple)) and len(it) >= 2:
            pts.append((float(it[0]), float(it[1])))
        else:
            raise ValueError(f"Unsupported raster_world point format: {type(it)}")
    return np.asarray(pts, dtype=np.float64)


def parse_start_world(data: Dict[str, Any]) -> Tuple[float, float]:
    sw = data.get("start", {}).get("world", None)
    if not (isinstance(sw, dict) and ("x" in sw) and ("y" in sw)):
        raise ValueError("Missing start.world.x/y.")
    return float(sw["x"]), float(sw["y"])


def atomic_write_json(path: str, obj: Dict[str, Any], indent: Optional[int] = None) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=indent)
    os.replace(tmp, path)


def iter_scene_label_paths_json_stream(root: str) -> Iterable[str]:
    """
    Stream JSON paths under:
      <root>/<scene>/label_paths/**/*.json
    Without building a full list.
    """
    if not os.path.isdir(root):
        return
    for scene_name in os.listdir(root):
        scene_dir = os.path.join(root, scene_name)
        if not os.path.isdir(scene_dir):
            continue
        lp_dir = os.path.join(scene_dir, "label_paths")
        if not os.path.isdir(lp_dir):
            continue
        for dirpath, _dirnames, filenames in os.walk(lp_dir):
            for fn in filenames:
                if fn.lower().endswith(".json"):
                    yield os.path.join(dirpath, fn)


def _pts_to_raster_world_dicts(pts: np.ndarray) -> List[Dict[str, float]]:
    return [{"x": float(p[0]), "y": float(p[1])} for p in pts]


def _smooth_list_to_pts(smooth_list: List[Dict[str, float]]) -> np.ndarray:
    return np.asarray([(float(d["x"]), float(d["y"])) for d in smooth_list], dtype=np.float64)


def rollout_actions(
    actions: List[int],
    start_xy: Tuple[float, float],
    th0: float,
    step_m: float,
    turn_deg: float,
    poly_for_snap: Optional[Polyline2D] = None,
    relocalize: bool = False,
    relocalize_thresh: float = 0.35,
) -> np.ndarray:
    turn_rad = deg2rad(turn_deg)
    x, y, th = float(start_xy[0]), float(start_xy[1]), float(th0)
    traj = [(x, y)]
    for a in actions:
        x, y, th = simulate_step(x, y, th, int(a), step_m, turn_rad)
        if relocalize and (poly_for_snap is not None):
            proj, _s, d = poly_for_snap.project(np.array([x, y], dtype=np.float64))
            if d <= relocalize_thresh:
                x, y = float(proj[0]), float(proj[1])
        traj.append((x, y))
    return np.asarray(traj, dtype=np.float64)


def save_viz_png(
    out_png: str,
    raw_pts: np.ndarray,
    smooth_pts: np.ndarray,
    traj_xy: np.ndarray,
    start_xy: Tuple[float, float],
    init_yaw_rad: float,
    title: str,
    dpi: int = 180,
) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")  # headless
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"[WARN] matplotlib not available, skip viz: {out_png} ({e})", file=sys.stderr, flush=True)
        return

    plt.figure()

    plt.plot(raw_pts[:, 0], raw_pts[:, 1], linewidth=1.2, linestyle="--", alpha=0.7, label="ref raw (raster_world)")
    plt.plot(smooth_pts[:, 0], smooth_pts[:, 1], linewidth=2.0, label="ref smoothed")
    plt.plot(traj_xy[:, 0], traj_xy[:, 1], marker="o", markersize=3, linewidth=1.5, label="rollout")

    plt.scatter([raw_pts[0, 0]], [raw_pts[0, 1]], marker="s", s=60, label="path first (raw[0])")
    plt.scatter([raw_pts[-1, 0]], [raw_pts[-1, 1]], marker="*", s=120, label="path end (raw[-1])")
    plt.scatter([start_xy[0]], [start_xy[1]], marker="o", s=70, label="agent start (start.world)")

    yaw_len = 0.6
    ax0, ay0 = float(start_xy[0]), float(start_xy[1])
    ax1 = ax0 + yaw_len * math.cos(float(init_yaw_rad))
    ay1 = ay0 + yaw_len * math.sin(float(init_yaw_rad))
    plt.plot([ax0, ax1], [ay0, ay1], linewidth=2.0, label="init yaw")

    plt.axis("equal")
    plt.grid(True)
    plt.legend()
    plt.title(title)
    plt.xlabel("x (m)")
    plt.ylabel("y (m)")
    plt.tight_layout()
    plt.savefig(out_png, dpi=int(dpi))
    plt.close()


# ---------- worker: read + compute (no writing) ----------
def worker_compute(
    fp: str,
    cfg: Dict[str, Any],
    field: str,
    try_restore_backup: bool,
) -> Tuple[str, str, Optional[List[int]], Optional[List[Dict[str, float]]], Optional[float], int, Optional[str]]:
    """
    Returns:
      (fp, status, actions, smooth_list, init_yaw_rad, pid, err)
    status in {"ok", "skip"}
    """
    pid = os.getpid()
    try:
        def _read_json(p: str) -> Dict[str, Any]:
            with open(p, "r", encoding="utf-8") as f:
                return json.load(f)

        try:
            data = _read_json(fp)
        except Exception as e0:
            if try_restore_backup:
                bak = fp + ".bak"
                if os.path.isfile(bak):
                    shutil.copy2(bak, fp)
                    data = _read_json(fp)
                else:
                    raise e0
            else:
                raise e0

        # ---- skip if existing metadata matches current config and outputs look complete ----
        if not cfg.get("force", False):
            meta = data.get("path", {}).get("raster_world_smooth_meta", None)
            if isinstance(meta, dict):
                templ = cfg.get("meta_template", {})
                if _template_match(meta, templ, tol=1e-9) and has_valid_existing_outputs(data, field):
                    return fp, "skip", None, None, None, pid, None

        # ---- legacy: user-controlled resume by field presence only ----
        if cfg.get("skip_if_present", False):
            existing = data.get(field, None)
            if isinstance(existing, list) and len(existing) > 0:
                return fp, "skip", None, None, None, pid, None

        raw_pts = parse_raster_world(data)
        start_xy = parse_start_world(data)

        pts_s = smooth_polyline(
            raw_pts,
            mode=cfg["smooth_mode"],
            win=cfg["smooth_win"],
            kind=cfg["smooth_kind"],
        )

        poly = Polyline2D(pts_s)
        th0 = initial_yaw_start_to_first_smooth(
            pts_for_yaw=pts_s,
            start_xy=start_xy,
            fallback_poly=poly,
            turn_deg=cfg["turn_deg"],
            eps=cfg["init_yaw_eps"],
            quantize=cfg["quantize_init_yaw"],
        )

        actions = plan_actions_beam_mpc(
            poly=poly,
            x0=start_xy[0], y0=start_xy[1], th0=th0,
            lookahead_m=cfg["lookahead"],
            horizon=cfg["horizon"],
            beam=cfg["beam"],
            step_m=cfg["step"],
            turn_deg=cfg["turn_deg"],
            goal_stop_m=cfg["stop_dist"],
            max_steps=cfg["max_steps"],
            relocalize=cfg["relocalize"],
            relocalize_thresh=cfg["relocalize_thresh"],

            w_step=cfg["w_step"],
            w_turn=cfg["w_turn"],
            w_perp=cfg["w_perp"],
            d0=cfg["d0"],
            w_head=cfg["w_head"],
            w_head_tangent=cfg["w_head_tangent"],
            w_switch=cfg["w_switch"],
            w_terminal=cfg["w_terminal"],
            w_progress=cfg["w_progress"],
            w_back=cfg["w_back"],

            w_goal_heur=cfg["w_goal_heur"],
            w_spin=cfg["w_spin"],
            turn_slack=cfg["turn_slack"],
            commit=cfg["commit"],
            stall_steps=cfg["stall_steps"],
            stall_ds_eps=cfg["stall_ds_eps"],

            endgame_dist=cfg["endgame_dist"],
            endgame_turn_tol_deg=cfg["endgame_turn_tol_deg"],

            w_stop_good=cfg["w_stop_good"],
            w_stop_bad=cfg["w_stop_bad"],
        )

        actions = _strip_stop(list(actions))

        max_actions = int(cfg.get("max_actions", 300))
        if len(actions) > max_actions:
            return fp, "ok", None, None, None, pid, f"actions too long: len={len(actions)} > max_actions={max_actions}"

        smooth_list = _pts_to_raster_world_dicts(pts_s)
        return fp, "ok", actions, smooth_list, float(th0), pid, None

    except Exception as e:
        return fp, "ok", None, None, None, pid, str(e)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("root", help="Root directory: .../task_outputs_10w")

    ap.add_argument("--field", default="actions", help="Field name to write actions. Default: actions")
    ap.add_argument("--indent", type=int, default=None, help="json indent when writing (optional).")

    # write / resume controls
    ap.add_argument("--dry_run", action="store_true", help="Do not write; only report.")
    ap.add_argument("--force", action="store_true",
                    help="Recompute even if existing raster_world_smooth_meta matches current config.")
    ap.add_argument("--skip_if_present", action="store_true",
                    help="Legacy resume: skip JSONs that already have non-empty data[--field].")

    # backup / restore
    ap.add_argument("--backup_if_missing", action="store_true",
                    help="Create <file>.bak before writing if missing (do not overwrite existing .bak).")
    ap.add_argument("--restore_backup_on_read_fail", action="store_true",
                    help="If reading JSON fails and <file>.bak exists, restore .bak then retry read.")

    # parallel scheduling
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 8) - 1),
                    help="Worker processes. Default: cpu_count-1")
    ap.add_argument("--max_inflight", type=int, default=0,
                    help="Max in-flight futures. 0 => 4*workers (recommended).")
    ap.add_argument("--log_every", type=int, default=200,
                    help="Print progress every N finished files. Default: 200")
    ap.add_argument("--log_secs", type=float, default=10.0,
                    help="Also print progress every N seconds. Default: 10s")

    # planner params
    ap.add_argument("--lookahead", type=float, default=5.0)
    ap.add_argument("--horizon", type=int, default=32)
    ap.add_argument("--beam", type=int, default=300)
    ap.add_argument("--step", type=float, default=0.25)
    ap.add_argument("--turn_deg", type=float, default=15.0)
    ap.add_argument("--stop_dist", type=float, default=0.15)
    ap.add_argument("--max_steps", type=int, default=800)

    # relocalize: DEFAULT OFF (enable explicitly)
    ap.add_argument("--relocalize", action="store_true",
                    help="Enable relocalization (snap to path). Default: off.")
    ap.add_argument("--relocalize_thresh", type=float, default=0.35)

    # initial yaw
    ap.add_argument("--init_yaw_eps", type=float, default=1e-3,
                    help="Skip points closer than eps to start when computing start->first yaw.")
    ap.add_argument("--quantize_init_yaw", action="store_true",
                    help="Quantize initial yaw to nearest turn_deg. Default: off.")

    # endgame controller (optional)
    ap.add_argument("--endgame_dist", type=float, default=0.0,
                    help="If >0, within this distance to end use endgame controller. Default: 0")
    ap.add_argument("--endgame_turn_tol_deg", type=float, default=7.5)

    # STOP shaping
    ap.add_argument("--w_stop_good", type=float, default=-80.0,
                    help="Reward (negative cost) for STOP when within stop_dist.")
    ap.add_argument("--w_stop_bad", type=float, default=300.0,
                    help="Penalty for STOP when outside stop_dist.")

    # cost weights
    ap.add_argument("--w_step", type=float, default=1.0)
    ap.add_argument("--w_turn", type=float, default=0.5)
    ap.add_argument("--w_perp", type=float, default=300.0)
    ap.add_argument("--d0", type=float, default=0.03)
    ap.add_argument("--w_head", type=float, default=0.10)
    ap.add_argument("--w_head_tangent", type=float, default=0.10)
    ap.add_argument("--w_switch", type=float, default=10.0)
    ap.add_argument("--w_terminal", type=float, default=120.0)
    ap.add_argument("--w_progress", type=float, default=2.0)
    ap.add_argument("--w_back", type=float, default=20.0)

    # pruning / anti-dither
    ap.add_argument("--w_goal_heur", type=float, default=1.5)
    ap.add_argument("--w_spin", type=float, default=10.0)
    ap.add_argument("--turn_slack", type=int, default=1)
    ap.add_argument("--commit", type=int, default=2)
    ap.add_argument("--stall_steps", type=int, default=20)
    ap.add_argument("--stall_ds_eps", type=float, default=1e-3)

    # smoothing params
    ap.add_argument("--smooth", choices=["none", "pos", "vel"], default="vel")
    ap.add_argument("--smooth_win", type=int, default=9)
    ap.add_argument("--smooth_kind", choices=["box", "tri", "gauss"], default="tri")

    # visualization
    ap.add_argument("--viz", action="store_true", help="Save a PNG visualization next to each JSON.")
    ap.add_argument("--viz_suffix", default=".viz.png",
                    help="PNG suffix appended to JSON basename. Default: .viz.png")
    ap.add_argument("--viz_dpi", type=int, default=180)

    # guardrail
    ap.add_argument("--max_actions", type=int, default=300,
                    help="Fail if planned actions length exceeds this. Default: 300")

    args = ap.parse_args()

    meta_template = build_expected_meta_template(args)

    cfg: Dict[str, Any] = {
        "lookahead": float(args.lookahead),
        "horizon": int(args.horizon),
        "beam": int(args.beam),
        "step": float(args.step),
        "turn_deg": float(args.turn_deg),
        "stop_dist": float(args.stop_dist),
        "max_steps": int(args.max_steps),

        "relocalize": bool(args.relocalize),              # DEFAULT OFF
        "relocalize_thresh": float(args.relocalize_thresh),

        "init_yaw_eps": float(args.init_yaw_eps),
        "quantize_init_yaw": bool(args.quantize_init_yaw),

        "endgame_dist": float(args.endgame_dist),
        "endgame_turn_tol_deg": float(args.endgame_turn_tol_deg),

        "w_stop_good": float(args.w_stop_good),
        "w_stop_bad": float(args.w_stop_bad),

        "w_step": float(args.w_step),
        "w_turn": float(args.w_turn),
        "w_perp": float(args.w_perp),
        "d0": float(args.d0),
        "w_head": float(args.w_head),
        "w_head_tangent": float(args.w_head_tangent),
        "w_switch": float(args.w_switch),
        "w_terminal": float(args.w_terminal),
        "w_progress": float(args.w_progress),
        "w_back": float(args.w_back),

        "w_goal_heur": float(args.w_goal_heur),
        "w_spin": float(args.w_spin),
        "turn_slack": int(args.turn_slack),
        "commit": int(args.commit),
        "stall_steps": int(args.stall_steps),
        "stall_ds_eps": float(args.stall_ds_eps),

        "smooth_mode": str(args.smooth),
        "smooth_win": int(args.smooth_win),
        "smooth_kind": str(args.smooth_kind),

        "max_actions": int(args.max_actions),

        "skip_if_present": bool(args.skip_if_present),
        "meta_template": meta_template,
        "force": bool(args.force),
    }

    max_inflight = int(args.max_inflight)
    if max_inflight <= 0:
        max_inflight = max(4, 4 * int(args.workers))

    print(
        f"[START] root={args.root} workers={args.workers} max_inflight={max_inflight} "
        f"dry_run={bool(args.dry_run)} force={bool(args.force)} relocalize={bool(args.relocalize)}",
        file=sys.stderr, flush=True
    )

    ok = 0
    skip = 0
    fail = 0
    done = 0
    submitted = 0

    t0 = time.time()
    last_log_t = t0

    file_iter = iter_scene_label_paths_json_stream(args.root)

    def submit_one(ex: cf.ProcessPoolExecutor, inflight: Dict[cf.Future, str]) -> bool:
        nonlocal submitted
        try:
            fp = next(file_iter)  # may raise StopIteration
        except StopIteration:
            return False
        fut = ex.submit(worker_compute, fp, cfg, args.field, bool(args.restore_backup_on_read_fail))
        inflight[fut] = fp
        submitted += 1
        return True

    with cf.ProcessPoolExecutor(max_workers=args.workers) as ex:
        inflight: Dict[cf.Future, str] = {}

        # prime
        while len(inflight) < max_inflight and submit_one(ex, inflight):
            pass

        # main loop
        while inflight:
            done_set, _ = cf.wait(inflight.keys(), return_when=cf.FIRST_COMPLETED)

            for fut in done_set:
                inflight.pop(fut, None)
                fp, status, actions, smooth_list, init_yaw_rad, pid, err = fut.result()
                done += 1

                if status == "skip":
                    skip += 1
                elif err is not None or actions is None or smooth_list is None or init_yaw_rad is None:
                    fail += 1
                    print(f"[FAIL][PID {pid}] {fp}: {err}", file=sys.stderr, flush=True)
                else:
                    # hard guarantee: no STOP in output
                    actions = _strip_stop(list(actions))

                    if args.dry_run:
                        ok += 1
                        print(
                            f"[DRY][PID {pid}] {fp} -> {args.field}[len={len(actions)}], "
                            f"raster_world_smooth[n={len(smooth_list)}] init_yaw={init_yaw_rad:.6f} "
                            f"({rad2deg(init_yaw_rad):.1f}deg)",
                            flush=True
                        )
                    else:
                        try:
                            # read latest
                            with open(fp, "r", encoding="utf-8") as f:
                                data = json.load(f)

                            # write outputs
                            data[args.field] = actions

                            if "path" not in data or not isinstance(data["path"], dict):
                                data["path"] = {}
                            data["path"]["raster_world_smooth"] = smooth_list
                            data["path"]["raster_world_smooth_meta"] = {
                                "mode": args.smooth,
                                "win": int(args.smooth_win),
                                "kind": args.smooth_kind,
                                "note": "smoothed version of path.raster_world used for planning",
                                "relocalize": bool(args.relocalize),
                                "init_yaw": "start_to_first_smooth",
                                "init_yaw_eps": float(args.init_yaw_eps),
                                "quantize_init_yaw": bool(args.quantize_init_yaw),
                                "init_yaw_rad": float(init_yaw_rad),
                                "planner": {
                                    "horizon": int(args.horizon),
                                    "beam": int(args.beam),
                                    "lookahead": float(args.lookahead),
                                    "step": float(args.step),
                                    "turn_deg": float(args.turn_deg),
                                    "stop_dist": float(args.stop_dist),
                                    "stop_shaping": {
                                        "w_stop_good": float(args.w_stop_good),
                                        "w_stop_bad": float(args.w_stop_bad),
                                    },
                                },
                                "endgame": {
                                    "endgame_dist": float(args.endgame_dist),
                                    "endgame_turn_tol_deg": float(args.endgame_turn_tol_deg),
                                },
                            }

                            # backup if missing
                            if args.backup_if_missing:
                                bak = fp + ".bak"
                                if not os.path.isfile(bak):
                                    shutil.copy2(fp, bak)

                            atomic_write_json(fp, data, indent=args.indent)
                            ok += 1

                            if args.viz:
                                raw_pts = parse_raster_world(data)
                                start_xy = parse_start_world(data)
                                smooth_pts = _smooth_list_to_pts(smooth_list)
                                poly = Polyline2D(smooth_pts)

                                traj_xy = rollout_actions(
                                    actions=data[args.field],
                                    start_xy=start_xy,
                                    th0=float(init_yaw_rad),
                                    step_m=float(args.step),
                                    turn_deg=float(args.turn_deg),
                                    poly_for_snap=poly,
                                    relocalize=bool(args.relocalize),
                                    relocalize_thresh=float(args.relocalize_thresh),
                                )

                                base = os.path.splitext(os.path.basename(fp))[0]
                                out_png = os.path.join(os.path.dirname(fp), base + str(args.viz_suffix))
                                title = (
                                    f"rollout(relocalize={bool(args.relocalize)}) "
                                    f"| H={args.horizon} B={args.beam} LA={args.lookahead} "
                                    f"| yaw={rad2deg(init_yaw_rad):.1f}deg"
                                )
                                save_viz_png(
                                    out_png=out_png,
                                    raw_pts=raw_pts,
                                    smooth_pts=smooth_pts,
                                    traj_xy=traj_xy,
                                    start_xy=start_xy,
                                    init_yaw_rad=float(init_yaw_rad),
                                    title=title,
                                    dpi=int(args.viz_dpi),
                                )

                        except Exception as e:
                            fail += 1
                            print(f"[FAIL][PID {pid}] {fp}: write/viz error: {e}", file=sys.stderr, flush=True)

                # count-based log
                if args.log_every > 0 and (done % int(args.log_every) == 0):
                    dt = time.time() - t0
                    rate = done / max(dt, 1e-9)
                    print(
                        f"[PROG] submitted={submitted} done={done} ok={ok} skip={skip} fail={fail} "
                        f"inflight={len(inflight)} rate={rate:.2f} files/s dt={dt:.1f}s",
                        file=sys.stderr, flush=True
                    )

                # time-based log
                now = time.time()
                if args.log_secs > 0 and (now - last_log_t) >= float(args.log_secs):
                    dt = now - t0
                    rate = done / max(dt, 1e-9)
                    print(
                        f"[PROG][T] submitted={submitted} done={done} ok={ok} skip={skip} fail={fail} "
                        f"inflight={len(inflight)} rate={rate:.2f} files/s dt={dt:.1f}s",
                        file=sys.stderr, flush=True
                    )
                    last_log_t = now

                # refill
                while len(inflight) < max_inflight and submit_one(ex, inflight):
                    pass

    dt = time.time() - t0
    print(
        f"[DONE] ok={ok} skip={skip} fail={fail} workers={args.workers} max_inflight={max_inflight} "
        f"dt={dt:.1f}s root={args.root}",
        file=sys.stderr, flush=True
    )
    return 0 if fail == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
