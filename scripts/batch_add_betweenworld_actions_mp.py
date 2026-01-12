#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Batch compute actions for followingData JSONs (NO relocalize/snapping at all).

Input per frame:
- frame["between_world"] : polyline points [{x,y}, ...] or [[x,y], ...] (>=2)
- frame["camera_world"]  : start point {x,y} or [x,y]

What it does (per frame):
1) Build path = [start(camera_world)] + between_world (merge if already near start),
   then smooth it -> smooth_pts (optional)
2) Initial yaw: from smooth_pts[0] -> smooth_pts[1] (scan forward if degenerate),
   optional quantization to turn grid (turn_deg).
3) Receding-horizon beam-search MPC with actions:
     STOP=0, FWD=1 (step), LEFT=2, RIGHT=3 (turn_deg)
   - STOP may be considered during planning
   - BUT final written action sequence NEVER contains STOP:
       if STOP chosen => terminate WITHOUT writing 0
       if within stop_dist => terminate WITHOUT writing 0
4) Write per-frame actions to frame[--field] (default: "between_actions")
5) Optionally write per-frame smoothed polyline to:
     frame["between_world_smooth"] and frame["between_world_smooth_meta"]

Resume / skip:
- Writes top-level meta to JSON: data[f"{field}_meta"]
- If meta matches current args and all frames already contain outputs (and smooth if requested),
  skip the JSON. If some frames missing, only compute missing frames.

NOTE:
- relocalize/snapping has been fully removed. Planner and visualization are pure kinematics.
- Progress logging:
  * file-level logging: --log_every
  * time-based heartbeat: --log_secs (prints even if no file finished yet)

Usage:
  # visualize-only: save first N frame plots quickly
  python -u batch_add_betweenworld_actions_mp.py \
    /path/to/followingData \
    --viz_dir /tmp/viz_bw --viz_n 100 --viz_only --workers 16 --max_inflight 64 \
    --stop_dist 0.15 --lookahead 1.5

  # full write-back (resume enabled by default)
  python -u batch_add_betweenworld_actions_mp.py \
    /path/to/followingData \
    --write_smooth --backup_if_missing --workers 32 --max_inflight 128 \
    --stop_dist 0.15 --lookahead 1.5

  # FORCE recompute (overwrite everything, ignore existing outputs)
  python -u batch_add_betweenworld_actions_mp.py \
    /path/to/followingData \
    --force --no_skip_same_config \
    --stop_dist 0.15 --lookahead 1.5
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import math
import os
import re
import shutil
import sys
import time
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np

# matplotlib for visualization (Agg for headless)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ======= VLN-CE / Habitat-style action ids =======
STOP = 0
FWD  = 1
LEFT = 2
RIGHT = 3


def wrap_pi(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi


def deg2rad(d: float) -> float:
    return d * math.pi / 180.0


def rad2deg(r: float) -> float:
    return r * 180.0 / math.pi


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


# ========= Polyline =========

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


# ========= Kinematics =========

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


# ========= Build smoothed path INCLUDING start + initial yaw from p0->p1 =========

def build_smooth_path_with_start(
    raw_pts: np.ndarray,
    start_xy: Tuple[float, float],
    mode: str,
    win: int,
    kind: str,
    merge_eps: float = 1e-3,
) -> np.ndarray:
    """
    Build smoothed path that INCLUDES start point as index 0.
    If raw_pts[0] is already very close to start, replace raw_pts[0] with start (avoid duplicate).
    """
    if raw_pts.ndim != 2 or raw_pts.shape[1] != 2 or len(raw_pts) < 2:
        raise ValueError("raw_pts must be (N,2) with N>=2")

    start = np.asarray([float(start_xy[0]), float(start_xy[1])], dtype=np.float64)

    d0 = float(np.hypot(raw_pts[0, 0] - start[0], raw_pts[0, 1] - start[1]))
    if d0 <= float(merge_eps):
        pts = raw_pts.astype(np.float64).copy()
        pts[0] = start
    else:
        pts = np.vstack([start[None, :], raw_pts.astype(np.float64)])

    pts_s = smooth_polyline(pts, mode=str(mode), win=int(win), kind=str(kind))

    # hard guarantee: first point equals start
    pts_s = pts_s.astype(np.float64)
    pts_s[0] = start
    return pts_s


def initial_yaw_from_smooth_0_to_1(
    smooth_pts: np.ndarray,
    turn_deg: float,
    eps: float = 1e-6,
    quantize: bool = True,
) -> float:
    """
    Initial yaw = direction from smooth_pts[0] to smooth_pts[1].
    If the first segment is too short, scan forward to find the first non-degenerate segment.
    """
    if smooth_pts.ndim != 2 or smooth_pts.shape[1] != 2 or len(smooth_pts) < 2:
        return 0.0

    th = 0.0
    found = False
    for i in range(len(smooth_pts) - 1):
        dx = float(smooth_pts[i + 1, 0] - smooth_pts[i, 0])
        dy = float(smooth_pts[i + 1, 1] - smooth_pts[i, 1])
        if (dx * dx + dy * dy) > float(eps) * float(eps):
            th = math.atan2(dy, dx)
            found = True
            break

    if not found:
        th = 0.0

    if quantize:
        q = deg2rad(float(turn_deg))
        if q > 0:
            th = q * round(th / q)

    return wrap_pi(th)


# ========= Planner (STOP can be considered, but output MUST contain NO STOP) =========

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
    if not actions:
        return []
    if STOP in actions:
        actions = actions[:actions.index(STOP)]
    return [int(a) for a in actions if int(a) != STOP]


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

    # stage weights
    w_step: float,
    w_turn: float,
    w_perp: float,
    d0: float,
    w_head: float,
    w_head_tangent: float,
    w_switch: float,
    w_terminal: float,
    w_progress: float,
    w_back: float,

    # pruning / anti-dither
    w_goal_heur: float,
    w_spin: float,
    turn_slack: int,
    commit: int,
    stall_steps: int,
    stall_ds_eps: float,

    # STOP shaping
    w_stop_good: float,
    w_stop_bad: float,
) -> List[int]:
    turn_rad = deg2rad(turn_deg)

    x, y, th = float(x0), float(y0), float(th0)
    _proj0, s, _ = poly.project(np.array([x, y], dtype=np.float64))

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
        if dist_to_end(x, y) <= goal_stop_m:
            break

        eff_lookahead = min(float(lookahead_m), float(horizon) * float(step_m))

        _proj_now, s_now, _ = poly.project(np.array([x, y], dtype=np.float64))
        s = s_now
        s_goal = min(s + eff_lookahead, poly.total_length)
        gpt = poly.point_at_s(s_goal)
        gx, gy = float(gpt[0]), float(gpt[1])

        beam_nodes: List[Node] = [Node(g=0.0, f=0.0, x=x, y=y, th=th, s=s, actions=[])]
        for _depth in range(int(horizon)):
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
            beam_nodes = cand[:int(beam)]

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

        if stall_ctr >= int(stall_steps):
            plan = [FWD]
            stall_ctr = 0

        kmax = max(1, int(commit))
        for k in range(min(kmax, len(plan))):
            a0 = int(plan[k])

            if (FWD not in plan) and (STOP not in plan) and (a0 in (LEFT, RIGHT)):
                a0 = FWD

            if a0 == STOP:
                terminated = True
                break

            x, y, th = simulate_step(x, y, th, a0, step_m, turn_rad)
            if a0 in (LEFT, RIGHT):
                prev_turn = a0

            _proj_exec, s_exec, _ = poly.project(np.array([x, y], dtype=np.float64))
            s = s_exec

            actions.append(a0)
            executed += 1

            ds_exec = s - s_last_exec
            if ds_exec > float(stall_ds_eps):
                stall_ctr = 0
            else:
                stall_ctr += 1
            s_last_exec = s

            if dist_to_end(x, y) <= goal_stop_m:
                terminated = True
                break

            if executed >= int(max_steps):
                break
            if a0 == FWD:
                break

    return _strip_stop(actions)


# ========= Pure rollout for visualization =========

def rollout_actions_pure(
    start_xy: Tuple[float, float],
    th0: float,
    actions: List[int],
    step_m: float,
    turn_deg: float,
) -> np.ndarray:
    turn_rad = deg2rad(turn_deg)
    x, y, th = float(start_xy[0]), float(start_xy[1]), float(th0)
    traj = [(x, y)]
    for a in actions:
        if int(a) == STOP:
            break
        x, y, th = simulate_step(x, y, th, int(a), step_m, turn_rad)
        traj.append((x, y))
    return np.asarray(traj, dtype=np.float64)


# ========= IO helpers =========

def atomic_write_json(path: str, obj: Dict[str, Any], indent: Optional[int] = None) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=indent)
    os.replace(tmp, path)


def iter_json_recursive_iter(root: str,
                            suffix: str = ".json",
                            exclude_suffixes: Tuple[str, ...] = (".bak", ".tmp")) -> Iterable[str]:
    if not os.path.isdir(root):
        return
    suf = suffix.lower()
    for dirpath, _dirnames, filenames in os.walk(root):
        for fn in filenames:
            lf = fn.lower()
            if any(lf.endswith(es) for es in exclude_suffixes):
                continue
            if suf and (not lf.endswith(suf)):
                continue
            yield os.path.join(dirpath, fn)


def parse_xy(pt: Any) -> Tuple[float, float]:
    if isinstance(pt, dict) and ("x" in pt) and ("y" in pt):
        return float(pt["x"]), float(pt["y"])
    if isinstance(pt, (list, tuple)) and len(pt) >= 2:
        return float(pt[0]), float(pt[1])
    raise ValueError(f"Unsupported point format: {type(pt)}")


def parse_between_world(frame: Dict[str, Any], key: str) -> np.ndarray:
    bw = frame.get(key, None)
    if bw is None or (not isinstance(bw, list)) or len(bw) < 2:
        raise ValueError(f"Missing/invalid frame.{key} (need list with >=2 points).")
    pts = [parse_xy(p) for p in bw]
    return np.asarray(pts, dtype=np.float64)


def parse_camera_world(frame: Dict[str, Any]) -> Tuple[float, float]:
    cw = frame.get("camera_world", None)
    if cw is None:
        raise ValueError("Missing frame.camera_world.")
    return parse_xy(cw)


def sanitize_filename(s: str, maxlen: int = 200) -> str:
    s = s.replace(os.sep, "_")
    s = re.sub(r"[^a-zA-Z0-9._-]+", "_", s)
    if len(s) > maxlen:
        s = s[:maxlen]
    return s


def restore_from_bak(fp: str) -> bool:
    bak = fp + ".bak"
    if os.path.isfile(bak):
        shutil.copy2(bak, fp)
        return True
    return False


# ========= Meta compare =========

def _is_number(x: Any) -> bool:
    return isinstance(x, (int, float)) and (not isinstance(x, bool))


def meta_equal(a: Any, b: Any, tol: float = 1e-9) -> bool:
    if a is b:
        return True
    if a is None or b is None:
        return a is None and b is None

    if _is_number(a) and _is_number(b):
        return math.isclose(float(a), float(b), rel_tol=tol, abs_tol=tol)

    if isinstance(a, str) and isinstance(b, str):
        return a == b
    if isinstance(a, bool) and isinstance(b, bool):
        return a == b

    if isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            return False
        return all(meta_equal(x, y, tol=tol) for x, y in zip(a, b))

    if isinstance(a, dict) and isinstance(b, dict):
        if set(a.keys()) != set(b.keys()):
            return False
        for k in a.keys():
            if not meta_equal(a[k], b[k], tol=tol):
                return False
        return True

    return a == b


def build_run_meta(args: argparse.Namespace) -> Dict[str, Any]:
    # format_version bump ensures older runs won't match meta
    return {
        "format_version": 3,
        "between_key": str(args.between_key),
        "field": str(args.field),
        "write_smooth": bool(args.write_smooth),
        "smooth": {
            "mode": str(args.smooth),
            "win": int(args.smooth_win),
            "kind": str(args.smooth_kind),
            "note": "smooth is applied on [camera_world(start)] + between_world (merged if already near start)",
        },
        "init_yaw": {
            "method": "smooth0_to_next",
            "eps": float(args.init_yaw_eps),
            "quantize": bool(not args.no_quantize_init_yaw),
            "quantize_deg": float(args.turn_deg),
        },
        "planner": {
            "lookahead": float(args.lookahead),
            "horizon": int(args.horizon),
            "beam": int(args.beam),
            "step": float(args.step),
            "turn_deg": float(args.turn_deg),
            "stop_dist": float(args.stop_dist),
            "max_steps": int(args.max_steps),
            "stop_shaping": {
                "w_stop_good": float(args.w_stop_good),
                "w_stop_bad": float(args.w_stop_bad),
            },
        },
        "weights": {
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
        },
        "anti_dither": {
            "w_goal_heur": float(args.w_goal_heur),
            "w_spin": float(args.w_spin),
            "turn_slack": int(args.turn_slack),
            "commit": int(args.commit),
            "stall_steps": int(args.stall_steps),
            "stall_ds_eps": float(args.stall_ds_eps),
        },
        "limits": {
            "max_actions": int(args.max_actions),
        },
        "note": "NO relocalize/snapping anywhere; STOP may be planned but will never be written.",
    }


# ========= Visualization =========

def compute_traj_path_errors(poly: Polyline2D, traj_xy: np.ndarray) -> Tuple[float, float]:
    ds = []
    for i in range(traj_xy.shape[0]):
        _, _, d = poly.project(traj_xy[i])
        ds.append(d)
    ds = np.asarray(ds, dtype=np.float64)
    rms = float(np.sqrt(np.mean(ds * ds))) if len(ds) else 0.0
    dmax = float(np.max(ds)) if len(ds) else 0.0
    return rms, dmax


def plot_compare(raw_pts: np.ndarray, smooth_pts: np.ndarray, traj_xy_pure: np.ndarray, out_path: str,
                 start_xy: Tuple[float, float],
                 title: str) -> None:
    plt.figure()
    plt.plot(raw_pts[:, 0], raw_pts[:, 1],
             linewidth=1.2, linestyle="--", alpha=0.7,
             label="reference raw (between_world)")
    plt.plot(smooth_pts[:, 0], smooth_pts[:, 1],
             linewidth=2.0,
             label="reference smoothed (includes start as p0)")
    plt.plot(traj_xy_pure[:, 0], traj_xy_pure[:, 1],
             marker="o", markersize=3, linewidth=1.5,
             label="PURE rollout (from discrete actions)")

    plt.scatter([start_xy[0]], [start_xy[1]], marker="s", s=60, label="start (camera_world)")
    plt.scatter([raw_pts[-1, 0]], [raw_pts[-1, 1]], marker="*", s=120, label="ref end (between_world[-1])")

    plt.axis("equal")
    plt.grid(True)
    plt.legend()
    plt.title(title)
    plt.xlabel("x (m)")
    plt.ylabel("y (m)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def visualize_first_n_frames(root_files: Iterable[str],
                             root_dir: str,
                             cfg: Dict[str, Any],
                             between_key: str,
                             viz_dir: str,
                             viz_n: int,
                             try_restore_on_fail: bool,
                             viz_log_every: int,
                             verbose: bool = True) -> int:
    os.makedirs(viz_dir, exist_ok=True)
    saved = 0
    tried = 0
    fail_ctr = 0
    last_fail = ""
    t0 = time.time()

    for fp in root_files:
        try:
            try:
                with open(fp, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except Exception:
                if try_restore_on_fail and restore_from_bak(fp):
                    with open(fp, "r", encoding="utf-8") as f:
                        data = json.load(f)
                else:
                    raise

            frames = data.get("frames", None)
            if frames is None or not isinstance(frames, list):
                continue

            scene = str(data.get("scene", "scene"))
            label = str(data.get("label", "label"))

            for fr in frames:
                if saved >= viz_n:
                    if verbose:
                        dt = time.time() - t0
                        print(f"[VIZ][DONE] saved={saved}/{viz_n} tried={tried} fails={fail_ctr} dt={dt:.1f}s dir={viz_dir}",
                              file=sys.stderr, flush=True)
                    return saved

                tried += 1
                if not isinstance(fr, dict):
                    fail_ctr += 1
                    last_fail = "frame not dict"
                    continue

                fid = fr.get("id", None)
                fid_str = str(fid) if fid is not None else "na"

                try:
                    raw_pts = parse_between_world(fr, between_key)
                    start_xy = parse_camera_world(fr)

                    smooth_pts = build_smooth_path_with_start(
                        raw_pts=raw_pts,
                        start_xy=start_xy,
                        mode=cfg["smooth_mode"],
                        win=cfg["smooth_win"],
                        kind=cfg["smooth_kind"],
                        merge_eps=cfg["init_yaw_eps"],
                    )
                    poly = Polyline2D(smooth_pts)

                    th0 = initial_yaw_from_smooth_0_to_1(
                        smooth_pts=smooth_pts,
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

                        w_stop_good=cfg["w_stop_good"],
                        w_stop_bad=cfg["w_stop_bad"],
                    )

                    traj_pure = rollout_actions_pure(
                        start_xy=start_xy,
                        th0=th0,
                        actions=actions,
                        step_m=cfg["step"],
                        turn_deg=cfg["turn_deg"],
                    )

                    rms_s, dmax_s = compute_traj_path_errors(poly, traj_pure)
                    end_xy = raw_pts[-1]
                    final_xy = traj_pure[-1]
                    d_end = float(np.hypot(final_xy[0] - float(end_xy[0]), final_xy[1] - float(end_xy[1])))

                    rel = sanitize_filename(os.path.relpath(fp, start=root_dir) if os.path.isabs(fp) else fp)
                    out_name = f"{saved:04d}_{scene}_{label}_frame{fid_str}_{rel}.png"
                    out_name = sanitize_filename(out_name, maxlen=220)
                    out_path = os.path.join(viz_dir, out_name)

                    title = (f"{scene}/{label} frame={fid_str} "
                             f"LA={cfg['lookahead']:.2f} stop={cfg['stop_dist']:.2f} "
                             f"acts={len(actions)} end_dist={d_end:.3f} RMS={rms_s:.3f} Max={dmax_s:.3f}")

                    plot_compare(raw_pts, smooth_pts, traj_pure, out_path, start_xy=start_xy, title=title)

                    saved += 1
                    if verbose:
                        dt = time.time() - t0
                        print(f"[VIZ] saved={saved}/{viz_n} tried={tried} fails={fail_ctr} dt={dt:.1f}s -> {out_path}",
                              file=sys.stderr, flush=True)

                except Exception as e:
                    fail_ctr += 1
                    last_fail = str(e)
                    if verbose and viz_log_every > 0 and (tried % int(viz_log_every) == 0):
                        dt = time.time() - t0
                        print(f"[VIZ][INFO] tried={tried} saved={saved} fails={fail_ctr} dt={dt:.1f}s last_fail={last_fail}",
                              file=sys.stderr, flush=True)
                    continue

        except Exception as e:
            fail_ctr += 1
            last_fail = str(e)
            continue

    if verbose:
        dt = time.time() - t0
        print(f"[VIZ][DONE] saved={saved}/{viz_n} tried={tried} fails={fail_ctr} dt={dt:.1f}s dir={viz_dir} last_fail={last_fail}",
              file=sys.stderr, flush=True)
    return saved


# ========= Worker compute =========

def worker_compute_file(fp: str,
                        cfg: Dict[str, Any],
                        between_key: str,
                        field: str,
                        write_smooth: bool,
                        meta_key: str,
                        run_meta: Dict[str, Any],
                        enable_skip: bool,
                        try_restore_on_fail: bool) -> Tuple[str, str, Optional[List[Dict[str, Any]]], int, Optional[str]]:
    """
    Returns: (fp, status, per_frame, pid, err)
      status in {"ok", "skip"}
    per_frame items:
      - computed frame: {"index": i, "ok": True, "actions": [...], "smooth": [...]?}
      - skipped frame:  {"index": i, "ok": True, "skip": True}
      - failed frame:   {"index": i, "ok": False, "err": "..."}
    """
    pid = os.getpid()
    try:
        try:
            with open(fp, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e0:
            if try_restore_on_fail and restore_from_bak(fp):
                with open(fp, "r", encoding="utf-8") as f:
                    data = json.load(f)
            else:
                raise e0

        frames = data.get("frames", None)
        if frames is None or not isinstance(frames, list) or len(frames) == 0:
            raise ValueError("Missing/invalid top-level 'frames'.")

        existing_meta = data.get(meta_key, None)
        meta_ok = (existing_meta is not None) and meta_equal(existing_meta, run_meta)

        # file-level skip if meta matches and all frames done
        if enable_skip and meta_ok:
            all_done = True
            for fr in frames:
                if not isinstance(fr, dict):
                    continue
                if (between_key in fr) and ("camera_world" in fr):
                    if field not in fr or not isinstance(fr.get(field), list):
                        all_done = False
                        break
                    if any(int(a) == STOP for a in fr.get(field, [])):
                        all_done = False
                        break
                    if write_smooth:
                        if ("between_world_smooth" not in fr) or (not isinstance(fr.get("between_world_smooth"), list)):
                            all_done = False
                            break
                        if ("between_world_smooth_meta" not in fr) or (not isinstance(fr.get("between_world_smooth_meta"), dict)):
                            all_done = False
                            break
            if all_done:
                return fp, "skip", None, pid, None

        out: List[Dict[str, Any]] = []

        for idx, fr in enumerate(frames):
            if not isinstance(fr, dict):
                out.append({"index": idx, "ok": False, "err": "frame not a dict"})
                continue

            # per-frame skip if meta matches and this frame already has what we need
            if enable_skip and meta_ok:
                if (between_key in fr) and ("camera_world" in fr):
                    has_actions = (field in fr) and isinstance(fr.get(field), list) and (not any(int(a) == STOP for a in fr.get(field, [])))
                    has_smooth = True
                    if write_smooth:
                        has_smooth = ("between_world_smooth" in fr) and isinstance(fr.get("between_world_smooth"), list) \
                                     and ("between_world_smooth_meta" in fr) and isinstance(fr.get("between_world_smooth_meta"), dict)
                    if has_actions and has_smooth:
                        out.append({"index": idx, "ok": True, "skip": True})
                        continue

            try:
                raw_pts = parse_between_world(fr, between_key)
                start_xy = parse_camera_world(fr)

                smooth_pts = build_smooth_path_with_start(
                    raw_pts=raw_pts,
                    start_xy=start_xy,
                    mode=cfg["smooth_mode"],
                    win=cfg["smooth_win"],
                    kind=cfg["smooth_kind"],
                    merge_eps=cfg["init_yaw_eps"],
                )
                poly = Polyline2D(smooth_pts)

                th0 = initial_yaw_from_smooth_0_to_1(
                    smooth_pts=smooth_pts,
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

                    w_stop_good=cfg["w_stop_good"],
                    w_stop_bad=cfg["w_stop_bad"],
                )

                actions = _strip_stop(list(actions))
                if len(actions) > int(cfg.get("max_actions", 300)):
                    raise ValueError(f"actions too long: len={len(actions)} > max_actions={cfg['max_actions']}")

                item: Dict[str, Any] = {
                    "index": idx,
                    "ok": True,
                    "actions": actions,  # NO STOP
                }
                if write_smooth:
                    item["smooth"] = [{"x": float(p[0]), "y": float(p[1])} for p in smooth_pts]
                out.append(item)

            except Exception as e:
                out.append({"index": idx, "ok": False, "err": str(e)})

        return fp, "ok", out, pid, None

    except Exception as e:
        return fp, "ok", None, pid, str(e)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("root", help="Root directory: .../followingData")
    ap.add_argument("--suffix", default=".json")

    ap.add_argument("--between_key", default="between_world")
    ap.add_argument("--field", default="between_actions")

    # skip/resume
    ap.add_argument("--force", action="store_true", help="Force recompute even if meta matches.")
    ap.add_argument("--no_skip_same_config", action="store_true",
                    help="Disable skip/resume based on meta match. Default: enabled.")

    # restore behavior
    ap.add_argument("--no_restore_backup", action="store_true",
                    help="Disable restoring from <file>.bak on read failure. Default: enabled on failure.")
    ap.add_argument("--backup_if_missing", action="store_true",
                    help="If <file>.bak does not exist, create it before writing. (Do NOT overwrite existing .bak)")

    # write options
    ap.add_argument("--dry_run", action="store_true")
    ap.add_argument("--indent", type=int, default=None, help="json indent when writing (optional).")

    # smoothing
    ap.add_argument("--smooth", choices=["none", "pos", "vel"], default="vel")
    ap.add_argument("--smooth_win", type=int, default=9)
    ap.add_argument("--smooth_kind", choices=["box", "tri", "gauss"], default="tri")
    ap.add_argument("--write_smooth", action="store_true",
                    help="Write per-frame smoothed polyline as frame['between_world_smooth'] + meta.")

    # initial yaw
    ap.add_argument("--init_yaw_eps", type=float, default=1e-3,
                    help="Used as (1) merge_eps when inserting start; (2) degenerate segment eps in yaw.")
    ap.add_argument("--no_quantize_init_yaw", action="store_true",
                    help="Disable quantizing initial yaw to nearest turn_deg. Default: quantize ON.")

    # MPC/planner params
    ap.add_argument("--lookahead", type=float, default=1.5)
    ap.add_argument("--horizon", type=int, default=12)
    ap.add_argument("--beam", type=int, default=200)
    ap.add_argument("--step", type=float, default=0.25)
    ap.add_argument("--turn_deg", type=float, default=15.0)
    ap.add_argument("--stop_dist", type=float, default=0.15)
    ap.add_argument("--max_steps", type=int, default=800)

    # STOP shaping (decision-only; output still NO STOP)
    ap.add_argument("--w_stop_good", type=float, default=-80.0,
                    help="Reward (negative cost) for STOP when within stop_dist.")
    ap.add_argument("--w_stop_bad", type=float, default=300.0,
                    help="Penalty for STOP when outside stop_dist.")

    # weights (tunable)
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

    ap.add_argument("--max_actions", type=int, default=80)

    # parallel scheduling
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 8) - 1))
    ap.add_argument("--max_inflight", type=int, default=0, help="0 => default(4*workers)")
    ap.add_argument("--log_every", type=int, default=200)
    ap.add_argument("--log_secs", type=float, default=10.0,
                    help="Time-based heartbeat: print progress every N seconds even if no file finished.")

    # visualization
    ap.add_argument("--viz_dir", default="", help="If set, save first N frame plots here (before full batch).")
    ap.add_argument("--viz_n", type=int, default=100)
    ap.add_argument("--viz_only", action="store_true",
                    help="Only generate visualization and exit (no write-back).")
    ap.add_argument("--viz_log_every", type=int, default=200,
                    help="Print viz failure stats every N tried frames (to avoid 'stuck' feeling).")

    args = ap.parse_args()

    try_restore_on_fail = (not args.no_restore_backup)

    cfg: Dict[str, Any] = {
        "lookahead": float(args.lookahead),
        "horizon": int(args.horizon),
        "beam": int(args.beam),
        "step": float(args.step),
        "turn_deg": float(args.turn_deg),
        "stop_dist": float(args.stop_dist),
        "max_steps": int(args.max_steps),

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

        "smooth_mode": args.smooth,
        "smooth_win": int(args.smooth_win),
        "smooth_kind": args.smooth_kind,

        "init_yaw_eps": float(args.init_yaw_eps),
        "quantize_init_yaw": bool(not args.no_quantize_init_yaw),

        "max_actions": int(args.max_actions),
    }

    enable_skip = (not args.no_skip_same_config) and (not args.force)
    meta_key = f"{args.field}_meta"
    run_meta = build_run_meta(args)

    # ---- Phase 1: visualization first (streaming) ----
    if args.viz_dir:
        n_saved = visualize_first_n_frames(
            root_files=iter_json_recursive_iter(args.root, suffix=args.suffix),
            root_dir=args.root,
            cfg=cfg,
            between_key=args.between_key,
            viz_dir=args.viz_dir,
            viz_n=int(args.viz_n),
            try_restore_on_fail=try_restore_on_fail,
            viz_log_every=int(args.viz_log_every),
            verbose=True,
        )
        print(f"[VIZ][DONE] saved={n_saved}/{args.viz_n} plots into: {args.viz_dir}", file=sys.stderr, flush=True)

        if args.viz_only:
            return 0

    if args.dry_run:
        print("[INFO] dry_run enabled: will compute but not write changes.", file=sys.stderr, flush=True)

    # ---- Phase 2: full batch compute + write-back ----
    ok = 0
    skip = 0
    fail = 0
    done = 0
    submitted = 0
    t0 = time.time()
    last_log_t = t0

    max_inflight = int(args.max_inflight)
    if max_inflight <= 0:
        max_inflight = max(4, 4 * int(args.workers))

    file_iter = iter_json_recursive_iter(args.root, suffix=args.suffix)

    def submit_one(ex: cf.ProcessPoolExecutor, inflight: Dict[cf.Future, str]) -> bool:
        nonlocal submitted
        try:
            fp = next(file_iter)
        except StopIteration:
            return False
        fut = ex.submit(
            worker_compute_file,
            fp,
            cfg,
            args.between_key,
            args.field,
            bool(args.write_smooth),
            meta_key,
            run_meta,
            bool(enable_skip),
            bool(try_restore_on_fail),
        )
        inflight[fut] = fp
        submitted += 1
        return True

    print(
        f"[START] root={args.root} workers={args.workers} max_inflight={max_inflight} "
        f"force={bool(args.force)} skip_same_config={bool(not args.no_skip_same_config)} "
        f"write_smooth={bool(args.write_smooth)}",
        file=sys.stderr, flush=True
    )

    with cf.ProcessPoolExecutor(max_workers=args.workers) as ex:
        inflight: Dict[cf.Future, str] = {}

        while len(inflight) < max_inflight and submit_one(ex, inflight):
            pass

        # heartbeat loop: use timeout so we can print even if nothing completes
        while inflight:
            done_set, _ = cf.wait(inflight.keys(), timeout=0.5, return_when=cf.FIRST_COMPLETED)

            if not done_set:
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
                continue

            for fut in done_set:
                inflight.pop(fut, None)
                fp, status, per_frame, pid, err = fut.result()
                done += 1

                if status == "skip":
                    skip += 1
                elif err is not None or per_frame is None:
                    fail += 1
                    print(f"[FAIL][PID {pid}] {fp}: {err}", file=sys.stderr, flush=True)
                else:
                    if args.dry_run:
                        ok += 1
                        ok_n = sum(1 for x in per_frame if x.get("ok"))
                        skip_n = sum(1 for x in per_frame if x.get("ok") and x.get("skip", False))
                        print(f"[DRY][PID {pid}] {fp} frames_ok={ok_n}/{len(per_frame)} (skipped_frames={skip_n}) -> field={args.field}",
                              file=sys.stderr, flush=True)
                    else:
                        try:
                            if args.backup_if_missing:
                                bak = fp + ".bak"
                                if not os.path.isfile(bak):
                                    shutil.copy2(fp, bak)

                            with open(fp, "r", encoding="utf-8") as f:
                                data = json.load(f)

                            frames = data.get("frames", None)
                            if frames is None or not isinstance(frames, list):
                                raise ValueError("write-time: invalid frames")

                            # write top-level meta
                            data[meta_key] = run_meta

                            for item in per_frame:
                                idx = int(item["index"])
                                if idx < 0 or idx >= len(frames):
                                    continue
                                if not isinstance(frames[idx], dict):
                                    continue
                                if not item.get("ok", False):
                                    continue
                                if item.get("skip", False):
                                    continue

                                frames[idx][args.field] = _strip_stop(list(item["actions"]))

                                if args.write_smooth and ("smooth" in item) and item["smooth"] is not None:
                                    frames[idx]["between_world_smooth"] = item["smooth"]
                                    frames[idx]["between_world_smooth_meta"] = {
                                        "mode": args.smooth,
                                        "win": int(args.smooth_win),
                                        "kind": args.smooth_kind,
                                        "note": f"smoothed version of [camera_world(start)] + frames[*].{args.between_key} used for planning",
                                        "init_yaw": {
                                            "method": "smooth0_to_next",
                                            "eps": float(args.init_yaw_eps),
                                            "quantize": bool(not args.no_quantize_init_yaw),
                                            "quantize_deg": float(args.turn_deg),
                                        },
                                        "planner": {
                                            "lookahead": float(args.lookahead),
                                            "horizon": int(args.horizon),
                                            "beam": int(args.beam),
                                            "step": float(args.step),
                                            "turn_deg": float(args.turn_deg),
                                            "stop_dist": float(args.stop_dist),
                                            "stop_shaping": {
                                                "w_stop_good": float(args.w_stop_good),
                                                "w_stop_bad": float(args.w_stop_bad),
                                            },
                                        },
                                        "note2": "NO relocalize/snapping; STOP never written.",
                                    }

                            atomic_write_json(fp, data, indent=args.indent)
                            ok += 1

                        except Exception as e:
                            fail += 1
                            print(f"[FAIL][PID {pid}] {fp}: write error: {e}", file=sys.stderr, flush=True)

                if args.log_every > 0 and (done % int(args.log_every) == 0):
                    now = time.time()
                    dt = now - t0
                    rate = done / max(dt, 1e-9)
                    print(
                        f"[PROG] submitted={submitted} done={done} ok={ok} skip={skip} fail={fail} "
                        f"inflight={len(inflight)} rate={rate:.2f} files/s dt={dt:.1f}s",
                        file=sys.stderr, flush=True
                    )
                    last_log_t = now

                while len(inflight) < max_inflight and submit_one(ex, inflight):
                    pass

    dt = time.time() - t0
    print(f"[DONE] submitted={submitted} done={done} ok={ok} skip={skip} fail={fail} "
          f"workers={args.workers} max_inflight={max_inflight} dt={dt:.1f}s root={args.root}",
          file=sys.stderr, flush=True)

    return 0 if fail == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
