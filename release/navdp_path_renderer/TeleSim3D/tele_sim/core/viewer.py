"""Viewer scaffolding for the TeleSim3D debug camera."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Dict, Optional, Protocol, Sequence, Tuple

import numpy as np


Vec3 = Tuple[float, float, float]


@dataclass(frozen=True)
class Pose:
    """Simple pose container using world-space coordinates."""

    position: Vec3
    orientation: Tuple[float, float, float, float]


@dataclass(frozen=True)
class ControlIntent:
    """Continuous movement command in local-space units per second."""

    move: Vec3 = (0.0, 0.0, 0.0)
    turn: float = 0.0
    pitch: float = 0.0
    roll: float = 0.0


@dataclass(frozen=True)
class DebugFrame:
    """Bundle describing a rendered frame for the verifier UI."""

    rgb: np.ndarray
    bev: np.ndarray
    pose: Pose
    frame_index: int
    frame_time: float


class RendererBackend(Protocol):
    """Bridge into the splatting renderer.

    Implementations may manage GPU state or external processes. The protocol stays
    minimal so the viewer can be unit tested with fakes.
    """

    def render_rgb(self, pose: Pose) -> np.ndarray:
        ...

    def render_bev(self, pose: Pose) -> np.ndarray:
        ...

    def shutdown(self) -> None:
        ...


class PoseIntegrator(Protocol):
    """Updates the simulated camera pose based on the supplied intent."""

    def step(self, intent: ControlIntent, delta_seconds: float) -> Pose:
        ...

    def set_pose(self, pose: Pose) -> None:
        ...

    def current_pose(self) -> Pose:
        ...


class InputProvider(Protocol):
    """Supplies movement intents gathered from user input devices."""

    def sample_intent(self) -> ControlIntent:
        ...


@dataclass
class ViewerConfig:
    """Runtime configuration for the viewer loop."""

    target_fps: float = 5.0
    max_frame_drop: int = 2
    input_deadzone: float = 1e-3

    def frame_period(self) -> float:
        if self.target_fps <= 0:
            raise ValueError("target_fps must be positive")
        return 1.0 / self.target_fps


class DebugViewer:
    """Coordinates navigation, rendering, and frame pacing for debugging."""

    def __init__(
        self,
        renderer: RendererBackend,
        integrator: PoseIntegrator,
        input_provider: InputProvider,
        config: Optional[ViewerConfig] = None,
    ) -> None:
        self._renderer = renderer
        self._integrator = integrator
        self._input_provider = input_provider
        self._config = config or ViewerConfig()
        self._frame_index = 0
        self._last_pose: Optional[Pose] = None
        self._lock = threading.Lock()

    def reset(self, pose: Pose) -> None:
        with self._lock:
            self._frame_index = 0
            self._last_pose = pose
            self._integrator.set_pose(pose)

    def step(self, now: Optional[float] = None) -> DebugFrame:
        """Advance the viewer loop once and return the rendered frame."""

        with self._lock:
            start = now if now is not None else time.monotonic()
            period = self._config.frame_period()
            intent = self._input_provider.sample_intent()
            if (
                abs(intent.move[0])
                + abs(intent.move[1])
                + abs(intent.move[2])
                + abs(intent.turn)
                + abs(intent.pitch)
                + abs(intent.roll)
            ) < self._config.input_deadzone:
                intent = ControlIntent()
            pose = self._integrator.step(intent, period)
            rgb = self._renderer.render_rgb(pose)
            bev = self._renderer.render_bev(pose)
            frame_time = time.monotonic() - start
            self._frame_index += 1
            self._last_pose = pose
            sleep_time = period - frame_time
        if sleep_time > 0:
            time.sleep(sleep_time)
        return DebugFrame(
            rgb=rgb,
            bev=bev,
            pose=pose,
            frame_index=self._frame_index,
            frame_time=frame_time,
        )

    def close(self) -> None:
        with self._lock:
            self._renderer.shutdown()

    def teleport(self, pose: Pose) -> None:
        with self._lock:
            self._integrator.set_pose(pose)
            self._last_pose = pose


class KeyboardInput(InputProvider):
    """Placeholder keyboard mapper; hook into an actual event system later."""

    def __init__(self) -> None:
        self._bindings: Dict[str, Vec3] = {
            "w": (0.0, 0.0, 1.0),
            "s": (0.0, 0.0, -1.0),
            "a": (-1.0, 0.0, 0.0),
            "d": (1.0, 0.0, 0.0),
            "r": (0.0, 1.0, 0.0),
            "f": (0.0, -1.0, 0.0),
        }
        self._turn_bindings: Dict[str, float] = {"q": 1.0, "e": -1.0}
        self._pitch_bindings: Dict[str, float] = {"t": 1.0, "g": -1.0}
        self._roll_bindings: Dict[str, float] = {"z": -1.0, "c": 1.0}
        keys = (
            set(self._bindings.keys())
            | set(self._turn_bindings.keys())
            | set(self._pitch_bindings.keys())
            | set(self._roll_bindings.keys())
        )
        self._active: Dict[str, bool] = {key: False for key in keys}
        self._mirror = False
        self._lock = threading.Lock()

    def set_key_state(self, key: str, pressed: bool) -> None:
        with self._lock:
            key_lower = key.lower()
            if key_lower in self._active:
                self._active[key_lower] = pressed

    def set_mirror(self, enabled: bool) -> None:
        with self._lock:
            self._mirror = enabled

    def sample_intent(self) -> ControlIntent:
        with self._lock:
            move = np.zeros(3, dtype=np.float32)
            for key, vec in self._bindings.items():
                if self._active.get(key, False):
                    move += np.array(vec, dtype=np.float32)
            turn = 0.0
            for key, value in self._turn_bindings.items():
                if self._active.get(key, False):
                    turn += value
            pitch = 0.0
            for key, value in self._pitch_bindings.items():
                if self._active.get(key, False):
                    pitch += value
            roll = 0.0
            for key, value in self._roll_bindings.items():
                if self._active.get(key, False):
                    roll += value
            if self._mirror:
                move *= -1.0
                turn *= -1.0
                pitch *= -1.0
                roll *= -1.0
        return ControlIntent(move=tuple(move.tolist()), turn=turn, pitch=pitch, roll=roll)


class SimplePoseIntegrator(PoseIntegrator):
    """Integrates control intents into poses using naive Euler stepping."""

    def __init__(
        self,
        speed: float = 1.0,
        turn_speed: float = 90.0,
        pitch_speed: float = 90.0,
        roll_speed: float = 90.0,
    ) -> None:
        self._speed = speed
        self._turn_speed = turn_speed
        self._pitch_speed = pitch_speed
        self._roll_speed = roll_speed
        self._pose = Pose(position=(0.0, 0.0, 0.0), orientation=(1.0, 0.0, 0.0, 0.0))
        self._lock = threading.Lock()

    def step(self, intent: ControlIntent, delta_seconds: float) -> Pose:
        with self._lock:
            move = np.array(intent.move, dtype=np.float64) * self._speed * delta_seconds
            turn_radians = np.deg2rad(intent.turn * self._turn_speed * delta_seconds)
            pitch_radians = np.deg2rad(intent.pitch * self._pitch_speed * delta_seconds)
            roll_radians = np.deg2rad(intent.roll * self._roll_speed * delta_seconds)
            px, py, pz = self._pose.position
            qw, qx, qy, qz = self._pose.orientation
            # Apply yaw and roll rotations
            if turn_radians != 0.0:
                half = turn_radians * 0.5
                sin_half = np.sin(half)
                delta_yaw = (np.cos(half), 0.0, sin_half, 0.0)
                qw, qx, qy, qz = _quat_multiply((qw, qx, qy, qz), delta_yaw)
            if pitch_radians != 0.0:
                half = pitch_radians * 0.5
                sin_half = np.sin(half)
                delta_pitch = (np.cos(half), sin_half, 0.0, 0.0)
                qw, qx, qy, qz = _quat_multiply((qw, qx, qy, qz), delta_pitch)
            if roll_radians != 0.0:
                half = roll_radians * 0.5
                sin_half = np.sin(half)
                delta_roll = (np.cos(half), 0.0, 0.0, sin_half)
                qw, qx, qy, qz = _quat_multiply((qw, qx, qy, qz), delta_roll)
            # Rotate movement into world space
            rotation_matrix = _quat_to_matrix((qw, qx, qy, qz))
            move_world = rotation_matrix @ move
            px += move_world[0]
            py += move_world[1]
            pz += move_world[2]
            self._pose = Pose(position=(px, py, pz), orientation=(qw, qx, qy, qz))
            return self._pose

    def set_pose(self, pose: Pose) -> None:
        with self._lock:
            self._pose = pose

    def current_pose(self) -> Pose:
        with self._lock:
            return self._pose


def _quat_multiply(a: Sequence[float], b: Sequence[float]) -> Tuple[float, float, float, float]:
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return (
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    )


def _quat_to_matrix(quat: Sequence[float]) -> np.ndarray:
    w, x, y, z = quat
    norm = w * w + x * x + y * y + z * z
    if norm == 0.0:
        return np.identity(3, dtype=np.float64)
    s = 2.0 / norm
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    return np.array(
        [
            [1.0 - s * (yy + zz), s * (xy - wz), s * (xz + wy)],
            [s * (xy + wz), 1.0 - s * (xx + zz), s * (yz - wx)],
            [s * (xz - wy), s * (yz + wx), 1.0 - s * (xx + yy)],
        ],
        dtype=np.float64,
    )


__all__ = [
    "ControlIntent",
    "DebugFrame",
    "DebugViewer",
    "InputProvider",
    "KeyboardInput",
    "Pose",
    "PoseIntegrator",
    "RendererBackend",
    "SimplePoseIntegrator",
    "ViewerConfig",
]
