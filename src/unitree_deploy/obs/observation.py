from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ObservationContext:
    """One control-step snapshot passed from controller.py into policy.py."""

    q: np.ndarray
    dq: np.ndarray
    quat: np.ndarray
    gyro: np.ndarray
    command: np.ndarray

def _normalize_quaternion(quat) -> np.ndarray:
    quat_array = np.asarray(quat, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(quat_array))
    return quat_array / norm

def _quat_to_body_gravity(quat) -> np.ndarray:
    # Equivalent to R(q).T @ [0, 0, -1], written directly to avoid building a matrix per step.
    w, x, y, z = _normalize_quaternion(quat)
    return np.array(
        [
            2.0 * (y * w - x * z),
            -2.0 * (y * z + x * w),
            2.0 * (x * x + y * y) - 1.0,
        ],
        dtype=np.float32,
    )


class ObservationBase:
    """Base class for one observation term with optional history stacking."""

    def __init__(self, *, base_dim: int, history_len: int, dtype=np.float32) -> None:
        self.base_dim = int(base_dim)
        self.history_len = int(history_len)
        self.dtype = np.dtype(dtype)
        self.buffer = np.zeros((self.history_len, self.base_dim), dtype=self.dtype)

    @property
    def size(self) -> int:
        return self.base_dim * self.history_len

    def reset(self) -> None:
        self.buffer.fill(0.0)

    def update(self, context: ObservationContext) -> None:
        current = np.asarray(self._compute_current(context), dtype=self.dtype).reshape(-1)
        if self.history_len > 1:
            self.buffer[1:] = self.buffer[:-1]
        self.buffer[0] = current

    def compute(self) -> np.ndarray:
        return self.buffer.reshape(-1)

    def _compute_current(self, context: ObservationContext) -> np.ndarray:
        raise NotImplementedError


class CommandObservation(ObservationBase):
    """Joystick command scaled from [-1, 1] into the policy command range."""

    def __init__(
        self,
        *,
        history_len: int,
        command_range,
        dtype=np.float32,
    ) -> None:
        super().__init__(base_dim=4, history_len=history_len, dtype=dtype)
        command_range_array = np.asarray(command_range, dtype=self.dtype)
        self.command_min = command_range_array[:, 0]
        self.command_max = command_range_array[:, 1]
        self._range = self.command_max - self.command_min
        self._current = np.empty(4, dtype=self.dtype)

    def _compute_current(self, context: ObservationContext) -> np.ndarray:
        joystick = np.asarray(context.command, dtype=self.dtype).reshape(-1)
        joystick = np.clip(joystick, -1.0, 1.0)
        self._current[:3] = 0.5 * (joystick + 1.0) * self._range + self.command_min
        return self._current


class ProjectedGravityObservation(ObservationBase):
    def __init__(self, *, history_len: int, dtype=np.float32) -> None:
        super().__init__(base_dim=3, history_len=history_len, dtype=dtype)

    def _compute_current(self, context: ObservationContext) -> np.ndarray:
        return _quat_to_body_gravity(context.quat).astype(self.dtype, copy=False)


class BaseAngularVelocityObservation(ObservationBase):
    def __init__(self, *, history_len: int, dtype=np.float32) -> None:
        super().__init__(base_dim=3, history_len=history_len, dtype=dtype)

    def _compute_current(self, context: ObservationContext) -> np.ndarray:
        return np.asarray(context.gyro, dtype=self.dtype).reshape(-1)


class JointPositionObservation(ObservationBase):
    def __init__(
        self,
        *,
        controlled_joint_indices: np.ndarray,
        history_len: int,
        dtype=np.float32,
    ) -> None:
        controlled_joint_indices = np.asarray(controlled_joint_indices, dtype=np.int64).reshape(-1)
        super().__init__(base_dim=controlled_joint_indices.size, history_len=history_len, dtype=dtype)
        self.controlled_joint_indices = controlled_joint_indices

    def _compute_current(self, context: ObservationContext) -> np.ndarray:
        q = np.asarray(context.q, dtype=self.dtype).reshape(-1)
        return q[self.controlled_joint_indices]


class JointVelocityObservation(ObservationBase):
    def __init__(
        self,
        *,
        controlled_joint_indices: np.ndarray,
        history_len: int,
        use_position_difference: bool = False,
        dtype=np.float32,
    ) -> None:
        controlled_joint_indices = np.asarray(controlled_joint_indices, dtype=np.int64).reshape(-1)
        super().__init__(base_dim=controlled_joint_indices.size, history_len=history_len, dtype=dtype)
        self.controlled_joint_indices = controlled_joint_indices
        self.use_position_difference = bool(use_position_difference)
        self._previous_q = None

    def _compute_current(self, context: ObservationContext) -> np.ndarray:
        dq = np.asarray(context.dq, dtype=self.dtype).reshape(-1)
        return dq[self.controlled_joint_indices]

    def reset(self) -> None:
        super().reset()
        self._previous_q = None


class PreviousActionObservation(ObservationBase):
    """Action history is updated after ONNX inference, not from the sensor snapshot."""

    def __init__(self, *, action_dim: int, history_len: int, dtype=np.float32) -> None:
        super().__init__(base_dim=action_dim, history_len=history_len, dtype=dtype)

    def _compute_current(self, context: ObservationContext) -> np.ndarray:
        return np.zeros(self.base_dim, dtype=self.dtype)

    def update(self, context: ObservationContext) -> None:
        del context

    def record_action(self, action) -> None:
        action_array = np.asarray(action, dtype=self.dtype).reshape(-1)
        if self.history_len > 1:
            self.buffer[1:] = self.buffer[:-1]
        self.buffer[0] = action_array


class ObservationGroup:
    """Fixed-layout observation vector assembled from named observation terms."""

    def __init__(self, observations: Sequence[ObservationBase], *, dtype=np.float32) -> None:
        self.observations = list(observations)
        self.dtype = np.dtype(dtype)
        self._slices: list[tuple[ObservationBase, slice]] = []
        offset = 0
        for observation in self.observations:
            next_offset = offset + observation.size
            self._slices.append((observation, slice(offset, next_offset)))
            offset = next_offset
        self.output = np.zeros(offset, dtype=self.dtype)

    @property
    def size(self) -> int:
        return int(self.output.size)

    def reset(self) -> None:
        for observation in self.observations:
            observation.reset()

    def update(self, context: ObservationContext) -> None:
        for observation in self.observations:
            observation.update(context)

    def compute(self) -> np.ndarray:
        for observation, output_slice in self._slices:
            self.output[output_slice] = observation.compute()
        return self.output
