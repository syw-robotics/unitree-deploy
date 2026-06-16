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
    """Base class for one observation term with optional history stacking.

    History is stored in chronological order (oldest at index 0, newest at -1),
    matching the convention used in Isaac Lab training and the C++ deploy code:
        buffer = [t-K+1, ..., t-1, t]
    """

    def __init__(self, *, base_dim: int, history_len: int, dtype=np.float32) -> None:
        self.base_dim = int(base_dim)
        self.history_len = int(history_len)
        self.dtype = np.dtype(dtype)
        self.buffer = np.zeros((self.history_len, self.base_dim), dtype=self.dtype)
        self._clip_min = None
        self._clip_max = None

    @property
    def size(self) -> int:
        return self.base_dim * self.history_len

    def set_clip(self, clipp) -> None:
        clip_range = np.asarray(clipp, dtype=self.dtype)
        if clip_range.shape == (2,):
            clip_min = clip_range[0]
            clip_max = clip_range[1]
        elif clip_range.shape == (self.base_dim, 2):
            clip_min = clip_range[:, 0]
            clip_max = clip_range[:, 1]
        else:
            raise ValueError(
                f"clipp must have shape (2,) or ({self.base_dim}, 2), got {clip_range.shape}"
            )
        if np.any(clip_min > clip_max):
            raise ValueError("clipp lower bounds must be <= upper bounds")
        self._clip_min = clip_min
        self._clip_max = clip_max

    def reset(self) -> None:
        self.buffer.fill(0.0)

    def prime(self, context: ObservationContext) -> None:
        current = self._compute_clipped_current_obs(context)
        self.buffer[:] = current

    def update(self, context: ObservationContext) -> None:
        current = self._compute_clipped_current_obs(context)
        if self.history_len > 1:
            # Shift left: drop oldest at index 0, move everyone down
            self.buffer[:-1] = self.buffer[1:]
        self.buffer[-1] = current

    def compute(self, context: ObservationContext) -> np.ndarray:
        raise NotImplementedError

    def _compute_clipped_current_obs(self, context: ObservationContext) -> np.ndarray:
        return self._clip_values(self.compute(context))

    def _clip_values(self, values) -> np.ndarray:
        current = np.asarray(values, dtype=self.dtype).reshape(-1)
        if current.size != self.base_dim:
            raise ValueError(f"observation term produced {current.size} values, expected {self.base_dim}")
        if self._clip_min is None:
            return current
        return np.clip(current, self._clip_min, self._clip_max)


# ---------- Observation Terms ----------

class CommandObservation(ObservationBase):
    """Joystick command clipped into the policy command range (velocity_command semantics)."""

    def __init__(
        self,
        *,
        history_len: int,
        command_range,
        dtype=np.float32,
    ) -> None:
        command_range_array = np.asarray(command_range, dtype=dtype)
        num_commands = command_range_array.shape[0]
        super().__init__(base_dim=num_commands, history_len=history_len, dtype=dtype)
        self.command_min = command_range_array[:, 0]
        self.command_max = command_range_array[:, 1]
        self._current = np.empty(num_commands, dtype=self.dtype)

    def compute(self, context: ObservationContext) -> np.ndarray:
        joystick = np.asarray(context.command, dtype=self.dtype).reshape(-1)
        np.clip(joystick[: self.base_dim], self.command_min, self.command_max, out=self._current)
        return self._current


class ProjectedGravityObservation(ObservationBase):
    def __init__(self, *, history_len: int, dtype=np.float32) -> None:
        super().__init__(base_dim=3, history_len=history_len, dtype=dtype)

    def compute(self, context: ObservationContext) -> np.ndarray:
        return _quat_to_body_gravity(context.quat).astype(self.dtype, copy=False)


class BaseAngularVelocityObservation(ObservationBase):
    """Angular velocity observation.

    When *scale* is provided, the gyro reading is multiplied by the scale factor.
    """

    def __init__(
        self,
        *,
        history_len: int,
        scale: float | list[float] | np.ndarray | None = None,
        dtype=np.float32,
    ) -> None:
        super().__init__(base_dim=3, history_len=history_len, dtype=dtype)
        if scale is not None:
            self._scale = np.asarray(scale, dtype=self.dtype).reshape(-1)
        else:
            self._scale = None

    def compute(self, context: ObservationContext) -> np.ndarray:
        values = np.asarray(context.gyro, dtype=self.dtype).reshape(-1)
        if self._scale is not None:
            values = values * self._scale[: self.base_dim]
        return values


class JointPositionObservation(ObservationBase):
    """Joint position observation.

    When *default_q* is provided, returns relative positions (joint_pos_rel):
        obs[i] = q[controlled_joint_indices[i]] - default_q[i]
    """

    def __init__(
        self,
        *,
        controlled_joint_indices: np.ndarray,
        history_len: int,
        default_q: np.ndarray | None = None,
        dtype=np.float32,
    ) -> None:
        controlled_joint_indices = np.asarray(controlled_joint_indices, dtype=np.int64).reshape(-1)
        super().__init__(base_dim=controlled_joint_indices.size, history_len=history_len, dtype=dtype)
        self.controlled_joint_indices = controlled_joint_indices
        self.default_q = np.asarray(default_q, dtype=dtype) if default_q is not None else None

    def compute(self, context: ObservationContext) -> np.ndarray:
        q = np.asarray(context.q, dtype=self.dtype).reshape(-1)
        values = q[self.controlled_joint_indices]
        if self.default_q is not None:
            values = values - self.default_q
        return values


class JointVelocityObservation(ObservationBase):
    """Joint velocity observation.

    When *scale* is provided, the velocity is multiplied by the scale factor
    (applied after indexing).
    """

    def __init__(
        self,
        *,
        controlled_joint_indices: np.ndarray,
        history_len: int,
        use_position_difference: bool = False,
        scale: float | np.ndarray | None = None,
        dtype=np.float32,
    ) -> None:
        controlled_joint_indices = np.asarray(controlled_joint_indices, dtype=np.int64).reshape(-1)
        super().__init__(base_dim=controlled_joint_indices.size, history_len=history_len, dtype=dtype)
        self.controlled_joint_indices = controlled_joint_indices
        self.use_position_difference = bool(use_position_difference)
        self._vel_scale = float(scale) if scale is not None else None
        self._previous_q = None

    def compute(self, context: ObservationContext) -> np.ndarray:
        dq = np.asarray(context.dq, dtype=self.dtype).reshape(-1)
        values = dq[self.controlled_joint_indices]
        if self._vel_scale is not None:
            values = values * self._vel_scale
        return values

    def reset(self) -> None:
        super().reset()
        self._previous_q = None


class PreviousActionObservation(ObservationBase):
    """Action history is updated after ONNX inference, not from the sensor snapshot."""

    def __init__(self, *, action_dim: int, history_len: int, dtype=np.float32) -> None:
        super().__init__(base_dim=action_dim, history_len=history_len, dtype=dtype)

    def compute(self, context: ObservationContext) -> np.ndarray:
        return np.zeros(self.base_dim, dtype=self.dtype)

    def update(self, context: ObservationContext) -> None:
        del context

    def record_action(self, action) -> None:
        action_array = self._clip_values(action)
        if self.history_len > 1:
            self.buffer[:-1] = self.buffer[1:]
        self.buffer[-1] = action_array


# ---------- Observation Group Wrapper ----------

class ObservationGroup:
    """Fixed-layout observation vector assembled from named observation terms."""

    def __init__(
        self,
        observations: Sequence[ObservationBase],
        *,
        obs_group_concat_mode: str = "term_major",
        dtype=np.float32,
    ) -> None:
        self.observations = list(observations)
        if obs_group_concat_mode not in ("term_major", "history_major"):
            raise ValueError("obs_group_concat_mode must be 'term_major' or 'history_major'")
        self.obs_group_concat_mode = obs_group_concat_mode
        self.dtype = np.dtype(dtype)
        self._slices: list[tuple[ObservationBase, slice]] = []
        offset = 0
        for observation in self.observations:
            next_offset = offset + observation.size
            self._slices.append((observation, slice(offset, next_offset)))
            offset = next_offset
        self.output = np.zeros(offset, dtype=self.dtype)
        self.history_len = self.observations[0].history_len if self.observations else 0
        if self.obs_group_concat_mode == "history_major":
            for observation in self.observations:
                if observation.history_len != self.history_len:
                    raise ValueError("frame obs_group_concat_mode requires all observations to use the same history_len")

    @property
    def size(self) -> int:
        return int(self.output.size)

    def reset(self) -> None:
        for observation in self.observations:
            observation.reset()

    def prime(self, context: ObservationContext) -> None:
        for observation in self.observations:
            observation.prime(context)

    def update(self, context: ObservationContext) -> None:
        for observation in self.observations:
            observation.update(context)

    def compute(self) -> np.ndarray:
        # history_major mode: concat by obs historical order
        if self.obs_group_concat_mode == "history_major":
            offset = 0
            for history_i in range(self.history_len):
                for observation in self.observations:
                    next_offset = offset + observation.base_dim
                    self.output[offset:next_offset] = observation.buffer[history_i]
                    offset = next_offset
            return self.output

        # term_major mode: concat by term order
        for observation, output_slice in self._slices:
            self.output[output_slice] = observation.buffer.reshape(-1)
        return self.output
