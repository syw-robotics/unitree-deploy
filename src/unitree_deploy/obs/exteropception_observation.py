from __future__ import annotations

import numpy as np
from .observation import ObservationBase, ObservationContext


class _SharedArrayObservation(ObservationBase):
    def __init__(
        self,
        *,
        history_len: int,
        height: int,
        width: int,
        sensor_buffer=None,
        shared_memory_name: str | None = None,
    ) -> None:
        self.height = int(height)
        self.width = int(width)
        self.sensor_buffer = sensor_buffer
        self.shared_memory_name = shared_memory_name

        base_dim = self.height * self.width
        super().__init__(base_dim=base_dim, history_len=history_len, dtype=np.float32)

    def _ensure_buffer(self) -> None:
        if self.sensor_buffer is not None or not self.shared_memory_name:
            return
        try:
            self.sensor_buffer = self._open_shared_buffer()
        except FileNotFoundError:
            return

    def _open_shared_buffer(self):
        from unitree_deploy.runtime.sensor.array_buffer import SharedArrayObservationBuffer

        return SharedArrayObservationBuffer.open(
            name=self.shared_memory_name,
            shape=(self.height, self.width),
        )

    def compute(self, context: ObservationContext) -> np.ndarray:
        del context
        self._ensure_buffer()
        if self.sensor_buffer is None:
            return np.zeros(self.base_dim, dtype=self.dtype)

        value = self.sensor_buffer.get_latest()
        return value.reshape(-1).astype(self.dtype)


class DepthObservation(_SharedArrayObservation):
    """Depth image observation updated asynchronously by a camera producer.

    The controller runs faster than the camera. Frame sequences, rather than
    pixel equality, determine when the depth history advances. This preserves
    sensor timing even when consecutive images happen to be identical.
    """

    def __init__(
        self,
        *,
        history_len: int,
        height: int,
        width: int,
        depth_buffer=None,
        shared_memory_name: str | None = None,
        history_skip_frames: int = 1,
    ) -> None:
        self.history_skip_frames = int(history_skip_frames)
        if self.history_skip_frames < 1:
            raise ValueError("history_skip_frames must be at least 1")
        self._last_sensor_sequence: int | None = None
        super().__init__(
            history_len=history_len,
            height=height,
            width=width,
            sensor_buffer=depth_buffer,
            shared_memory_name=shared_memory_name,
        )

    def _open_shared_buffer(self):
        from unitree_deploy.runtime.sensor.depth_camera.depth_buffer import (
            SharedDepthObservationBuffer,
        )

        return SharedDepthObservationBuffer.open(
            name=self.shared_memory_name,
            height=self.height,
            width=self.width,
        )

    def _current_with_sequence(
        self,
        context: ObservationContext,
    ) -> tuple[np.ndarray, int | None]:
        del context
        self._ensure_buffer()
        if self.sensor_buffer is None:
            return np.zeros(self.base_dim, dtype=self.dtype), None

        if hasattr(self.sensor_buffer, "get_latest_with_sequence"):
            value, sequence = self.sensor_buffer.get_latest_with_sequence()
        else:
            value = self.sensor_buffer.get_latest()
            sequence = None
        return self._process_values(value), sequence

    def reset(self) -> None:
        super().reset()
        self._last_sensor_sequence = None

    def prime(self, context: ObservationContext) -> None:
        current, sequence = self._current_with_sequence(context)
        self.buffer[:] = current
        self._last_sensor_sequence = sequence

    def update(self, context: ObservationContext) -> None:
        current, sequence = self._current_with_sequence(context)
        if sequence is not None:
            if sequence == 0 or sequence == self._last_sensor_sequence:
                return
            if (
                self._last_sensor_sequence is not None
                and sequence > self._last_sensor_sequence
                and sequence - self._last_sensor_sequence < self.history_skip_frames
            ):
                return
            self._last_sensor_sequence = sequence
        elif self.history_len > 1 and np.array_equal(current, self.buffer[-1]):
            return
        if self.history_len > 1:
            self.buffer[:-1] = self.buffer[1:]
        self.buffer[-1] = current


class HeightScanObservation(_SharedArrayObservation):
    """Height grid observation updated asynchronously by a terrain sensor producer."""


# Export for registration
OBSERVATION_TYPES = {
    "depth": DepthObservation,
    "height_scan": HeightScanObservation,
}
