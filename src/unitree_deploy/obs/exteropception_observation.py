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
            from unitree_deploy.runtime.sensor.array_buffer import SharedArrayObservationBuffer

            self.sensor_buffer = SharedArrayObservationBuffer.open(
                name=self.shared_memory_name,
                shape=(self.height, self.width),
            )
        except FileNotFoundError:
            return

    def compute(self, context: ObservationContext) -> np.ndarray:
        del context
        self._ensure_buffer()
        if self.sensor_buffer is None:
            return np.zeros(self.base_dim, dtype=self.dtype)

        value = self.sensor_buffer.get_latest()
        return value.reshape(-1).astype(self.dtype)


class DepthObservation(_SharedArrayObservation):
    """Depth image observation updated asynchronously by a camera producer."""

    def __init__(
        self,
        *,
        history_len: int,
        height: int,
        width: int,
        depth_buffer=None,
        shared_memory_name: str | None = None,
    ) -> None:
        super().__init__(
            history_len=history_len,
            height=height,
            width=width,
            sensor_buffer=depth_buffer,
            shared_memory_name=shared_memory_name,
        )


class HeightScanObservation(_SharedArrayObservation):
    """Height grid observation updated asynchronously by a terrain sensor producer."""


# Export for registration
OBSERVATION_TYPES = {
    "depth": DepthObservation,
    "height_scan": HeightScanObservation,
}
