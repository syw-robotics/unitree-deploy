from __future__ import annotations

import numpy as np
from .observation import ObservationBase, ObservationContext


class DepthObservation(ObservationBase):
    """Depth image observation with async update support.

    Depth images are updated at a lower frequency (e.g., 10Hz) than policy
    inference (e.g., 50Hz). This observation reads from a shared buffer that
    is updated asynchronously by the camera thread.
    """

    def __init__(
        self,
        *,
        history_len: int,
        height: int,
        width: int,
        depth_buffer=None,
    ) -> None:
        self.height = int(height)
        self.width = int(width)
        self.depth_buffer = depth_buffer

        base_dim = self.height * self.width
        super().__init__(base_dim=base_dim, history_len=history_len, dtype=np.float32)

    def _compute_current_obs(self, context: ObservationContext) -> np.ndarray:
        """Get latest depth image from shared buffer."""
        if self.depth_buffer is None:
            return np.zeros(self.base_dim, dtype=self.dtype)

        # Get latest depth frame (non-blocking)
        depth_image = self.depth_buffer.get_latest()

        # Flatten to 1D vector
        return depth_image.reshape(-1).astype(self.dtype)

    def compute(self) -> np.ndarray:
        """Return flattened history buffer."""
        return self.buffer.reshape(-1)


# Export for registration
OBSERVATION_TYPES = {
    "depth": DepthObservation,
}
