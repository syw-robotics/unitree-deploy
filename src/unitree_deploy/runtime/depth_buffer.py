from __future__ import annotations

import threading
import time
import numpy as np


class DepthObservationBuffer:
    """Thread-safe buffer for async depth image updates.

    Camera thread writes at ~10Hz, policy thread reads at ~50Hz.
    """

    def __init__(self, height: int, width: int) -> None:
        self.height = height
        self.width = width
        self._buffer = np.zeros((height, width), dtype=np.float32)
        self._lock = threading.Lock()
        self._timestamp = 0.0

    def update(self, depth_image: np.ndarray) -> None:
        """Update buffer with new depth image (called by camera thread)."""
        if depth_image.shape != (self.height, self.width):
            raise ValueError(
                f"depth_image shape {depth_image.shape} != expected ({self.height}, {self.width})"
            )

        with self._lock:
            self._buffer[:] = depth_image
            self._timestamp = time.time()

    def get_latest(self) -> np.ndarray:
        """Get latest depth image (called by policy thread)."""
        with self._lock:
            return self._buffer.copy()

    def get_timestamp(self) -> float:
        """Get timestamp of last update."""
        with self._lock:
            return self._timestamp
