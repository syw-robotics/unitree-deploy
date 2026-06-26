from __future__ import annotations

from multiprocessing import shared_memory
import threading
import time
import numpy as np

from unitree_deploy.runtime.sensor.array_buffer import SharedArrayObservationBuffer


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


class SharedDepthObservationBuffer(SharedArrayObservationBuffer):
    """Process-shared depth image buffer backed by POSIX shared memory."""

    def __init__(
        self,
        *,
        name: str,
        height: int,
        width: int,
        create: bool,
        owner: bool = False,
    ) -> None:
        self.height = int(height)
        self.width = int(width)
        super().__init__(
            name=name,
            shape=(self.height, self.width),
            create=create,
            owner=owner,
        )

    @classmethod
    def create(cls, *, name: str, height: int, width: int) -> "SharedDepthObservationBuffer":
        try:
            return cls(name=name, height=height, width=width, create=True, owner=True)
        except FileExistsError:
            stale = shared_memory.SharedMemory(name=name, create=False)
            stale.close()
            stale.unlink()
            return cls(name=name, height=height, width=width, create=True, owner=True)

    @classmethod
    def open(cls, *, name: str, height: int, width: int) -> "SharedDepthObservationBuffer":
        return cls(name=name, height=height, width=width, create=False, owner=False)

    def update(self, depth_image: np.ndarray) -> None:
        if depth_image.shape != (self.height, self.width):
            raise ValueError(
                f"depth_image shape {depth_image.shape} != expected ({self.height}, {self.width})"
            )
        super().update(depth_image)

    def get_latest(self) -> np.ndarray:
        return self._buffer.copy()

    def get_timestamp(self) -> float:
        return 0.0
