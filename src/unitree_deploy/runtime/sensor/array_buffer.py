from __future__ import annotations

from multiprocessing import shared_memory

import numpy as np


class SharedArrayObservationBuffer:
    """Float32 shared-memory array used by sensor producers and policy observations."""

    def __init__(
        self,
        *,
        name: str,
        shape: tuple[int, ...],
        create: bool,
        owner: bool = False,
    ) -> None:
        self.name = name
        self.shape = tuple(int(value) for value in shape)
        self.owner = owner
        self._closed = False
        self._size = int(np.prod(self.shape)) * np.dtype(np.float32).itemsize
        self._shm = shared_memory.SharedMemory(name=self.name, create=create, size=self._size)
        self._buffer = np.ndarray(self.shape, dtype=np.float32, buffer=self._shm.buf)
        if create:
            self._buffer.fill(0.0)

    @classmethod
    def create(cls, *, name: str, shape: tuple[int, ...]) -> "SharedArrayObservationBuffer":
        try:
            return cls(name=name, shape=shape, create=True, owner=True)
        except FileExistsError:
            stale = shared_memory.SharedMemory(name=name, create=False)
            stale.close()
            stale.unlink()
            return cls(name=name, shape=shape, create=True, owner=True)

    @classmethod
    def open(cls, *, name: str, shape: tuple[int, ...]) -> "SharedArrayObservationBuffer":
        return cls(name=name, shape=shape, create=False, owner=False)

    def update(self, value: np.ndarray) -> None:
        if value.shape != self.shape:
            raise ValueError(f"sensor array shape {value.shape} != expected {self.shape}")
        self._buffer[:] = value

    def get_latest(self) -> np.ndarray:
        return self._buffer.copy()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._shm.close()
        if self.owner:
            try:
                self._shm.unlink()
            except FileNotFoundError:
                pass
