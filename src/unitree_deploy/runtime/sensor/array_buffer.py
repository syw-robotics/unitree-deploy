from __future__ import annotations

from multiprocessing import resource_tracker, shared_memory
import time

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
        sequenced: bool = False,
        untrack_non_owner: bool = True,
    ) -> None:
        self.name = name
        self.shape = tuple(int(value) for value in shape)
        self.owner = owner
        self.sequenced = bool(sequenced)
        self._closed = False
        self._header_size = np.dtype(np.uint64).itemsize if self.sequenced else 0
        self._data_size = int(np.prod(self.shape)) * np.dtype(np.float32).itemsize
        self._size = self._header_size + self._data_size
        self._shm = shared_memory.SharedMemory(name=self.name, create=create, size=self._size)
        if not create and not owner and untrack_non_owner:
            # Python <3.13 otherwise lets a standalone consumer's tracker unlink
            # shared memory that is still owned and written by another process.
            resource_tracker.unregister(self._shm._name, "shared_memory")
        self._sequence = (
            np.ndarray((1,), dtype=np.uint64, buffer=self._shm.buf)
            if self.sequenced
            else None
        )
        self._buffer = np.ndarray(
            self.shape,
            dtype=np.float32,
            buffer=self._shm.buf,
            offset=self._header_size,
        )
        if create:
            if self._sequence is not None:
                self._sequence[0] = 0
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
        if self._sequence is None:
            self._buffer[:] = value
            return

        # Odd values mark a write in progress; even values identify complete frames.
        write_sequence = int(self._sequence[0]) + 1
        if write_sequence % 2 == 0:
            write_sequence += 1
        self._sequence[0] = write_sequence
        self._buffer[:] = value
        self._sequence[0] = write_sequence + 1

    def get_latest(self) -> np.ndarray:
        if self._sequence is not None:
            return self.get_latest_with_sequence()[0]
        return self._buffer.copy()

    def get_latest_with_sequence(self) -> tuple[np.ndarray, int | None]:
        if self._sequence is None:
            return self._buffer.copy(), None

        deadline = time.monotonic() + 0.01
        while time.monotonic() < deadline:
            sequence_before = int(self._sequence[0])
            if sequence_before % 2 != 0:
                time.sleep(0)
                continue
            value = self._buffer.copy()
            sequence_after = int(self._sequence[0])
            if sequence_before == sequence_after:
                return value, sequence_after // 2
            time.sleep(0)
        raise RuntimeError(f"timed out reading a complete sensor frame from shared memory {self.name!r}")

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
