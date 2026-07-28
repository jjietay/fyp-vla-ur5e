"""In-memory capture of a demonstration episode.

The recorder moves nothing and reads nothing. Whatever is driving the robot
hands it state via `record(...)`, which is what makes it identical against sim
and the real UR5e.

Writing to disk lives in `hdf5_store.py`; this module only buffers. The split
matters because the buffer is pure and testable with no h5py and no filesystem,
while the writer is the part that has to change if the storage format ever does.
"""

from dataclasses import dataclass
from pathlib import Path
import time

import numpy as np


@dataclass
class TimestepSnapshot:

    timestamp:          float
    joint_positions:    np.ndarray
    tcp_pose:           np.ndarray
    gripper_state:      int
    image:              np.ndarray


class DemoRecorder:
    def __init__(self):
        self._buffer: list[TimestepSnapshot] = []
        self._start_time: float | None = None

    def __len__(self) -> int:
        return len(self._buffer)

    @property
    def frames(self) -> list[TimestepSnapshot]:
        return self._buffer

    def start_episode(self) -> None:
        self._buffer = []
        self._start_time = time.monotonic()

    def record(
                self,
                joint_positions : np.ndarray,
                tcp_pose: np.ndarray,
                gripper_state: int,
                image: np.ndarray,
                timestamp: float | None = None,
                ) -> None:
        if self._start_time is None:
            raise RuntimeError("Call start_episode() before record().")

        ts = timestamp if timestamp is not None else (time.monotonic() - self._start_time)
        snapshot = TimestepSnapshot(
            timestamp = ts,
            joint_positions = joint_positions,
            tcp_pose = tcp_pose,
            gripper_state = gripper_state,
            image = image
            )
        self._buffer.append(snapshot)

    def save_episode(self, path: str | Path) -> None:
        from fyp.demos.hdf5_store import save_snapshots

        save_snapshots(self._buffer, path)
