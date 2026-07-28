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
    """One moment in time. Five fields, recorded at every timestep."""

    timestamp:          float
    joint_positions:    np.ndarray  # shape of (6,) --> UR5e's 6 joints
    tcp_pose:           np.ndarray  # tcp position (6,) --> (x,y,z,rx,ry,rz)
    gripper_state:      int         # 0 or 1 (0 represents closed)
    image:              np.ndarray  # (H, W, C) where channel, C == 3, dtype == uint8


class DemoRecorder:
    def __init__(self):
        self._buffer: list[TimestepSnapshot] = []
        self._start_time: float | None = None

    def __len__(self) -> int:
        """Frames buffered so far.

        Exposed so callers (e.g. the teleop server) can check progress without
        reaching into the private `_buffer`.
        """
        return len(self._buffer)

    @property
    def frames(self) -> list[TimestepSnapshot]:
        """Read-only view of the buffered snapshots, for the storage layer."""
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
        """Append a snapshot.

        timestamp: seconds since episode start. If None (default), wall-clock
        elapsed time is used — correct for real-time recording (teleop server).
        For offline / scripted generation that does not run in real time, pass
        an explicit SIMULATED time (e.g. step_count * control_dt) so the stored
        cadence reflects sim-time, not how fast the CPU happened to run.
        """
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
        """Write the buffered episode to HDF5.

        Kept as a method so every existing call site is unchanged; the actual
        writing is delegated to the storage layer.
        """
        from fyp.demos.hdf5_store import save_snapshots

        save_snapshots(self._buffer, path)
