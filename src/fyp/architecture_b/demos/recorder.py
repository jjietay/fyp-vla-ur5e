""" recorder.py

In-memory capture of a demonstration episode. Stored as a list in _buffer.

The recorder doesn't read anything or move the arm. Whatever moving the robot will
hand this file its state via record().

Can be used interchangeably btw sim and real.

Writing to disk lives in `hdf5_store.py`; this module only buffers. This matters because
the buffer is pure (no h5py or filesystem dependencies) and testable.
"""

from dataclasses import dataclass
from pathlib import Path
import time

import numpy as np


@dataclass
class TimestepSnapshot:
    """
    This is just a data class that temporarily stores the below information
    at a given time.
    """

    timestamp:          float
    joint_positions:    np.ndarray
    tcp_pose:           np.ndarray
    gripper_state:      int
    image:              np.ndarray


class DemoRecorder:
    def __init__(self):
        """
        Initialize as an empty list first"""
        self._buffer: list[TimestepSnapshot] = []
        self._start_time: float | None = None

    def __len__(self) -> int:
        """
        allows us to check the length of the buffer list"""
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
        """
        This simply takes in key recording values and append them into the buffer list
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
        from fyp.architecture_b.demos.hdf5_store import save_snapshots

        save_snapshots(self._buffer, path)
