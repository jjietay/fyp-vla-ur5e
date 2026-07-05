from dataclasses import dataclass
import numpy as np
import time
import h5py
from pathlib import Path

@dataclass
class TimestepSnapshot:
    timestamp:          float
    joint_positions:    np.ndarray  # shape of (6,) --> UR5e's 6 joints
    tcp_pose:           np.ndarray  # tcp position (6,) --> (x,y,z,rx,ry,rz)
    gripper_state:      int         # 0 or 1 (o represents closed)
    image:              np.ndarray  # (H, W, C) where channel, C == 3, dtype == uint8


class DemoRecorder:
    def __init__(self):
        self._buffer: list[TimestepSnapshot] = []
        self._start_time: float | None = None

    def start_episode(self) -> None:
        self._buffer = []
        self._start_time = time.monotonic()

    def record(
                self,
                joint_positions : np.ndarray,
                tcp_pose: np.ndarray, 
                gripper_state: int,
                image: np.ndarray,
                ) -> None:
        
        if self._start_time is None:
            raise RuntimeError("Call start_episode() before record().")
        
        snapshot = TimestepSnapshot(
            timestamp = time.monotonic() - self._start_time,
            joint_positions = joint_positions,
            tcp_pose = tcp_pose,
            gripper_state = gripper_state,
            image = image
            )
        self._buffer.append(snapshot)

    def save_episode(self, path: str | Path) -> None:
        if len(self._buffer) == 0:
            raise RuntimeError("Nothing to save — buffer is empty.")

        # Stack per-field arrays from self._buffer
        timestamps      = np.array([snap.timestamp for snap in self._buffer])
        joint_positions = np.stack([snap.joint_positions for snap in self._buffer])
        tcp_poses       = np.stack([snap.tcp_pose for snap in self._buffer])
        gripper_states  = np.array([snap.gripper_state for snap in self._buffer], dtype=np.int8)
        images          = np.stack([snap.image for snap in self._buffer])

        # Write to HDF5
        with h5py.File(path, "w") as f:
            f.create_dataset("timestamps",      data=timestamps)
            f.create_dataset("joint_positions", data=joint_positions)
            f.create_dataset("tcp_poses",       data=tcp_poses)
            f.create_dataset("gripper_states",  data=gripper_states)
            f.create_dataset("images",          data=images)


class RateControl:
    def __init__(self, hz: float = 20.0):
        self.dt = 1.0 / hz # period in seconds (0.05 for 20Hz)
        self._next_tick: float | None = None

    def start(self) -> None:
        """Call once before the loop begins."""
        self._next_tick = time.monotonic()

    def wait(self) -> None:
        """Call at the end of each loop iteration to hold the rate."""
        if self._next_tick is None:
            raise RuntimeError("Call start() before wait().")
        self._next_tick += self.dt
        sleep_time = self._next_tick - time.monotonic()
        if sleep_time > 0:
            time.sleep(sleep_time)


