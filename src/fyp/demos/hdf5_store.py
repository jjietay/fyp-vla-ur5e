"""HDF5 persistence for recorded episodes.

Episode schema (N timesteps) — this is the contract the LeRobot exporter and
any future consumer read against, so changing it is a breaking change:

    timestamps       (N,)          float   seconds since episode start
    joint_positions  (N, 6)        float   UR5e joint angles
    tcp_poses        (N, 6)        float   [x, y, z, rx, ry, rz]  (axis-angle)
    gripper_states   (N,)          int8    0 = closed, 1 = open
    images           (N, H, W, 3)  uint8   fixed-camera RGB
"""
from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np


def save_snapshots(snapshots: list, path: str | Path) -> None:
    """Stack per-field arrays from a list of TimestepSnapshot and write HDF5."""
    if len(snapshots) == 0:
        raise RuntimeError("Nothing to save — buffer is empty.")

    timestamps      = np.array([snap.timestamp for snap in snapshots])
    joint_positions = np.stack([snap.joint_positions for snap in snapshots])
    tcp_poses       = np.stack([snap.tcp_pose for snap in snapshots])
    gripper_states  = np.array([snap.gripper_state for snap in snapshots], dtype=np.int8)
    images          = np.stack([snap.image for snap in snapshots])

    with h5py.File(path, "w") as f:
        f.create_dataset("timestamps",      data=timestamps)
        f.create_dataset("joint_positions", data=joint_positions)
        f.create_dataset("tcp_poses",       data=tcp_poses)
        f.create_dataset("gripper_states",  data=gripper_states)
        f.create_dataset("images",          data=images)


def load_episode(path: str | Path) -> dict:
    """Read an episode back into plain numpy arrays."""
    with h5py.File(path, "r") as f:
        return {
            "timestamps":      f["timestamps"][:],
            "joint_positions": f["joint_positions"][:],
            "tcp_poses":       f["tcp_poses"][:],
            "gripper_states":  f["gripper_states"][:],
            "images":          f["images"][:],
        }


def episode_paths(episodes_dir: str | Path) -> list[Path]:
    """Every .h5/.hdf5 episode in a directory, sorted."""
    d = Path(episodes_dir)
    return sorted(d.glob("*.h5")) + sorted(d.glob("*.hdf5"))
