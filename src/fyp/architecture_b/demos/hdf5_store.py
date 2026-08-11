"""Using HDF5 for storing of recorded episodes.

Episode format (N timesteps) which LeRobot exporter is built upon:

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
    """
    This takes the recorder's list of snapshots and writes them to one HDF5 file.

    It pivots the data on the way, where a list of N snapshots (each with the same 5 fields)
    becomes 5 arrays of length N, one dataset each.
    """
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
    """
    This takse a path to an episode file and gives back all of its arrays in a dict.
    In other words, it returns every frame.
    """
    with h5py.File(path, "r") as f:
        return {
            "timestamps":      f["timestamps"][:],
            "joint_positions": f["joint_positions"][:],
            "tcp_poses":       f["tcp_poses"][:],
            "gripper_states":  f["gripper_states"][:],
            "images":          f["images"][:],
        }


def episode_paths(episodes_dir: str | Path) -> list[Path]:
    """
    It takes a folder and gives you every episode file in it, in one consistent
    order across both extensions.

    Sorted by filename stem, so ep_001.h5 comes before ep_002.hdf5 rather than
    after it. The previous version returned sorted(*.h5) + sorted(*.hdf5), which
    put every .h5 before every .hdf5 regardless of name. With mixed extensions
    that silently reorders a dataset: frames stay intact, episode boundaries move,
    and nothing raises. You would only notice it as a model that will not converge.
    """
    d = Path(episodes_dir)
    return sorted([*d.glob("*.h5"), *d.glob("*.hdf5")], key=lambda p: (p.stem, p.suffix))
