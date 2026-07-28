"""Convert recorded HDF5 episodes into a LeRobot dataset for SmolVLA.

Bridges two formats. DemoRecorder writes ABSOLUTE states with AXIS-ANGLE
orientation; SmolVLA / LeRobot want per-frame observations plus DELTA actions
with orientation in EULER angles. Format-agnostic by construction: the same
code converts sim-recorded and real-UR5e-recorded episodes, because both share
the recorder's schema (see hdf5_store.py).

Output LeRobot features:
    observation.state         (7,)   float32  [x,y,z, roll,pitch,yaw, gripper]  ABSOLUTE, Euler
    action                    (7,)   float32  [dx,dy,dz, droll,dpitch,dyaw, gripper]  DELTA + gripper target
    observation.images.<cam>  (H,W,3) video   fixed-camera RGB

The LeRobot import is deliberately inside `convert()` rather than at module
level: the pure conversion maths above it then stays importable (and testable)
from the FYP venv, which does not have lerobot installed.
"""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np

from fyp.helpers.rotations import rotvec_to_euler
from fyp.helpers.transforms import pose_inv, pose_trans


def build_state(tcp_pose: np.ndarray, gripper: int) -> np.ndarray:
    xyz = tcp_pose[:3]
    rpy = rotvec_to_euler(tcp_pose[3:])
    return np.concatenate([xyz, rpy, [float(gripper)]]).astype(np.float32)


def compute_delta_action(pose_t: np.ndarray,
                         pose_next: np.ndarray,
                         gripper_next: int) -> np.ndarray:
    dpos = pose_next[:3] - pose_t[:3]


    rel = pose_trans(pose_next, pose_inv(pose_t))
    drpy = rotvec_to_euler(rel[3:])

    return np.concatenate([dpos, drpy, [float(gripper_next)]]).astype(np.float32)


def infer_fps(timestamps: np.ndarray, fallback: float = 20.0) -> int:
    if len(timestamps) < 2:
        return int(fallback)
    dt = float(np.median(np.diff(timestamps)))
    return int(round(1.0 / dt)) if dt > 0 else int(fallback)


def convert(episodes_dir: str, repo_id: str, task: str,
            camera: str = "top", fps: int | None = None,
            root: str | None = None) -> None:


    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    from fyp.demos.hdf5_store import episode_paths

    paths = episode_paths(episodes_dir)
    if not paths:
        raise FileNotFoundError(f"No .h5/.hdf5 episodes found in {episodes_dir}")


    with h5py.File(paths[0], "r") as f:
        H, W = f["images"].shape[1:3]
        fps = fps or infer_fps(f["timestamps"][:])


    features = {
        "observation.state": {
            "dtype": "float32", "shape": (7,),
            "names": ["x", "y", "z", "roll", "pitch", "yaw", "gripper"],
        },
        "action": {
            "dtype": "float32", "shape": (7,),
            "names": ["dx", "dy", "dz", "droll", "dpitch", "dyaw", "gripper"],
        },
        f"observation.images.{camera}": {
            "dtype": "video", "shape": (H, W, 3),
            "names": ["height", "width", "channel"],
        },
    }

    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        fps=int(fps),
        features=features,
        robot_type="ur5e",
        use_videos=True,
        root=root,
    )

    for ep_path in paths:
        with h5py.File(ep_path, "r") as f:
            tcp = f["tcp_poses"][:]
            grip = f["gripper_states"][:]
            imgs = f["images"][:]
        N = len(tcp)

        for t in range(N):
            if t < N - 1:
                action = compute_delta_action(tcp[t], tcp[t + 1], int(grip[t + 1]))
            else:
                action = np.concatenate([np.zeros(6), [float(grip[t])]]).astype(np.float32)

            frame = {
                "observation.state": build_state(tcp[t], int(grip[t])),
                "action": action,
                f"observation.images.{camera}": imgs[t],
                "task": task,
            }
            dataset.add_frame(frame)

        dataset.save_episode()
        print(f"converted {ep_path.name}: {N} frames")

    dataset.finalize()
    print(f"done: {len(paths)} episodes -> {repo_id}  (fps={fps})")
