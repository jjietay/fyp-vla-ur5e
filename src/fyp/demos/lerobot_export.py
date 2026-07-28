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
    """Build one ABSOLUTE observation.state vector [x,y,z, roll,pitch,yaw, gripper].

    Position passes through; orientation is converted axis-angle -> Euler; the
    gripper is appended as a scalar. This is the proprioceptive state SmolVLA
    conditions on. (Alternative design: use the 6 joint angles instead of the
    TCP pose -- swap this function if you prefer joint-space state.)
    """
    xyz = tcp_pose[:3]
    rpy = rotvec_to_euler(tcp_pose[3:])
    return np.concatenate([xyz, rpy, [float(gripper)]]).astype(np.float32)


def compute_delta_action(pose_t: np.ndarray,
                         pose_next: np.ndarray,
                         gripper_next: int) -> np.ndarray:
    """Build one DELTA action [dx,dy,dz, droll,dpitch,dyaw, gripper_target].

    Position delta is the base-frame difference p_next - p_t (correct for
    translation). Orientation delta is the RELATIVE rotation R_next @ R_t^T,
    computed via the frame math in helpers/transforms.py (NOT naive Euler
    subtraction, which is wrong near gimbal lock). Gripper term is the ABSOLUTE
    target at t+1 (0/1), a common convention for parallel-jaw grippers.

    Both deltas are expressed in the base frame.

    Args:
        pose_t:       (6,) TCP pose at time t   (axis-angle).
        pose_next:    (6,) TCP pose at time t+1 (axis-angle).
        gripper_next: gripper state at t+1.

    Returns:
        (7,) float32 action vector.
    """
    dpos = pose_next[:3] - pose_t[:3]                       # base-frame translation delta

    # base-frame relative rotation R_next @ R_t^T, obtained by chaining pose_next
    # with the inverse of pose_t; we keep only its rotation part (indices 3:).
    rel = pose_trans(pose_next, pose_inv(pose_t))
    drpy = rotvec_to_euler(rel[3:])

    return np.concatenate([dpos, drpy, [float(gripper_next)]]).astype(np.float32)


def infer_fps(timestamps: np.ndarray, fallback: float = 20.0) -> int:
    """Estimate capture rate from the recorded timestamps (median dt -> Hz)."""
    if len(timestamps) < 2:
        return int(fallback)
    dt = float(np.median(np.diff(timestamps)))
    return int(round(1.0 / dt)) if dt > 0 else int(fallback)


def convert(episodes_dir: str, repo_id: str, task: str,
            camera: str = "top", fps: int | None = None,
            root: str | None = None) -> None:
    """Convert every HDF5 episode in a directory into one LeRobot dataset."""
    # LeRobot 0.6.1 (confirmed on this machine): v3.x dataset layout lives at
    # lerobot.datasets.lerobot_dataset. Imported here, not at module level, so
    # the maths above stays usable without lerobot installed.
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    from fyp.demos.hdf5_store import episode_paths

    paths = episode_paths(episodes_dir)
    if not paths:
        raise FileNotFoundError(f"No .h5/.hdf5 episodes found in {episodes_dir}")

    # Peek at the first episode to fix image size and (if not given) the fps.
    with h5py.File(paths[0], "r") as f:
        H, W = f["images"].shape[1:3]
        fps = fps or infer_fps(f["timestamps"][:])

    # Feature schema. Every frame we add must have exactly these keys + shapes.
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
        use_videos=True,     # encode images as video (needs ffmpeg installed)
        root=root,           # None -> ~/.cache/huggingface/lerobot/<repo_id>
    )

    for ep_path in paths:
        with h5py.File(ep_path, "r") as f:
            tcp = f["tcp_poses"][:]         # (N, 6) axis-angle
            grip = f["gripper_states"][:]   # (N,)
            imgs = f["images"][:]           # (N, H, W, 3) uint8
        N = len(tcp)

        for t in range(N):
            if t < N - 1:                                   # normal frame: delta to t+1
                action = compute_delta_action(tcp[t], tcp[t + 1], int(grip[t + 1]))
            else:                                           # last frame has no t+1: hold (zero delta)
                action = np.concatenate([np.zeros(6), [float(grip[t])]]).astype(np.float32)

            frame = {
                "observation.state": build_state(tcp[t], int(grip[t])),
                "action": action,
                f"observation.images.{camera}": imgs[t],    # (H, W, 3) uint8
                "task": task,                                # 0.6.1: task is per-frame, inside the dict
            }
            dataset.add_frame(frame)                         # 0.6.1 add_frame(frame) takes no task kwarg

        dataset.save_episode()
        print(f"converted {ep_path.name}: {N} frames")

    dataset.finalize()   # 0.6.1: finalize() closes the writers (no consolidate() here)
    print(f"done: {len(paths)} episodes -> {repo_id}  (fps={fps})")
