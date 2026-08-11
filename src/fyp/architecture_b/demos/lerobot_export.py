""" lerobot_export.py

Convert recorded HDF5 episodes into a LeRobot dataset for SmolVLA.

Bridges two formats. DemoRecorder writes ABSOLUTE states with AXIS-ANGLE
orientation; SmolVLA / LeRobot want per-frame observations plus DELTA actions
with orientation in EULER angles. Format-agnostic by construction: the same
code converts every episode regardless of how it was teleoperated, because all share
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

from fyp.shared.helpers.rotations import rotvec_to_euler
from fyp.shared.helpers.transforms import pose_inv, pose_trans


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


REQUIRED_DATASETS = ("tcp_poses", "gripper_states", "images", "timestamps")


class MalformedEpisode(Exception):
    """Raised when an episode is unusable. Never swallowed: a bad episode must not be exported."""


def validate_episode(path, expect_shape: tuple | None = None,
                     expect_fps: int | None = None, fps_tolerance: float = 0.15) -> dict:
    """
    It takes an episode file and gives you its vital statistics, or raises
    MalformedEpisode explaining what is wrong with it.

    Exporting a truncated episode does not fail. It produces a dataset that
    trains, converges to something, and behaves oddly on the robot, and by then
    the cause is twenty hours of lab time in the past. Every check here is
    cheaper than that.
    """
    stats: dict = {"path": str(path)}
    try:
        with h5py.File(path, "r") as f:
            missing = [k for k in REQUIRED_DATASETS if k not in f]
            if missing:
                raise MalformedEpisode(f"{path.name}: missing datasets {missing}")

            lengths = {k: len(f[k]) for k in REQUIRED_DATASETS}
            if len(set(lengths.values())) != 1:
                raise MalformedEpisode(
                    f"{path.name}: datasets disagree on length {lengths}. "
                    "Usually means the recorder was killed mid-write.")

            n = next(iter(lengths.values()))
            if n < 2:
                raise MalformedEpisode(f"{path.name}: only {n} frames, nothing to learn from")

            tcp = f["tcp_poses"][:]
            if tcp.ndim != 2 or tcp.shape[1] != 6:
                raise MalformedEpisode(f"{path.name}: tcp_poses is {tcp.shape}, expected (N, 6)")
            if not np.all(np.isfinite(tcp)):
                bad = int(np.count_nonzero(~np.isfinite(tcp)))
                raise MalformedEpisode(f"{path.name}: {bad} non-finite values in tcp_poses")

            ts = f["timestamps"][:]
            if np.any(np.diff(ts) <= 0):
                raise MalformedEpisode(
                    f"{path.name}: timestamps are not strictly increasing; "
                    "the recorder wrote frames out of order")

            shape = tuple(f["images"].shape[1:3])
            if expect_shape is not None and shape != expect_shape:
                raise MalformedEpisode(
                    f"{path.name}: images are {shape}, but the dataset was created "
                    f"for {expect_shape}. Mixing resolutions silently degrades training.")

            actual_fps = infer_fps(ts)
            if expect_fps is not None:
                drift = abs(actual_fps - expect_fps) / float(expect_fps)
                if drift > fps_tolerance:
                    raise MalformedEpisode(
                        f"{path.name}: recorded at ~{actual_fps} Hz but the dataset is "
                        f"{expect_fps} Hz ({drift:.0%} off). Timestamps become wrong and "
                        "the policy learns the wrong control rate.")

            stats.update(frames=n, fps=actual_fps, shape=shape,
                         duration_s=round(float(ts[-1] - ts[0]), 2))
    except OSError as e:
        raise MalformedEpisode(f"{path.name}: cannot be opened, likely truncated ({e})") from e
    return stats


def convert(episodes_dir: str, repo_id: str, task: str,
            camera: str = "top", fps: int | None = None,
            root: str | None = None) -> None:


    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    from fyp.architecture_b.demos.hdf5_store import episode_paths

    paths = episode_paths(episodes_dir)
    if not paths:
        raise FileNotFoundError(f"No .h5/.hdf5 episodes found in {episodes_dir}")


    # Validate everything BEFORE creating the dataset. Failing on episode 47 of
    # 50 leaves a half-written dataset directory that has to be cleaned up by
    # hand, and the whole point is to fail before any of that exists.
    with h5py.File(paths[0], "r") as f:
        H, W = f["images"].shape[1:3]
        fps = int(fps or infer_fps(f["timestamps"][:]))

    stats = [validate_episode(p, expect_shape=(H, W), expect_fps=fps) for p in paths]
    total = sum(s["frames"] for s in stats)
    print(f"validated {len(stats)} episodes, {total} frames, {H}x{W} at {fps} Hz")


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
    print(f"done: {len(paths)} episodes, {total} frames -> {repo_id}  (fps={fps})")
