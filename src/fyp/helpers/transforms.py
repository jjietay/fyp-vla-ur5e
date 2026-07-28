"""Pose composition: 6-vector poses <-> 4x4 homogeneous transforms.

A "pose" here is a 6-vector (x, y, z, rx, ry, rz): position in metres plus an
axis-angle rotation vector (the convention used by ur_rtde and by tcp_pose).
Rotation-only conversions live in `rotations.py`; this module is about
composing, inverting and chaining whole frames — which is what hand-eye
calibration and the pick/place primitives need.
"""

import numpy as np

from fyp.helpers.rotations import R_to_rotvec, rotvec_to_R


def pose_to_T(pose: np.ndarray) -> np.ndarray:
    pose = np.asarray(pose, dtype=float)
    T = np.eye(4)
    T[:3, :3] = rotvec_to_R(pose[3:])
    T[:3, 3] = pose[:3]
    return T


def T_to_pose(T: np.ndarray) -> np.ndarray:
    T = np.asarray(T, dtype=float)
    position = T[:3, 3]
    rotvec = R_to_rotvec(T[:3, :3])
    return np.concatenate([position, rotvec])


def pose_trans(pose_from: np.ndarray, pose_from_to: np.ndarray) -> np.ndarray:
    T = pose_to_T(pose_from) @ pose_to_T(pose_from_to)
    return T_to_pose(T)


def pose_inv(pose: np.ndarray) -> np.ndarray:
    T = pose_to_T(pose)
    R = T[:3, :3]
    t = T[:3, 3]

    T_inv = np.eye(4)
    T_inv[:3, :3] = R.T
    T_inv[:3, 3] = -R.T @ t
    return T_to_pose(T_inv)
