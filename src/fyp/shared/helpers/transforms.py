""" transforms.py

This file contains the conversions between:
1) Pose to Transform
2) Transform to Pose
3) Pose Transformation (T @ T)
4) Finding the inverse of a Pose
5) Moving a POINT into another frame (transform_point)

Pose:
- stands for the 6-vector (x,y,z,rx,ry,rz)
- contains the position in metres
- axis-angle rotation vector (rx,ry,rz)

Transform:
- in the form of 4x4 homogeneuous matrix
-   [R11 R12 R13 tx]
    [R21 R22 R23 ty]
    [R31 R32 R33 tz]
    [ 0   0   0   1]
- 3x3 matrix at top left == rotation matrix (R) with regards to global frame
- tx,ty,tz is the translation vector with regards to the global frame

"""

import numpy as np
from fyp.shared.helpers.rotations import R_to_rotvec, rotvec_to_R


def pose_to_T(pose: np.ndarray) -> np.ndarray:
    """
    This function converts pose to a transform, used as intermediate step
    for calculating transformation of pose:
    T_new = T_old @ T_change
    """
    pose = np.asarray(pose, dtype=float)
    T = np.eye(4)
    T[:3, :3] = rotvec_to_R(pose[3:])
    T[:3, 3] = pose[:3]
    return T


def T_to_pose(T: np.ndarray) -> np.ndarray:
    """
    This function converts Transform to pose
    """
    T = np.asarray(T, dtype=float)
    position = T[:3, 3]
    rotvec = R_to_rotvec(T[:3, :3])
    return np.concatenate([position, rotvec])


def pose_trans(pose_current: np.ndarray, pose_change: np.ndarray) -> np.ndarray:
    """
    This function takes in current pose, and a pose change (with regards to previous pose's frame),
    converts all poses to tranforms so that we can use @, matrix multiplication to calculate new pose
    """
    T = pose_to_T(pose_current) @ pose_to_T(pose_change)
    return T_to_pose(T)


def transform_point(T: np.ndarray, p: np.ndarray) -> np.ndarray:
    """
    This function takes a 4x4 transform and a 3D point (or an (N,3) stack of
    points) and gives you that point expressed in the other frame.

    Points are not poses. A pose carries an orientation, so it needs the full
    4x4 matrix product. A point has no orientation to carry, so it only gets
    rotated and shifted: R @ p + t.

    Note it takes a TRANSFORM (4x4), not a pose, unlike everything else in this
    file. Use pose_to_T first if what you have is a pose.
    """
    T = np.asarray(T, dtype=float)
    p = np.asarray(p, dtype=float)
    R, t = T[:3, :3], T[:3, 3]
    return p @ R.T + t if p.ndim > 1 else R @ p + t


def pose_inv(pose: np.ndarray) -> np.ndarray:
    """
    This function finds the inverse of a pose or transform. Something like an undo button to
    undo a pose change to get back to the start.
    """
    T = pose_to_T(pose)
    R = T[:3, :3]
    t = T[:3, 3]

    T_inv = np.eye(4)
    T_inv[:3, :3] = R.T
    T_inv[:3, 3] = -R.T @ t
    return T_to_pose(T_inv)
