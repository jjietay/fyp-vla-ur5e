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
    """Convert a 6-vector pose [x, y, z, rx, ry, rz] into a 4x4 homogeneous transform.

    Packs translation and rotation into one matrix:

        [ R  R  R | x ]
        [ R  R  R | y ]
        [ R  R  R | z ]
        [ 0  0  0 | 1 ]

    so poses can be composed, and points transformed, by plain matrix multiply.
    The rotation block is built from the axis-angle part via rotvec_to_R.

    Args:
        pose: array-like of shape (6,): position (x, y, z) in metres, then the
              axis-angle rotation vector (rx, ry, rz).

    Returns:
        np.ndarray of shape (4, 4): the homogeneous transform T.
    """
    pose = np.asarray(pose, dtype=float)
    T = np.eye(4)                        # identity gives the [0,0,0,1] bottom row + corner 1 for free
    T[:3, :3] = rotvec_to_R(pose[3:])    # top-left 3x3 = rotation, from the (rx, ry, rz) part
    T[:3, 3] = pose[:3]                  # top-right column = translation (x, y, z)
    return T


def T_to_pose(T: np.ndarray) -> np.ndarray:
    """Convert a 4x4 homogeneous transform back to a 6-vector pose [x, y, z, rx, ry, rz].

    The inverse of pose_to_T: read the translation from the top-right column and
    convert the top-left 3x3 rotation block back to axis-angle via R_to_rotvec.

    Args:
        T: array-like of shape (4, 4), a homogeneous transform.

    Returns:
        np.ndarray of shape (6,): position (x, y, z) then rotation vector (rx, ry, rz).
    """
    T = np.asarray(T, dtype=float)
    position = T[:3, 3]                  # top-right column = translation (x, y, z)
    rotvec = R_to_rotvec(T[:3, :3])      # top-left 3x3 = rotation -> back to axis-angle
    return np.concatenate([position, rotvec])   # stitch into a single (6,) pose


def pose_trans(pose_from: np.ndarray, pose_from_to: np.ndarray) -> np.ndarray:
    """Compose two poses: apply pose_from_to *within* the frame of pose_from.

    Matches UR's pose_trans semantics. Both poses are turned into 4x4 transforms
    and multiplied (T_from @ T_from_to); the result is converted back to a
    6-vector pose. Reads as: "start at pose_from's frame, then move by
    pose_from_to expressed in that frame."

    Example:
        object_in_base = pose_trans(base_from_camera, camera_from_object)

    Args:
        pose_from:    (6,) pose of an intermediate frame, in some parent frame.
        pose_from_to: (6,) pose expressed *in* the pose_from frame.

    Returns:
        np.ndarray of shape (6,): the composed pose, expressed in the parent frame.
    """
    T = pose_to_T(pose_from) @ pose_to_T(pose_from_to)   # chain frames: parent-to-intermediate, then intermediate-to-target
    return T_to_pose(T)                                  # bring the 4x4 result back down to a 6-vector pose


def pose_inv(pose: np.ndarray) -> np.ndarray:
    """Invert a pose: return the transform that undoes it.

    For T = [[R, t], [0, 1]], the inverse is [[R^T, -R^T @ t], [0, 1]],
    using the fact that a rotation matrix's inverse is its transpose (R is
    orthonormal), so no general matrix inversion is needed.

    If pose is "B expressed in A", pose_inv(pose) is "A expressed in B".
    Example: camera_from_base = pose_inv(base_from_camera).

    Args:
        pose: array-like of shape (6,): position then axis-angle rotation vector.

    Returns:
        np.ndarray of shape (6,): the inverse pose.
    """
    T = pose_to_T(pose)
    R = T[:3, :3]
    t = T[:3, 3]

    T_inv = np.eye(4)               # identity again gives the [0,0,0,1] row for free
    T_inv[:3, :3] = R.T             # inverse rotation = transpose (R is orthonormal)
    T_inv[:3, 3] = -R.T @ t         # inverse translation: rotate t back, then negate
    return T_to_pose(T_inv)
