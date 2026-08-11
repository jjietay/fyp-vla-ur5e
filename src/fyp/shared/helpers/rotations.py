""" rotations.py

This file contains only functions that are conversions between:
1) quaternions to rotation vectors
2) rotation vectors to rotation matrix (R)
3) rotation matrix (R) to rotation vector
4) rotation vector to euler

Difference between rotation vector and rotation matrix is:
- rotation matrix is 3x3 that represents rotation in global coordinates
- rotation vector is 3x1 that simply encodes rotation magnitude about an axis of rotation

UR5e's ur_rtde uses axis-angle (x,y,z,rx,ry,rz) representation. (rx,ry,rz) is the rotation vector that
encodes the axis of rotation and the angle of rotation about that axis,
while (x,y,z) is the position in metres.

This file is the single source of truth for these conversions.
"""

import numpy as np


def quat_to_rotvec(quat: np.ndarray) -> np.ndarray:
    """
    This function convert quaternions to rotation vectors
    """

    quat = quat / np.linalg.norm(quat)
    angle = 2.0 * np.arccos(np.clip(quat[0], -1.0, 1.0))
    s = np.sqrt(max(1.0 - quat[0] ** 2, 1e-12))
    if s < 1e-8:
        return np.zeros(3)
    axis = quat[1:] / s
    return axis * angle


def rotvec_to_R(rotvec: np.ndarray) -> np.ndarray:
    """
    This function convert rotation vector to rotation matrix
    """

    rotvec = np.asarray(rotvec, dtype=float)
    theta = np.linalg.norm(rotvec)

    if theta < 1e-8:
        return np.eye(3)

    k = rotvec / theta
    K = np.array([
        [0.0,   -k[2],  k[1]],
        [k[2],   0.0,  -k[0]],
        [-k[1],  k[0],  0.0],
    ])
    return (
        np.eye(3)
        + np.sin(theta) * K
        + (1.0 - np.cos(theta)) * (K @ K)
    )


def R_to_rotvec(R: np.ndarray) -> np.ndarray:
    """
    It takes a 3x3 rotation matrix and gives you the equivalent UR rotation vector,
    with the angle in [0, pi].

    Goes via a quaternion using Shepperd's largest-pivot method rather than the
    textbook axis = (R - R.T) / (2 sin theta). That formula divides by sin theta,
    which collapses as theta approaches pi, and pi is not an edge case here: the
    default tool orientation is the gripper pointing straight down, whose rotation
    vector is about [3.142, 0, 0], i.e. theta = pi almost exactly. The old
    implementation lost roughly seven digits there. Choosing the pivot from the
    largest diagonal element keeps every division well conditioned for any input.
    """
    R = np.asarray(R, dtype=float)
    if R.shape != (3, 3):
        raise ValueError(f"expected a 3x3 rotation matrix, got {R.shape}")

    trace = R[0, 0] + R[1, 1] + R[2, 2]
    if trace > 0.0:
        s = np.sqrt(trace + 1.0) * 2.0
        w = 0.25 * s
        v = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
        w = (R[2, 1] - R[1, 2]) / s
        v = np.array([0.25 * s, (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s])
    elif R[1, 1] > R[2, 2]:
        s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
        w = (R[0, 2] - R[2, 0]) / s
        v = np.array([(R[0, 1] + R[1, 0]) / s, 0.25 * s, (R[1, 2] + R[2, 1]) / s])
    else:
        s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
        w = (R[1, 0] - R[0, 1]) / s
        v = np.array([(R[0, 2] + R[2, 0]) / s, (R[1, 2] + R[2, 1]) / s, 0.25 * s])

    # q and -q are the same rotation; pick w >= 0 so the angle lands in [0, pi]
    # instead of coming back as its 2pi complement.
    if w < 0.0:
        w, v = -w, -v

    norm = float(np.linalg.norm(v))
    if norm < 1e-15:
        return np.zeros(3)
    theta = 2.0 * np.arctan2(norm, w)
    return v / norm * theta


def rotvec_to_euler(rotvec: np.ndarray) -> np.ndarray:
    """
    This function convert rotation vector to euler representation
    """

    R = rotvec_to_R(rotvec)
    cy = np.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
    if cy > 1e-6:
        roll = np.arctan2(R[2, 1], R[2, 2])
        pitch = np.arctan2(-R[2, 0], cy)
        yaw = np.arctan2(R[1, 0], R[0, 0])
    else:
        roll = np.arctan2(-R[1, 2], R[1, 1])
        pitch = np.arctan2(-R[2, 0], cy)
        yaw = 0.0
    return np.array([roll, pitch, yaw])


def euler_to_R(euler: np.ndarray) -> np.ndarray:
    """
    It takes roll, pitch and yaw in radians and gives you the 3x3 rotation matrix.

    Convention is intrinsic Z-Y-X, so R = Rz(yaw) @ Ry(pitch) @ Rx(roll). That is
    the exact inverse of what `rotvec_to_euler` decomposes, and the two are
    round-trip tested against each other. Euler conventions are a well-known
    source of silent frame bugs, so do not change this one without changing that
    function and the tests together.
    """
    roll, pitch, yaw = (float(v) for v in np.asarray(euler, dtype=float).reshape(3))
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)

    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]], dtype=float)
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]], dtype=float)
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]], dtype=float)
    return Rz @ Ry @ Rx


def euler_to_rotvec(euler: np.ndarray) -> np.ndarray:
    """
    It takes roll, pitch and yaw in radians and gives you the UR rotation vector.

    The only orientation format `moveL` accepts is a rotation vector, so anything
    that reasons in Euler angles has to come back through here before it can be
    commanded.
    """
    return R_to_rotvec(euler_to_R(euler))
