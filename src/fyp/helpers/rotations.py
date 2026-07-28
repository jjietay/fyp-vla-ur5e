"""Rotation conversions: quaternion, rotation vector, matrix, Euler.

A rotation vector (axis-angle) [rx, ry, rz] is the project's canonical
orientation format: it is what ur_rtde uses for tcp_pose, so keeping it as the
hub minimises conversions on the real robot.

This module is the single source of truth for these conversions. Three separate
copies of quaternion -> rotation vector existed before (transforms.py, ik.py,
mujoco_controller.py); they now all call `quat_to_rotvec` here.
"""

import numpy as np


def quat_to_rotvec(quat: np.ndarray) -> np.ndarray:
    quat = quat / np.linalg.norm(quat)
    angle = 2.0 * np.arccos(np.clip(quat[0], -1.0, 1.0))
    s = np.sqrt(max(1.0 - quat[0] ** 2, 1e-12))
    if s < 1e-8:
        return np.zeros(3)
    axis = quat[1:] / s
    return axis * angle


def rotvec_to_R(rotvec: np.ndarray) -> np.ndarray:
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
    R = np.asarray(R, dtype=float)

    cos_theta = (np.trace(R) - 1.0) / 2.0
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    theta = np.arccos(cos_theta)

    if theta < 1e-8:
        return np.zeros(3)

    sin_theta = np.sin(theta)
    if abs(sin_theta) > 1e-6:
        axis = np.array([
            R[2, 1] - R[1, 2],
            R[0, 2] - R[2, 0],
            R[1, 0] - R[0, 1],
        ]) / (2.0 * sin_theta)
        return axis * theta


    k = np.sqrt(np.clip((np.diag(R) + 1.0) / 2.0, 0.0, None))
    i = int(np.argmax(k))
    if i == 0:
        k[1] = np.copysign(k[1], R[0, 1])
        k[2] = np.copysign(k[2], R[0, 2])
    elif i == 1:
        k[0] = np.copysign(k[0], R[0, 1])
        k[2] = np.copysign(k[2], R[1, 2])
    else:
        k[0] = np.copysign(k[0], R[0, 2])
        k[1] = np.copysign(k[1], R[1, 2])
    k = k / np.linalg.norm(k)
    return k * theta


def rotvec_to_euler(rotvec: np.ndarray) -> np.ndarray:
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
