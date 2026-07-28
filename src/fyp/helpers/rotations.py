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
    """Convert a MuJoCo quaternion [w, x, y, z] to a rotation vector [rx, ry, rz].

    A unit quaternion stores a rotation of angle theta about unit axis a as
    w = cos(theta/2) and (x, y, z) = sin(theta/2) * a. We invert that: read
    the angle back out of w, read the axis back out of the vector part, and
    return axis * angle (the rotation-vector / axis-angle form).

    Args:
        quat: array-like of shape (4,), ordered [w, x, y, z]. Need not be
            unit-length; it is normalised internally.

    Returns:
        np.ndarray of shape (3,): the rotation vector [rx, ry, rz], whose
        direction is the rotation axis and whose length is the angle in radians.
    """
    quat = quat / np.linalg.norm(quat)                 # force unit length; the identities above only hold for unit quats
    angle = 2.0 * np.arccos(np.clip(quat[0], -1.0, 1.0))  # theta = 2*arccos(w); clip guards arccos against tiny float overshoots past +/-1
    s = np.sqrt(max(1.0 - quat[0] ** 2, 1e-12))        # s = sin(theta/2) = sqrt(1 - w^2); the max(...) floor stops a divide-by-zero
    if s < 1e-8:                                        # near-zero rotation: axis is undefined and the angle is ~0 anyway
        return np.zeros(3)                             # so the rotation vector is just (0, 0, 0)
    axis = quat[1:] / s                                # recover the unit axis a by removing the sin(theta/2) scaling
    return axis * angle                                # rotation vector = angle * axis


def rotvec_to_R(rotvec: np.ndarray) -> np.ndarray:
    """Convert a rotation vector [rx, ry, rz] to a 3x3 rotation matrix.

    Uses Rodrigues' rotation formula. A rotation vector encodes "rotate by
    angle theta about unit axis k", where theta = |rotvec| and k = rotvec/theta.
    Rodrigues assembles the matrix performing that rotation:

        R = I + sin(theta) * K + (1 - cos(theta)) * (K @ K)

    where K is the skew-symmetric ("cross-product") matrix of the unit axis k,
    i.e. K @ v == k x v for any vector v.

    Args:
        rotvec: array-like of shape (3,); its direction is the rotation axis and
                its length is the rotation angle in radians.

    Returns:
        np.ndarray of shape (3, 3): an orthonormal rotation matrix.
    """
    rotvec = np.asarray(rotvec, dtype=float)     # accept lists/tuples; force float maths
    theta = np.linalg.norm(rotvec)               # angle = length of the rotation vector

    if theta < 1e-8:                             # (near-)zero rotation: the axis is undefined...
        return np.eye(3)                         # ...but "rotate by nothing" is exactly the identity

    k = rotvec / theta                           # peel off the angle to get the unit axis
    K = np.array([                               # skew-symmetric matrix of k, so that K @ v == k x v
        [0.0,   -k[2],  k[1]],
        [k[2],   0.0,  -k[0]],
        [-k[1],  k[0],  0.0],
    ])
    return (                                     # Rodrigues, assembled term by term:
        np.eye(3)                                #   I         - start from no rotation
        + np.sin(theta) * K                      #   + swing   - the perpendicular tilt
        + (1.0 - np.cos(theta)) * (K @ K)        #   + correct - curve back onto the rotation cone
    )


def R_to_rotvec(R: np.ndarray) -> np.ndarray:
    """Convert a 3x3 rotation matrix to a rotation vector [rx, ry, rz].

    Inverts Rodrigues. The angle comes from the trace
    (trace(R) = 1 + 2*cos(theta)); the axis comes from the antisymmetric part
    (R - R^T = 2*sin(theta)*K). Two singular cases need special handling:

      * theta ~ 0  : no rotation, axis undefined -> return zeros.
      * theta ~ pi : sin(theta) ~ 0 so the antisymmetric trick fails; the axis
                     is recovered from the symmetric part (R + I = 2 * k k^T),
                     with signs fixed from the off-diagonals.

    Args:
        R: array-like of shape (3, 3), assumed a proper (orthonormal) rotation.

    Returns:
        np.ndarray of shape (3,): the rotation vector (axis * angle).
    """
    R = np.asarray(R, dtype=float)

    cos_theta = (np.trace(R) - 1.0) / 2.0            # trace(R) = 1 + 2 cos(theta)
    cos_theta = np.clip(cos_theta, -1.0, 1.0)        # guard arccos against float overshoot past +/-1
    theta = np.arccos(cos_theta)                     # the rotation angle

    if theta < 1e-8:                                 # case 1: no rotation
        return np.zeros(3)                           # axis undefined but answer is unambiguous

    sin_theta = np.sin(theta)
    if abs(sin_theta) > 1e-6:                        # generic case: axis from the antisymmetric part
        axis = np.array([
            R[2, 1] - R[1, 2],                       # these three differences are the
            R[0, 2] - R[2, 0],                       # entries of R - R^T = 2 sin(theta) K,
            R[1, 0] - R[0, 1],                       # which encode 2 sin(theta) * axis
        ]) / (2.0 * sin_theta)
        return axis * theta                          # rotation vector = angle * unit axis

    # case 2: theta ~ pi. sin(theta) ~ 0, so use the symmetric part instead.
    # at theta = pi, R + I = 2 k k^T  =>  k_i = sqrt((R[i,i] + 1) / 2)
    k = np.sqrt(np.clip((np.diag(R) + 1.0) / 2.0, 0.0, None))   # magnitudes of axis components
    i = int(np.argmax(k))                            # anchor on the largest (most reliable) component
    if i == 0:
        k[1] = np.copysign(k[1], R[0, 1])            # sign of k_j from sign of (k_i * k_j) = R[i, j]/...
        k[2] = np.copysign(k[2], R[0, 2])
    elif i == 1:
        k[0] = np.copysign(k[0], R[0, 1])
        k[2] = np.copysign(k[2], R[1, 2])
    else:
        k[0] = np.copysign(k[0], R[0, 2])
        k[1] = np.copysign(k[1], R[1, 2])
    k = k / np.linalg.norm(k)                        # renormalise to a clean unit axis
    return k * theta


def rotvec_to_euler(rotvec: np.ndarray) -> np.ndarray:
    """Convert an axis-angle rotation vector to ZYX Euler angles [roll, pitch, yaw].

    Reuses rotvec_to_R to get the rotation matrix, then decomposes it with the
    standard ZYX (aerospace RPY) convention  R = Rz(yaw) @ Ry(pitch) @ Rx(roll),
    with a guard for the gimbal-lock case (pitch ~ +/-90 deg). The Euler order
    is a CHOICE: whatever consumes these angles (e.g. the LeRobot converter's
    state/action vectors) must use this same convention consistently.

    Args:
        rotvec: (3,) axis-angle rotation vector.

    Returns:
        (3,) array [roll, pitch, yaw] in radians.
    """
    R = rotvec_to_R(rotvec)
    cy = np.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)    # cos(pitch): shrinks to 0 at gimbal lock
    if cy > 1e-6:                                  # generic (non-degenerate) case
        roll = np.arctan2(R[2, 1], R[2, 2])
        pitch = np.arctan2(-R[2, 0], cy)
        yaw = np.arctan2(R[1, 0], R[0, 0])
    else:                                          # gimbal lock: pitch ~ +/-90 deg, yaw/roll couple
        roll = np.arctan2(-R[1, 2], R[1, 1])
        pitch = np.arctan2(-R[2, 0], cy)
        yaw = 0.0
    return np.array([roll, pitch, yaw])
