"""Rotation and pose conversions for the FYP robotics stack.

A "pose" here is a 6-vector (x, y, z, rx, ry, rz): position in metres plus an
axis-angle rotation vector (the convention used by ur_rtde and by tcp_pose).
This module is the single source of truth for converting between rotation
vectors, rotation matrices, quaternions, Euler angles, and 4x4 homogeneous
transforms.
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
    Example (Week 6): camera_from_base = pose_inv(base_from_camera).

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