"""Damped least-squares inverse kinematics for the UR5e in MuJoCo.

Sim-only. Solves for joint angles that place the TCP site at a target
pose (position + orientation), using the site Jacobian.
"""

import numpy as np
import mujoco


def _quat_to_axis_angle(quat: np.ndarray) -> np.ndarray:
    """MuJoCo quat [w, x, y, z] -> rotation vector [rx, ry, rz]."""
    quat = quat / np.linalg.norm(quat)
    angle = 2.0 * np.arccos(np.clip(quat[0], -1.0, 1.0))
    s = np.sqrt(max(1.0 - quat[0] ** 2, 1e-12))
    if s < 1e-8:
        return np.zeros(3)
    axis = quat[1:] / s
    return axis * angle


def _pose_error(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    site_id: int,
    target_pos: np.ndarray,
    target_mat: np.ndarray,
) -> np.ndarray:
    """6-vector error [dpos(3), drot(3)] between current site pose and target."""
    cur_pos = data.site_xpos[site_id]
    pos_err = target_pos - cur_pos

    cur_mat = data.site_xmat[site_id].reshape(3, 3)
    # Rotation error: R_err = R_target * R_cur^T, expressed as rotation vector
    r_err_mat = target_mat @ cur_mat.T
    quat = np.zeros(4)
    mujoco.mju_mat2Quat(quat, r_err_mat.flatten())
    rot_err = _quat_to_axis_angle(quat)

    return np.concatenate([pos_err, rot_err])


def solve_ik(
    model: mujoco.MjModel, # it defines a tree and sites (attachment site, etc)
    data: mujoco.MjData, # this contains joint position now, joint vel now, actuator commands, computed world pose of TCP site
    site_id: int,
    target_pos: np.ndarray,
    target_mat: np.ndarray,
    q_init: np.ndarray | None = None,
    max_iters: int = 100,
    tol: float = 1e-4,
    damping: float = 1e-2,
    step_scale: float = 1.0,
) -> tuple[np.ndarray, bool]:
    """Solve IK for the TCP site.

    Args:
        model, data: MuJoCo handles (data is mutated during solve).
        site_id: id of the TCP site.
        target_pos: (3,) target position in world frame.
        target_mat: (3, 3) target rotation matrix in world frame.
        q_init: (6,) seed joint config; defaults to current qpos.
        max_iters: max Gauss-Newton iterations.
        tol: convergence threshold on the pose-error norm.
        damping: Levenberg-Marquardt damping (avoids singularities).
        step_scale: fraction of the computed step to apply per iter.

    Returns:
        (q_solution (6,), converged: bool)
    """
    if q_init is None:
        q = data.qpos[:6].copy()
    else:
        q = np.asarray(q_init, dtype=float).copy()

    jacp = np.zeros((3, model.nv))
    jacr = np.zeros((3, model.nv))

    for _ in range(max_iters):
        data.qpos[:6] = q
        mujoco.mj_forward(model, data)

        err = _pose_error(model, data, site_id, target_pos, target_mat)
        if np.linalg.norm(err) < tol:
            return q, True

        mujoco.mj_jacSite(model, data, jacp, jacr, site_id)
        J = np.vstack([jacp[:, :6], jacr[:, :6]])  # (6, 6)

        # Damped least squares: dq = J^T (J J^T + lambda^2 I)^-1 err
        JJt = J @ J.T
        dq = J.T @ np.linalg.solve(JJt + (damping ** 2) * np.eye(6), err)
        q = q + step_scale * dq

        # Clip to joint limits
        q = np.clip(q, model.jnt_range[:6, 0], model.jnt_range[:6, 1])

    return q, False