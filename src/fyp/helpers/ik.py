"""Damped least-squares inverse kinematics for the UR5e in MuJoCo.

Sim-only, and the one module in helpers/ that imports mujoco. It exists purely
because MuJoCo has no built-in IK; the real UR5e solves IK in firmware, so this
leaves the project when the sim does.

Solves for joint angles that place the TCP site at a target pose (position +
orientation), using the site Jacobian.
"""

import numpy as np
import mujoco

from fyp.helpers.rotations import quat_to_rotvec


def _pose_error(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    site_id: int,
    target_pos: np.ndarray,
    target_mat: np.ndarray,
) -> np.ndarray:
    cur_pos = data.site_xpos[site_id]
    pos_err = target_pos - cur_pos

    cur_mat = data.site_xmat[site_id].reshape(3, 3)

    r_err_mat = target_mat @ cur_mat.T
    quat = np.zeros(4)
    mujoco.mju_mat2Quat(quat, r_err_mat.flatten())
    rot_err = quat_to_rotvec(quat)

    return np.concatenate([pos_err, rot_err])


def solve_ik(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    site_id: int,
    target_pos: np.ndarray,
    target_mat: np.ndarray,
    q_init: np.ndarray | None = None,
    max_iters: int = 100,
    tol: float = 1e-4,
    damping: float = 1e-2,
    step_scale: float = 1.0,
) -> tuple[np.ndarray, bool]:
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
        J = np.vstack([jacp[:, :6], jacr[:, :6]])


        JJt = J @ J.T
        dq = J.T @ np.linalg.solve(JJt + (damping ** 2) * np.eye(6), err)
        q = q + step_scale * dq


        q = np.clip(q, model.jnt_range[:6, 0], model.jnt_range[:6, 1])

    return q, False
