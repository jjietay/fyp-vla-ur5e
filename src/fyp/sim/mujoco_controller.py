"""MuJoCo backend for URController — same public API as the ur_rtde controller.

Sim-only. Drives the UR5e in MuJoCo so that DemoRecorder, scripted primitives,
and other downstream code can run identically against sim or the real robot.

Gripper convention: 0 = closed, 1 = open (matches DemoRecorder / real controller).
TCP pose format: [x, y, z, rx, ry, rz] (axis-angle), matching RTDE.
Interpolation: constant-velocity, capped at `speed` (rad/s); `acc` accepted
but ignored (B1).
"""

from pathlib import Path

import numpy as np
import mujoco

from fyp.sim.ik import solve_ik


class URControllerMuJoCo:
    def __init__(
        self,
        scene_path: str | Path,
        default_speed: float = 1.0,   # rad/s, joint velocity cap
        default_acc: float = 1.0,     # accepted for signature parity, ignored
        control_dt: float = 0.002,    # sim step for motion loop (500 Hz)
    ):
        self.model = mujoco.MjModel.from_xml_path(str(scene_path))
        self.data = mujoco.MjData(self.model)

        self._tcp_site_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_SITE, "attachment_site"
        )
        if self._tcp_site_id == -1:
            raise RuntimeError("Site 'attachment_site' not found in model.")

        self._gripper_state = 0
        self.default_speed = default_speed
        self.default_acc = default_acc
        self.control_dt = control_dt

        # Snap to home keyframe (id 0) and compute derived quantities.
        mujoco.mj_resetDataKeyframe(self.model, self.data, 0)
        self.data.ctrl[:6] = self.data.qpos[:6]
        mujoco.mj_forward(self.model, self.data)

    # ---- state reads -------------------------------------------------------

    def _tcp_pose(self) -> list[float]:
        """Current TCP pose as [x, y, z, rx, ry, rz] (axis-angle)."""
        pos = self.data.site_xpos[self._tcp_site_id].copy()

        mat = self.data.site_xmat[self._tcp_site_id].copy()  # (9,)
        quat = np.zeros(4)
        mujoco.mju_mat2Quat(quat, mat)

        angle = 2.0 * np.arccos(np.clip(quat[0], -1.0, 1.0))
        s = np.sqrt(max(1.0 - quat[0] ** 2, 1e-12))
        if s < 1e-8:
            rvec = np.zeros(3)
        else:
            rvec = (quat[1:] / s) * angle

        return [*pos.tolist(), *rvec.tolist()]

    def get_state(self) -> dict:
        return {
            "joint_pos": self.data.qpos[:6].copy().tolist(),
            "tcp_pose": self._tcp_pose(),
            "gripper_state": self._gripper_state,
        }

    # ---- gripper (stubbed — no physical gripper in the MJCF) ---------------

    def gripper_start(self, pin_power: int | None = 1, pin_control: int | None = 2):
        """No-op in sim. Kept for API parity with the real controller."""
        return "Gripper initialized (sim stub)."

    def gripper_toggle(self, state: int):
        if state in (0, 1):
            self._gripper_state = state
        else:
            return "Unable to control gripper."

    # ---- motion ------------------------------------------------------------

    def move_joints(
        self,
        q,
        speed: float | None = None,
        acc: float | None = None,
    ) -> bool:
        """Constant-velocity interpolate current -> q, stepping the sim.

        `speed` caps joint velocity (rad/s). `acc` accepted but ignored (B1).
        Blocks until the target is reached. Returns True on completion.
        """
        speed = speed if speed is not None else self.default_speed
        q_target = np.asarray(q, dtype=float)
        q_start = self.data.qpos[:6].copy()

        delta = q_target - q_start
        max_joint_move = float(np.max(np.abs(delta)))
        if max_joint_move < 1e-6:
            return True

        # Duration set by the slowest joint at the velocity cap.
        duration = max_joint_move / speed
        n_steps = max(int(np.ceil(duration / self.control_dt)), 1)

        for i in range(1, n_steps + 1):
            alpha = i / n_steps
            q_cmd = q_start + alpha * delta
            self.data.ctrl[:6] = q_cmd
            mujoco.mj_step(self.model, self.data)

        # Settle so qpos matches the final command.
        self.data.ctrl[:6] = q_target
        for _ in range(50):
            mujoco.mj_step(self.model, self.data)

        return True

    def move_to_pose(
        self,
        pose,
        speed: float | None = None,
        acc: float | None = None,
    ) -> bool:
        """Move so the TCP reaches `pose` = [x, y, z, rx, ry, rz] (axis-angle).

        Solves IK to joint targets, then delegates to move_joints.
        """
        pose = np.asarray(pose, dtype=float)
        target_pos = pose[:3]

        rvec = pose[3:6]
        angle = float(np.linalg.norm(rvec))
        if angle < 1e-8:
            target_mat = np.eye(3)
        else:
            axis = rvec / angle
            quat = np.zeros(4)
            mujoco.mju_axisAngle2Quat(quat, axis, angle)
            mat_flat = np.zeros(9)
            mujoco.mju_quat2Mat(mat_flat, quat)
            target_mat = mat_flat.reshape(3, 3)

        q_sol, ok = solve_ik(
            self.model,
            self.data,
            self._tcp_site_id,
            target_pos,
            target_mat,
            q_init=self.data.qpos[:6].copy(),
        )
        if not ok:
            raise RuntimeError("IK did not converge for the requested pose.")

        return self.move_joints(q_sol, speed=speed, acc=acc)

    # ---- lifecycle ---------------------------------------------------------

    def close(self):
        """No external resources in sim. Kept for API parity."""
        return None