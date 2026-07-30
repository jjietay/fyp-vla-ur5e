"""mujoco_controller.py

MuJoCo backend for URController with the same public API as the ur_rtde controller.

Sim-only. Drives the UR5e in MuJoCo so that the recorder, scripted primitives,
and other downstream code can run identically against sim or the real robot.

Key points:
* Gripper convention: 0 = closed, 1 = open (matches the recorder / real controller)
* The Robotiq 2F-85 actuator (index 6) takes ctrl 0..255: 0 = open, 255 = closed
* TCP pose format: [x, y, z, rx, ry, rz] (axis-angle), matching RTDE
* Interpolation: constant-velocity, capped at `speed` (rad/s); `acc` accepted but ignored (B1)
"""

from pathlib import Path

import numpy as np
import mujoco

from fyp.helpers.config import get_config, resolve
from fyp.helpers.ik import solve_ik
from fyp.helpers.rotations import quat_to_rotvec


_GRIPPER_ACT_IDX = 6


class URControllerMuJoCo:
    def __init__(
        self,
        scene_path: str | Path | None = None,
        default_speed: float | None = None,
        control_dt: float | None = None,
    ):
        cfg = get_config()["sim"]
        if scene_path is None:
            scene_path = resolve(cfg["scene"])

        self.model = mujoco.MjModel.from_xml_path(str(scene_path))
        self.data = mujoco.MjData(self.model)

        self._tcp_site = cfg["tcp_site"]
        self._tcp_site_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_SITE, self._tcp_site
        )
        if self._tcp_site_id == -1:
            raise RuntimeError(f"Site '{self._tcp_site}' not found in model.")


        self._has_gripper = self.model.nu > 6

        self._gripper_state = 0
        self.default_speed = (
            default_speed if default_speed is not None else cfg["motion"]["default_speed"]
        )
        self.control_dt = control_dt if control_dt is not None else cfg["control_dt"]
        self._open_ctrl = cfg["gripper"]["open_ctrl"]
        self._close_ctrl = cfg["gripper"]["close_ctrl"]
        self._ik_cfg = cfg["ik"]


        mujoco.mj_resetDataKeyframe(self.model, self.data, 0)
        self.data.ctrl[:6] = self.data.qpos[:6]

        if self._has_gripper:
            self.data.ctrl[_GRIPPER_ACT_IDX] = self._open_ctrl
            self._gripper_state = 1
        mujoco.mj_forward(self.model, self.data)


    def step(self) -> None:
        """Advance physics one control_dt with gravity compensation.

        The XML's position servos are PD only (kp=2000), so holding the arm up
        requires a standing position error of gravity_torque/kp - about 10 mm at
        the TCP. The real UR controller adds gravity feedforward internally, so
        cancelling qfrc_bias here is what makes sim match hardware rather than
        baking a systematic droop into every recorded demo.
        """
        self.data.qfrc_applied[:6] = self.data.qfrc_bias[:6]
        mujoco.mj_step(self.model, self.data)

    def settle(self, q_target, tol: float = 1e-4, max_steps: int = 2000) -> bool:
        """Step until the joints reach q_target, or give up. True if reached."""
        for _ in range(max_steps):
            if float(np.abs(self.data.qpos[:6] - q_target).max()) < tol:
                return True
            self.step()
        return False

    def _tcp_pose(self) -> list[float]:
        pos = self.data.site_xpos[self._tcp_site_id].copy()

        mat = self.data.site_xmat[self._tcp_site_id].copy()
        quat = np.zeros(4)
        mujoco.mju_mat2Quat(quat, mat)


        rvec = quat_to_rotvec(quat)

        return [*pos.tolist(), *rvec.tolist()]

    def get_state(self) -> dict:
        return {
            "joint_pos": self.data.qpos[:6].copy().tolist(),
            "tcp_pose": self._tcp_pose(),
            "gripper_state": self._gripper_state,
        }


    def gripper_start(self, pin_power: int | None = 1, pin_control: int | None = 2):
        return "Gripper initialized (sim stub)."

    def gripper_toggle(self, state: int):
        if state not in (0, 1):
            return "Unable to control gripper."

        self._gripper_state = state
        if self._has_gripper:
            self.data.ctrl[_GRIPPER_ACT_IDX] = (
                self._open_ctrl if state == 1 else self._close_ctrl
            )

            for _ in range(100):
                self.step()


    def move_joints(
        self,
        q,
        speed: float | None = None,
        acc: float | None = None,
    ) -> bool:
        speed = speed if speed is not None else self.default_speed
        q_target = np.asarray(q, dtype=float)
        q_start = self.data.qpos[:6].copy()

        delta = q_target - q_start
        max_joint_move = float(np.max(np.abs(delta)))
        if max_joint_move < 1e-6:
            return True


        duration = max_joint_move / speed
        n_steps = max(int(np.ceil(duration / self.control_dt)), 1)

        for i in range(1, n_steps + 1):
            alpha = i / n_steps
            q_cmd = q_start + alpha * delta
            self.data.ctrl[:6] = q_cmd
            self.step()


        self.data.ctrl[:6] = q_target
        return self.settle(q_target)

    def move_to_pose(
        self,
        pose,
        speed: float | None = None,
        acc: float | None = None,
    ) -> bool:
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
            **self._ik_cfg,
        )
        if not ok:
            raise RuntimeError("IK did not converge for the requested pose.")

        return self.move_joints(q_sol, speed=speed, acc=acc)


    def close(self):
        return None
