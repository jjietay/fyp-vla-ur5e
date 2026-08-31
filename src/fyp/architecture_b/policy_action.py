"""policy_action.py

It takes one action from the policy and gives you the pose to command and the
gripper state to set.

This is the whole of Architecture B's inference logic, and it lives here rather
than in the LeRobot adapter so that it can be TESTED. The adapter imports
`lerobot`, which is deliberately absent from the FYP venv, so anything inside it
is unreachable from `tests/unit/`. Everything in this file is plain Python and
numpy, and `integrations/lerobot_robot_ur5e/` is a thin shell over it that adds
the LeRobot interface and the hardware calls.

That split is the point. The clamp, the delta arithmetic, the envelope check and
the gripper threshold are the parts that decide where the arm goes, and they are
now exercised by the same tests that would catch them being wrong.

Order of operations in `translate`, and it matters:

  clamp first, so the envelope sees the pose the arm will really be given
  apply the delta second, in the base frame
  check third, so an out of bounds target never reaches the controller

Doing the check before the clamp would pass a pose that is then modified, which
is the kind of thing that looks correct in review and is not.
"""

from __future__ import annotations

import logging

import numpy as np

from fyp.architecture_b.demos.lerobot_export import (ACTION_NAMES, STATE_NAMES,
                                                     apply_delta_action, build_state)
from fyp.shared.hardware.safety import WorkspaceEnvelope
from fyp.shared.helpers.config import get_config

logger = logging.getLogger(__name__)


class PolicyActionTranslator:
    """
    It takes the config and gives you the translator between a policy's numbers
    and the arm's numbers.

    Holds no hardware and performs no motion. Hand it the current TCP pose and an
    action, and it gives back what should be commanded, or raises
    `SafetyViolation` if that would leave the workspace.
    """

    def __init__(self, cfg: dict | None = None,
                 envelope: WorkspaceEnvelope | None = None):
        cfg = cfg or get_config()
        b = cfg["architecture_b"]
        w = cfg["workspace"]
        self.camera = str(b["camera"])
        self.gripper_threshold = float(b["gripper_threshold"])

        # DERIVED, never configured. One control period at Architecture A's speed
        # ceiling is the furthest a single action may move the arm, so B is held
        # to the same limit A is without anyone having to keep two numbers in
        # step. Writing the per-step distance directly is how B ended up briefly
        # permitted to move at 1.0 m/s against A's 0.25.
        self.control_period_s = float(cfg["robot"]["servo"]["time"])
        self.max_speed = float(w["max_speed"])
        self.max_angular_speed = float(w["max_angular_speed"])
        self.max_step_m = self.max_speed * self.control_period_s
        self.max_step_rad = self.max_angular_speed * self.control_period_s
        self.image_shape = (int(cfg["camera"]["height"]), int(cfg["camera"]["width"]), 3)
        self.env = envelope or WorkspaceEnvelope(cfg)

    # ------------------------------------------------------------- features

    @property
    def observation_features(self) -> dict:
        """
        It takes nothing and gives you the observation spec LeRobot expects.

        The scalar names come from `STATE_NAMES` rather than being retyped,
        because LeRobot assembles `observation.state` by looking each name from
        the DATASET up in the dict the robot returns. A name that matches by eye
        and not by string is a KeyError on the first observation.
        """
        features: dict = {name: float for name in STATE_NAMES}
        features[self.camera] = self.image_shape
        return features

    @property
    def action_features(self) -> dict:
        """
        It takes nothing and gives you the action spec, in wire order.

        Insertion order IS the wire order. LeRobot's client unpacks the policy's
        action tensor by the index of these keys, with no names involved, so a
        reordering raises nothing and sends roll to z.
        """
        return {name: float for name in ACTION_NAMES}

    # ---------------------------------------------------------- observation

    def observation(self, tcp_pose, gripper_state: int, rgb) -> dict:
        """
        It takes the arm state and a colour frame and gives you the flat
        observation dict LeRobot groups into state and images.

        State is built by the converter's own `build_state`, so the vector a
        policy sees at inference is produced by the same code that produced the
        vector it was trained on. Duplicating that arithmetic is how a project
        ends up with a policy that works on the dataset and not on the arm.
        """
        vector = build_state(np.asarray(tcp_pose, dtype=float), int(gripper_state))
        obs = {name: float(v) for name, v in zip(STATE_NAMES, vector)}
        obs[self.camera] = rgb
        return obs

    # --------------------------------------------------------------- action

    def clamp(self, action) -> np.ndarray:
        """
        It takes a requested delta and gives you one no larger than the per-step
        ceilings, scaled rather than truncated.

        Scaling the whole vector keeps the direction the policy asked for.
        Clipping each axis on its own bends the movement towards the box corners,
        which is a silent change of direction rather than of speed.

        The ceilings are one control period at the workspace speed limits, so a
        policy that saturates this clamp on every step is moving at exactly the
        speed Architecture A is allowed to move at, and no faster.
        """
        out = np.asarray(action, dtype=float).reshape(-1).copy()

        for lo, hi, limit, what in ((0, 3, self.max_step_m, "translation"),
                                    (3, 6, self.max_step_rad, "rotation")):
            norm = float(np.linalg.norm(out[lo:hi]))
            if norm > limit > 0:
                out[lo:hi] *= limit / norm
                logger.warning("clamped %s step from %.4f to %.4f", what, norm, limit)

        return out

    def gripper(self, value: float) -> int:
        """It takes the policy's gripper output and gives you 0 or 1."""
        return 1 if float(value) >= self.gripper_threshold else 0

    def translate(self, current_pose, action: dict | np.ndarray) -> tuple:
        """
        It takes the current TCP pose and one policy action, and gives you
        (target_pose, gripper_state, clamped_action).

        Raises `SafetyViolation` when the target would leave the workspace, which
        the caller must handle by stopping the arm rather than by skipping the
        step. A skipped step at 20 Hz means the arm keeps tracking its last
        target with nobody watching.

        `clamped_action` comes back so the caller can report what was really
        sent. Reporting the request instead would hide every clamp that fired.
        """
        if isinstance(action, dict):
            action = [float(action[name]) for name in ACTION_NAMES]

        clamped = self.clamp(action)
        target = apply_delta_action(np.asarray(current_pose, dtype=float), clamped)
        self.env.check_pose(target, "policy action")
        return target, self.gripper(clamped[6]), clamped
