"""ur5e.py

It takes a policy action and gives you a UR5e that moves, and it takes the arm
and camera and gives you the observation a policy expects.

A THIN SHELL, on purpose. Everything that decides where the arm goes lives in
`fyp.architecture_b.policy_action`: the clamp, the delta arithmetic, the envelope
check and the gripper threshold. This file only adds the LeRobot interface and
the hardware calls.

The reason for the split is testability. This module imports `lerobot`, which is
deliberately absent from the FYP venv, so nothing in here can be reached from
`tests/unit/`. Keeping the decisions out of it means the decisions are tested and
only the plumbing is not.

If you are about to add a number to this file, put it in `config.yaml` and read
it in the translator instead. config.yaml is the single source of truth for
tunables, and a constant here is invisible to every test.
"""

from __future__ import annotations

import logging
from functools import cached_property
from typing import Any

import numpy as np

from lerobot.robots import Robot

from lerobot_robot_ur5e.config import UR5eConfig

logger = logging.getLogger(__name__)


class UR5eRobot(Robot):
    """
    It takes a UR5eConfig and gives you a LeRobot robot backed by the real arm.

    Built but not connected. `connect` opens the arm and the camera, and the
    feature properties work before that, which LeRobot requires so it can
    describe the robot with no hardware present.
    """

    config_class = UR5eConfig
    name = "ur5e"

    def __init__(self, config: UR5eConfig):
        super().__init__(config)
        self.config = config

        from fyp.architecture_b.policy_action import PolicyActionTranslator
        from fyp.shared.helpers.config import get_config

        # Read at construction, not at connect, because the feature properties
        # have to answer before any hardware is open.
        self._cfg = get_config(config.fyp_config) if config.fyp_config else get_config()
        self._t = PolicyActionTranslator(self._cfg)

        self._controller = None
        self._camera = None
        self._gripper_commanded: int | None = None

    # ------------------------------------------------------------- features

    @cached_property
    def observation_features(self) -> dict[str, type | tuple]:
        return self._t.observation_features

    @cached_property
    def action_features(self) -> dict[str, type]:
        return self._t.action_features

    # ------------------------------------------------------------ lifecycle

    @property
    def is_connected(self) -> bool:
        return getattr(self, "_controller", None) is not None and \
            getattr(self, "_camera", None) is not None

    @property
    def is_calibrated(self) -> bool:
        # A UR5e is calibrated at its own controller and reports poses in the
        # base frame already. There is nothing for LeRobot to calibrate, and the
        # base class asks for True in exactly that case.
        return True

    def calibrate(self) -> None:
        """No-op. See `is_calibrated`."""

    def configure(self) -> None:
        """
        No-op. Servo gains, speed ceilings and the workspace box all come from
        config.yaml and are applied per call, so there is no one-time
        configuration to push to the arm.
        """

    def connect(self, calibrate: bool = True) -> None:
        from fyp.shared.hardware.camera import RealSenseCamera
        from fyp.shared.hardware.ur5e_controller import URController

        if self.is_connected:
            raise RuntimeError(f"{self} is already connected")

        self._controller = URController()
        self._controller.gripper_start()

        self._camera = RealSenseCamera(self._cfg)
        self._camera.start()

        # Whatever the gripper is doing now is what to compare against, so the
        # first action does not re-command a gripper already in that state and
        # pay the actuation delay for nothing.
        self._gripper_commanded = int(self._controller.get_state()["gripper_state"])
        logger.info("UR5e connected, envelope: %s", self._t.env.describe())

    def disconnect(self) -> None:
        """
        It takes nothing and gives you a stopped arm and a released camera.

        `servo_stop` first and unconditionally. The controller keeps driving
        towards the last servo target until told otherwise, so an episode that
        ends by closing the socket leaves the arm creeping.
        """
        if self._controller is not None:
            try:
                self._controller.servo_stop()
            except Exception:
                logger.exception("servo_stop failed during disconnect")
            self._controller.close()
            self._controller = None

        if self._camera is not None:
            self._camera.stop()
            self._camera = None

    # --------------------------------------------------------------- the loop

    def get_observation(self) -> dict[str, Any]:
        """It takes nothing and gives you one observation from the arm and camera."""
        if not self.is_connected:
            raise RuntimeError(f"{self} is not connected")

        state = self._controller.get_state()
        # Colour only. SmolVLA has no depth input, so the depth channel is
        # captured and dropped rather than not captured, since the camera
        # delivers them together anyway.
        return self._t.observation(state["tcp_pose"], int(state["gripper_state"]),
                                   self._camera.capture().rgb)

    def send_action(self, action: dict[str, Any]) -> dict[str, Any]:
        """
        It takes one delta action from the policy and gives you back the action
        that was actually sent, after clamping.

        All the arithmetic and the safety check happen in the translator. What is
        left here is the part that cannot be unit tested: talking to the arm, and
        stopping it before an exception unwinds.
        """
        from fyp.architecture_b.demos.lerobot_export import ACTION_NAMES
        from fyp.shared.hardware.safety import SafetyViolation

        if not self.is_connected:
            raise RuntimeError(f"{self} is not connected")

        current = self._controller.get_state()["tcp_pose"]
        try:
            target, gripper, clamped = self._t.translate(current, action)
        except SafetyViolation:
            # Stop before raising. The exception unwinds through robot_client,
            # and an arm left in an open servo stream keeps moving while it does.
            self._controller.servo_stop()
            raise

        self._controller.servo_to_pose(target)

        # Only on a change of decision. `gripper_toggle` sleeps 0.1 s, which is
        # two whole control periods at 20 Hz, so re-commanding an already open
        # gripper would stall the loop on every step and the arm would stutter.
        if gripper != self._gripper_commanded:
            self._controller.gripper_toggle(gripper)
            self._gripper_commanded = gripper

        return dict(zip(ACTION_NAMES, (float(v) for v in clamped)))
